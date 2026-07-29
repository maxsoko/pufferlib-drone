#!/usr/bin/env python3
"""Verify N181's continuous physics-based Gate-3 stopping brake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
    import policy_callable_six_gate_composite as tail
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts import policy_callable_six_gate_composite as tail


PASSING_CANDIDATES = {"016", "018", "020", "021", "022"}
EXPECTED_FAILURE_CHANGES = {
    "030": 3,
    "031": 3,
    "034": 2,
    "037": 2,
    "038": 4,
    "039": 7,
    "040": 2,
    "041": 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        governed = np.asarray(
            hybrid._gate3_late_lateral_rate_brake(values, recorded.tolist()),
            dtype=float,
        )
        if governed.tolist() == recorded.tolist():
            continue
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        time_to_terminal_s = max(forward_m - 2.0, 0.0) / closing_m_s
        projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
        stopping_distance_m = max(
            -right_m,
            hybrid.GATE3_LATE_RATE_BRAKE_MIN_STOPPING_DISTANCE_M,
        )
        stopping_accel_m_s2 = right_rate_m_s**2 / (2.0 * stopping_distance_m)
        uncapped_roll_rad = -math.atan(
            stopping_accel_m_s2 / hybrid.GATE3_LATE_RATE_BRAKE_GRAVITY_M_S2
        )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "projected_right_at_terminal_m": projected_right_m,
                "stopping_distance_m": stopping_distance_m,
                "stopping_accel_m_s2": stopping_accel_m_s2,
                "uncapped_roll_rad": uncapped_roll_rad,
                "roll_cap_active": (
                    uncapped_roll_rad
                    < -hybrid.GATE3_LATE_RATE_BRAKE_MAX_ROLL_RAD
                ),
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            }
        )

    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": report.get(
            "official_active_gate_index"
        ),
        "collision_id": latest.get("collision_id"),
        "collision_threat_level": latest.get("collision_threat_level"),
        "collision_impact": latest.get("collision_impact"),
        "changed_samples": len(rows),
        "max_non_roll_error": max(
            (
                max(
                    abs(row["governed_action"][index] - row["recorded_action"][index])
                    for index in (0, 2, 3)
                )
                for row in rows
            ),
            default=0.0,
        ),
        "invalid_roll_outputs": sum(
            not (
                math.isfinite(row["governed_action"][1])
                and -0.5 <= row["governed_action"][1] < 0.0
            )
            for row in rows
        ),
        "nonmargin_positive_roll_changes": sum(
            row["recorded_action"][1] > 0.0
            and row["recorded_action"][1]
            != hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate041-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []

    for candidate in sorted(PASSING_CANDIDATES):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_clean_path_changed")

    for candidate, expected in EXPECTED_FAILURE_CHANGES.items():
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["changed_samples"] != expected:
            blockers.append(f"candidate{candidate}_change_count_not_{expected}")

    for candidate in ("038", "039", "041"):
        report = by_candidate.get(candidate)
        if report is not None and not all(
            -0.5 <= row["governed_action"][1] < 0.0 for row in report["rows"]
        ):
            blockers.append(f"candidate{candidate}_stopping_roll_not_moderate")

    for report in reports:
        candidate = report["candidate"]
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["invalid_roll_outputs"]:
            blockers.append(f"candidate{candidate}_invalid_roll_output")
        if report["nonmargin_positive_roll_changes"]:
            blockers.append(f"candidate{candidate}_changes_protected_positive_roll")

    poststop = json.loads(args.candidate041_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate041_poststop_not_command_free")
    if int(race.get("race_start_boot_time_ms", -2)) != -1:
        blockers.append("candidate041_poststop_race_not_inactive")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate041_poststop_finish_unexpected")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate041_poststop_not_disarmed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "formula": "roll=clamp(-atan((right_rate^2/(2*max(-right,min_distance)))/g),-max_roll,0)/policy_max_roll",
            "gravity_m_s2": hybrid.GATE3_LATE_RATE_BRAKE_GRAVITY_M_S2,
            "min_stopping_distance_m": hybrid.GATE3_LATE_RATE_BRAKE_MIN_STOPPING_DISTANCE_M,
            "max_brake_roll_rad": hybrid.GATE3_LATE_RATE_BRAKE_MAX_ROLL_RAD,
            "policy_max_roll_rad": hybrid.GATE3_POLICY_MAX_ROLL_RAD,
            "max_normalized_brake_magnitude": (
                hybrid.GATE3_LATE_RATE_BRAKE_MAX_ROLL_RAD
                / hybrid.GATE3_POLICY_MAX_ROLL_RAD
            ),
            "ordinary_max_forward_m": hybrid.GATE3_LATE_RATE_BRAKE_ORDINARY_MAX_FORWARD_M,
            "ordinary_min_right_rate_m_s": hybrid.GATE3_LATE_RATE_BRAKE_MIN_RIGHT_RATE_M_S,
            "ordinary_max_abs_projected_right_m": hybrid.GATE3_LATE_RATE_BRAKE_MAX_ABS_PROJECTED_RIGHT_M,
            "high_rate_max_forward_m": hybrid.GATE3_LATE_RATE_BRAKE_MAX_FORWARD_M,
            "high_rate_min_right_rate_m_s": hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MIN_RIGHT_M_S,
            "high_rate_max_abs_projected_right_m": hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MAX_ABS_PROJECTED_RIGHT_M,
            "preserve_pitch_thrust_yaw_and_nonmargin_positive_roll": True,
        },
        "candidate041_poststop": {
            "path": str(args.candidate041_poststop),
            "sha256": _sha256(args.candidate041_poststop),
            "reset_sent": poststop.get("reset_sent"),
            "base_mode": latest.get("base_mode"),
            "system_status": latest.get("system_status"),
            "race_status": race,
        },
        "reports": reports,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
