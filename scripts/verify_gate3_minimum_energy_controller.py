#!/usr/bin/env python3
"""Verify N182's bounded minimum-energy Gate-3 lateral controller."""

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


EXPECTED_CHANGES = {
    "016": 3,
    "018": 5,
    "020": 5,
    "021": 2,
    "022": 2,
    "030": 3,
    "031": 4,
    "034": 4,
    "037": 2,
    "038": 4,
    "039": 5,
    "040": 4,
    "041": 4,
    "042": 2,
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
            hybrid._gate3_minimum_energy_lateral_controller(
                values, recorded.tolist()
            ),
            dtype=float,
        )
        if governed.tolist() == recorded.tolist():
            continue
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        confidence = float(values[30])
        leveled = forward_m <= hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
        horizon_s = None
        lateral_accel_m_s2 = 0.0
        uncapped_roll_rad = 0.0
        if not leveled:
            horizon_s = max(
                (
                    forward_m
                    - hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
                )
                / closing_m_s,
                hybrid.GATE3_MINIMUM_ENERGY_MIN_HORIZON_S,
            )
            lateral_accel_m_s2 = (
                6.0
                * (hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_M - right_m)
                / horizon_s**2
                - (
                    4.0 * right_rate_m_s
                    + 2.0
                    * hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_RATE_M_S
                )
                / horizon_s
            )
            uncapped_roll_rad = math.atan(
                lateral_accel_m_s2
                / hybrid.GATE3_MINIMUM_ENERGY_GRAVITY_M_S2
            )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "closing_m_s": closing_m_s,
                "confidence": confidence,
                "leveled_at_handoff": leveled,
                "horizon_s": horizon_s,
                "lateral_accel_m_s2": lateral_accel_m_s2,
                "uncapped_roll_rad": uncapped_roll_rad,
                "roll_cap_active": abs(uncapped_roll_rad)
                > hybrid.GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD,
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
        "positive_roll_samples": sum(
            row["governed_action"][1] > 0.0 for row in rows
        ),
        "negative_roll_samples": sum(
            row["governed_action"][1] < 0.0 for row in rows
        ),
        "level_roll_samples": sum(
            row["governed_action"][1] == 0.0 for row in rows
        ),
        "max_non_roll_error": max(
            (
                max(
                    abs(
                        row["governed_action"][index]
                        - row["recorded_action"][index]
                    )
                    for index in (0, 2, 3)
                )
                for row in rows
            ),
            default=0.0,
        ),
        "invalid_roll_outputs": sum(
            not (
                math.isfinite(row["governed_action"][1])
                and -0.5 <= row["governed_action"][1] <= 0.5
            )
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate042-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate, expected in EXPECTED_CHANGES.items():
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["changed_samples"] != expected:
            blockers.append(f"candidate{candidate}_change_count_not_{expected}")

    for report in reports:
        candidate = report["candidate"]
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["invalid_roll_outputs"]:
            blockers.append(f"candidate{candidate}_invalid_roll_output")
        for row in report["rows"]:
            if row["confidence"] < hybrid.GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE:
                blockers.append(f"candidate{candidate}_low_confidence_change")
            if row["leveled_at_handoff"] and row["governed_action"][1] != 0.0:
                blockers.append(f"candidate{candidate}_handoff_not_level")

    clean021 = by_candidate.get("021") or {}
    failed042 = by_candidate.get("042") or {}
    if not any(
        row["forward_m"] > hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
        and row["governed_action"][1] > 0.0
        for row in clean021.get("rows", [])
    ):
        blockers.append("candidate021_missing_positive_steering")
    if not any(
        row["forward_m"] > hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
        and row["governed_action"][1] < 0.0
        for row in failed042.get("rows", [])
    ):
        blockers.append("candidate042_missing_negative_steering")
    if not any(
        row["leveled_at_handoff"] for row in failed042.get("rows", [])
    ):
        blockers.append("candidate042_missing_level_handoff")

    poststop = json.loads(args.candidate042_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate042_poststop_not_command_free")
    if int(race.get("race_start_boot_time_ms", -2)) != -1:
        blockers.append("candidate042_poststop_race_not_inactive")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate042_poststop_finish_unexpected")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate042_poststop_not_disarmed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "objective": "minimum integral lateral acceleration squared",
            "dynamics": "right_dot=right_rate; right_rate_dot=accel",
            "first_acceleration": "6*(target_right-right)/T^2-(4*right_rate+2*target_rate)/T",
            "handoff_forward_m": hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M,
            "target_right_m": hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_M,
            "target_right_rate_m_s": hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_RATE_M_S,
            "min_confidence": hybrid.GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE,
            "min_horizon_s": hybrid.GATE3_MINIMUM_ENERGY_MIN_HORIZON_S,
            "max_roll_rad": hybrid.GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD,
            "policy_max_roll_rad": hybrid.GATE3_POLICY_MAX_ROLL_RAD,
            "preserve_pitch_thrust_yaw": True,
        },
        "candidate042_poststop": {
            "path": str(args.candidate042_poststop),
            "sha256": _sha256(args.candidate042_poststop),
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
