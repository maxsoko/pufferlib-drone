#!/usr/bin/env python3
"""Verify the N177 strong extended Gate-3 lateral-rate brake on live traces."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "030": 2,
    "031": 2,
    "034": 1,
    "037": 2,
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
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        time_to_terminal_s = (
            max(forward_m - 2.0, 0.0) / closing_m_s
            if closing_m_s >= 1.0
            else 0.0
        )
        projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
        governed = np.asarray(
            hybrid._gate3_late_lateral_rate_brake(values, recorded.tolist()),
            dtype=float,
        )
        if governed.tolist() == recorded.tolist():
            continue
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "projected_right_at_terminal_m": projected_right_m,
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
        "roll_output_violations": sum(
            row["governed_action"][1]
            != hybrid.GATE3_LATE_RATE_BRAKE_ROLL_NORM
            for row in rows
        ),
        "positive_roll_changes": sum(
            row["recorded_action"][1] > 0.0 for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate037-poststop", required=True, type=Path)
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
            blockers.append(
                f"candidate{candidate}_change_count_not_{expected}"
            )

    candidate37 = by_candidate.get("037")
    if candidate37 is not None and candidate37["rows"]:
        ranges = [row["forward_m"] for row in candidate37["rows"]]
        if not (
            hybrid.GATE3_LATE_RATE_BRAKE_MIN_FORWARD_M < min(ranges) < 3.0
            and 3.0 < max(ranges) <= hybrid.GATE3_LATE_RATE_BRAKE_MAX_FORWARD_M
        ):
            blockers.append("candidate037_does_not_span_extended_brake_window")

    for report in reports:
        candidate = report["candidate"]
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["roll_output_violations"]:
            blockers.append(f"candidate{candidate}_roll_output_violation")
        if report["positive_roll_changes"]:
            blockers.append(f"candidate{candidate}_changes_positive_roll")

    poststop = json.loads(args.candidate037_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate037_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 2:
        blockers.append("candidate037_poststop_gate_index_not_two")
    if int(race.get("last_gate_race_time", -1)) != 7647600650:
        blockers.append("candidate037_poststop_gate_time_mismatch")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate037_poststop_finish_unexpected")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate037_poststop_not_disarmed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "min_forward_m_exclusive": (
                hybrid.GATE3_LATE_RATE_BRAKE_MIN_FORWARD_M
            ),
            "max_forward_m": hybrid.GATE3_LATE_RATE_BRAKE_MAX_FORWARD_M,
            "min_right_rate_m_s": (
                hybrid.GATE3_LATE_RATE_BRAKE_MIN_RIGHT_RATE_M_S
            ),
            "max_abs_projected_right_m": (
                hybrid.GATE3_LATE_RATE_BRAKE_MAX_ABS_PROJECTED_RIGHT_M
            ),
            "normalized_roll": hybrid.GATE3_LATE_RATE_BRAKE_ROLL_NORM,
            "left_side_only": True,
            "preserve_pitch_thrust_yaw_positive_and_stronger_negative_roll": True,
        },
        "candidate037_poststop": {
            "path": str(args.candidate037_poststop),
            "sha256": _sha256(args.candidate037_poststop),
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
