#!/usr/bin/env python3
"""Verify N179's separate high-rate Gate-3 margin-brake tier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import policy_callable_six_gate_hybrid as hybrid
    from verify_gate3_strong_extended_lateral_rate_brake import analyze
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts.verify_gate3_strong_extended_lateral_rate_brake import analyze


PASSING_CANDIDATES = {"016", "018", "020", "021", "022"}
EXPECTED_FAILURE_CHANGES = {
    "030": 3,
    "031": 2,
    "034": 2,
    "037": 2,
    "038": 2,
    "039": 3,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate039-poststop", required=True, type=Path)
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

    candidate39 = by_candidate.get("039")
    if candidate39 is not None:
        rows = candidate39["rows"]
        if not (
            len(rows) == 3
            and all(
                hybrid.GATE3_LATE_RATE_BRAKE_MIN_FORWARD_M
                < row["forward_m"]
                <= hybrid.GATE3_LATE_RATE_BRAKE_MAX_FORWARD_M
                and row["right_rate_m_s"]
                >= hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MIN_RIGHT_M_S
                and abs(row["projected_right_at_terminal_m"])
                <= hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MAX_ABS_PROJECTED_RIGHT_M
                and row["recorded_action"][1]
                == hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM
                for row in rows
            )
            and any(
                row["forward_m"]
                > hybrid.GATE3_LATE_RATE_BRAKE_ORDINARY_MAX_FORWARD_M
                for row in rows
            )
        ):
            blockers.append("candidate039_high_rate_margin_samples_unexpected")

    for report in reports:
        candidate = report["candidate"]
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["roll_output_violations"]:
            blockers.append(f"candidate{candidate}_roll_output_violation")
        expected_positive = 3 if candidate == "039" else (1 if candidate == "038" else 0)
        if report["positive_roll_changes"] != expected_positive:
            blockers.append(f"candidate{candidate}_positive_roll_change_unexpected")
        for row in report["rows"]:
            if row["recorded_action"][1] > 0.0 and (
                row["recorded_action"][1]
                != hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM
            ):
                blockers.append(f"candidate{candidate}_overrides_nonmargin_positive_roll")

    poststop = json.loads(args.candidate039_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate039_poststop_not_command_free")
    if int(race.get("race_start_boot_time_ms", -2)) != -1:
        blockers.append("candidate039_poststop_race_not_inactive")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate039_poststop_finish_unexpected")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate039_poststop_not_disarmed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "ordinary": {
                "max_forward_m": hybrid.GATE3_LATE_RATE_BRAKE_ORDINARY_MAX_FORWARD_M,
                "min_right_rate_m_s": hybrid.GATE3_LATE_RATE_BRAKE_MIN_RIGHT_RATE_M_S,
                "max_abs_projected_right_m": hybrid.GATE3_LATE_RATE_BRAKE_MAX_ABS_PROJECTED_RIGHT_M,
            },
            "high_rate": {
                "max_forward_m": hybrid.GATE3_LATE_RATE_BRAKE_MAX_FORWARD_M,
                "min_right_rate_m_s": hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MIN_RIGHT_M_S,
                "max_abs_projected_right_m": hybrid.GATE3_LATE_RATE_BRAKE_HIGH_RATE_MAX_ABS_PROJECTED_RIGHT_M,
            },
            "min_forward_m_exclusive": hybrid.GATE3_LATE_RATE_BRAKE_MIN_FORWARD_M,
            "normalized_roll": hybrid.GATE3_LATE_RATE_BRAKE_ROLL_NORM,
            "override_exact_terminal_margin_floor": hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM,
            "preserve_pitch_thrust_yaw_and_other_positive_roll": True,
        },
        "candidate039_poststop": {
            "path": str(args.candidate039_poststop),
            "sha256": _sha256(args.candidate039_poststop),
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
