#!/usr/bin/env python3
"""Verify N174's confidence-gated late Gate-3 positive-residual roll floor."""

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


ARCHIVED_CANDIDATES = {f"{candidate:03d}" for candidate in range(13, 35)}
CLEAN_GATE3_CANDIDATES = {"016", "018", "020", "021", "022"}


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
    low_confidence_geometric_triggers = 0
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
        confidence = float(values[30])
        time_to_terminal_s = (
            max(forward_m - 2.0, 0.0) / closing_m_s
            if closing_m_s >= 1.0
            else 0.0
        )
        projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
        geometric_trigger = (
            float(values[10]) >= 0.5
            and 0.0
            < forward_m
            <= hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MAX_FORWARD_M
            and closing_m_s >= 1.0
            and right_m >= hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_M
            and right_rate_m_s
            >= hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_RATE_M_S
            and projected_right_m
            > hybrid.GATE3_LATE_POSITIVE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M
        )
        if (
            geometric_trigger
            and confidence < hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_CONFIDENCE
        ):
            low_confidence_geometric_triggers += 1
        governed = np.asarray(
            hybrid._gate3_late_positive_residual_roll_floor(
                values, recorded.tolist()
            ),
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
                "current_frame_confidence": confidence,
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
        "changed_samples": len(rows),
        "low_confidence_geometric_triggers": low_confidence_geometric_triggers,
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
        "roll_floor_violations": sum(
            row["governed_action"][1]
            != hybrid.GATE3_LATE_POSITIVE_RESIDUAL_ROLL_FLOOR_NORM
            for row in rows
        ),
        "stronger_roll_changes": sum(
            row["recorded_action"][1]
            <= hybrid.GATE3_LATE_POSITIVE_RESIDUAL_ROLL_FLOOR_NORM
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate034-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate in sorted(ARCHIVED_CANDIDATES):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif candidate == "034":
            if report["changed_samples"] != 3:
                blockers.append("candidate034_change_count_not_three")
            if report["low_confidence_geometric_triggers"] < 1:
                blockers.append("candidate034_stale_confidence_guard_unexercised")
        elif report["changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_changed")
    for candidate in sorted(CLEAN_GATE3_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and report["changed_samples"]:
            blockers.append(f"candidate{candidate}_clean_path_changed")
    for report in reports:
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changes_non_roll")
        if report["roll_floor_violations"]:
            blockers.append(f"candidate{report['candidate']}_roll_floor_violation")
        if report["stronger_roll_changes"]:
            blockers.append(f"candidate{report['candidate']}_changes_stronger_roll")

    poststop = json.loads(args.candidate034_poststop.read_text(encoding="utf-8"))
    race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate034_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 2:
        blockers.append("candidate034_poststop_gate_index_unexpected")
    if int(race.get("last_gate_race_time", -1)) != 7677293777:
        blockers.append("candidate034_poststop_last_gate_time_unexpected")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate034_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "max_forward_m": (
                hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MAX_FORWARD_M
            ),
            "min_current_frame_confidence": (
                hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_CONFIDENCE
            ),
            "min_right_m": hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_M,
            "min_right_rate_m_s": (
                hybrid.GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_RATE_M_S
            ),
            "projected_right_trigger_m": (
                hybrid.GATE3_LATE_POSITIVE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M
            ),
            "normalized_roll_floor": (
                hybrid.GATE3_LATE_POSITIVE_RESIDUAL_ROLL_FLOOR_NORM
            ),
            "preserve_pitch_thrust_yaw_and_stronger_roll": True,
        },
        "clean_gate3_candidates": sorted(CLEAN_GATE3_CANDIDATES),
        "candidate034_poststop": {
            "path": str(args.candidate034_poststop),
            "sha256": _sha256(args.candidate034_poststop),
            "reset_sent": poststop.get("reset_sent"),
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
