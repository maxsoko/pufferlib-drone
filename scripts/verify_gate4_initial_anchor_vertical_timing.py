#!/usr/bin/env python3
"""Verify N158 Gate-4 initial-anchor admission and vertical timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import policy_callable_six_gate_hybrid as hybrid
    from verify_gate4_safe_handoff import analyze
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts.verify_gate4_safe_handoff import analyze


def _is_fail_safe(row: dict) -> bool:
    action = row["governed_action"]
    return (
        abs(action[0] + hybrid.GATE4_MAX_PITCH_RAD / 0.5) <= 1e-8
        and abs(action[1]) <= 1e-8
        and abs(action[2]) <= 1e-8
    )


def _is_full_descent(action: list[float]) -> bool:
    return abs(action[2] - hybrid._encode_gate4_thrust(0.22)) <= 1e-7


def _first(rows: list[dict], predicate) -> dict | None:
    return next((row for row in rows if predicate(row)), None)


def _summary(analysis: dict) -> dict:
    rows = analysis["rows"]
    visible = [row for row in rows if row["raw_gate_vector_m"] is not None]
    initial_fail_safe = [
        row
        for row in visible
        if (
            row["raw_gate_vector_m"][0] > hybrid.GATE4_MAX_INITIAL_RANGE_M
            or abs(row["raw_gate_vector_m"][1])
            > hybrid.GATE4_INITIAL_MAX_ABS_RIGHT_M
        )
        and _is_fail_safe(row)
    ]
    first_governed = _first(visible, lambda row: not _is_fail_safe(row))
    first_descent = _first(
        visible, lambda row: _is_full_descent(row["governed_action"])
    )
    recorded_first_descent = _first(
        visible, lambda row: _is_full_descent(row["recorded_action"])
    )
    return {
        "path": analysis["path"],
        "sha256": analysis["sha256"],
        "official_active_gate_index": analysis["official_active_gate_index"],
        "collision": analysis["collision"],
        "gate4_samples": analysis["gate4_samples"],
        "visible_samples": analysis["visible_samples"],
        "dropout_samples": analysis["dropout_samples"],
        "initial_fail_safe_count": len(initial_fail_safe),
        "first_governed": first_governed,
        "first_full_descent": first_descent,
        "recorded_first_full_descent": recorded_first_descent,
        "descent_lead_s": (
            None
            if first_descent is None or recorded_first_descent is None
            else recorded_first_descent["elapsed_s"] - first_descent["elapsed_s"]
        ),
        "final_row": rows[-1],
        "final_reanchor_count": analysis["final_reanchor_count"],
        "max_abs_governed_roll": analysis["max_abs_governed_roll"],
        "min_governed_thrust": analysis["min_governed_thrust"],
        "dropout_max_abs_roll": analysis["dropout_max_abs_roll"],
        "dropout_max_abs_thrust": analysis["dropout_max_abs_thrust"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("candidate18", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    clean = _summary(analyze(args.candidate16))
    failed = _summary(analyze(args.candidate18))
    blockers: list[str] = []
    for name, evidence in (("candidate16", clean), ("candidate18", failed)):
        if evidence["official_active_gate_index"] != 3:
            blockers.append(f"{name}_did_not_reach_gate4")
        if evidence["collision"]["id"] != 1002:
            blockers.append(f"{name}_failure_family_changed")
        if evidence["max_abs_governed_roll"] > 0.200001:
            blockers.append(f"{name}_roll_bound_exceeded")
        if (
            evidence["min_governed_thrust"]
            < hybrid._encode_gate4_thrust(0.22) - 1e-8
        ):
            blockers.append(f"{name}_thrust_floor_exceeded")
        if evidence["dropout_max_abs_roll"] != 0.0:
            blockers.append(f"{name}_dropout_roll_not_level")
        if evidence["dropout_max_abs_thrust"] != 0.0:
            blockers.append(f"{name}_dropout_thrust_not_hover")

    clean_first = clean["first_governed"]
    if clean_first is None or abs(clean_first["raw_gate_vector_m"][1]) > 4.5:
        blockers.append("candidate16_initial_anchor_not_centered")
    if clean["final_reanchor_count"] != hybrid.GATE4_REANCHOR_MAX_COUNT:
        blockers.append("candidate16_reanchor_coverage_changed")

    failed_first = failed["first_governed"]
    if failed["initial_fail_safe_count"] < 5:
        blockers.append("candidate18_false_initial_family_not_rejected")
    if failed_first is None or failed_first["elapsed_s"] != 12.656:
        blockers.append("candidate18_centered_family_not_first_anchor")
    elif abs(failed_first["raw_gate_vector_m"][1]) > 4.5:
        blockers.append("candidate18_first_anchor_outside_lateral_bound")
    if failed["descent_lead_s"] is None or failed["descent_lead_s"] < 1.0:
        blockers.append("candidate18_vertical_correction_not_advanced")
    if not _is_full_descent(failed["final_row"]["governed_action"]):
        blockers.append("candidate18_final_visible_sample_not_descending")
    expected_final_roll = hybrid.tail._clamp(
        -hybrid.GATE4_LATERAL_ROLL_KP
        * failed["final_row"]["raw_gate_vector_m"][1]
        / 0.5
    )
    if abs(failed["final_row"]["governed_action"][1] - expected_final_roll) > 1e-6:
        blockers.append("candidate18_final_visible_sample_not_associated")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "change": {
            "gate_index": 3,
            "initial_max_abs_right_m": hybrid.GATE4_INITIAL_MAX_ABS_RIGHT_M,
            "association_grace_s": hybrid.GATE4_ASSOCIATION_GRACE_S,
            "vertical_control_delay_s": hybrid.GATE4_VERTICAL_CONTROL_DELAY_S,
            "reanchor_delay_s": hybrid.GATE4_DESCENT_DELAY_S,
            "reanchor_max_abs_right_m": hybrid.GATE4_REANCHOR_MAX_ABS_RIGHT_M,
            "roll_kp": hybrid.GATE4_LATERAL_ROLL_KP,
            "max_roll_rad": hybrid.GATE4_MAX_LATERAL_ROLL_RAD,
            "descent_thrust": hybrid.GATE4_DESCENT_THRUST,
        },
        "candidate16_clean_gate4": clean,
        "candidate18_gate4_collision": failed,
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
