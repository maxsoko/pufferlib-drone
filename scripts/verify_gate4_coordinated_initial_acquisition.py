#!/usr/bin/env python3
"""Verify N161 bounded roll+yaw acquisition before Gate 4's first anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
    from verify_gate4_safe_handoff import _raw_observation
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts.verify_gate4_safe_handoff import _raw_observation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in ((report.get("policy_trace") or {}).get("samples") or [])
        if int(sample.get("official_active_gate_index", -1)) == 3
    ]
    first_elapsed_s = float(samples[0]["elapsed_s"])
    hybrid._clear_gate4_state()
    rows: list[dict] = []
    first_anchor_elapsed_s: float | None = None
    for sample in samples:
        values = _raw_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        elapsed_s = float(sample["elapsed_s"])
        phase_elapsed_s = elapsed_s - first_elapsed_s
        anchor_absent_before = hybrid._GATE4_ANCHOR_VECTOR is None
        current_yaw_rad = hybrid._yaw_from_quaternion(values)
        governed = np.asarray(
            hybrid._gate4_safe_handoff(
                values,
                recorded.tolist(),
                now_s=100.0 + phase_elapsed_s,
            ),
            dtype=float,
        )
        anchor_absent_after = hybrid._GATE4_ANCHOR_VECTOR is None
        if first_anchor_elapsed_s is None and not anchor_absent_after:
            first_anchor_elapsed_s = elapsed_s
        visible = float(values[10]) > 0.5
        coordinated = bool(
            anchor_absent_before
            and anchor_absent_after
            and visible
            and phase_elapsed_s >= hybrid.GATE4_ACQUISITION_YAW_DELAY_S
        )
        right_m = (
            hybrid.tail._inverse_tanh_norm(float(values[12]), 5.0)
            if visible
            else None
        )
        yaw_error_rad = (
            hybrid.tail._clamp(float(values[14])) * (math.pi / 4.0)
            if visible
            else None
        )
        expected_roll_norm = (
            None
            if not coordinated
            else hybrid.tail._clamp(
                hybrid.tail._clamp(
                    -hybrid.GATE4_LATERAL_ROLL_KP * float(right_m),
                    -hybrid.GATE4_MAX_LATERAL_ROLL_RAD,
                    hybrid.GATE4_MAX_LATERAL_ROLL_RAD,
                )
                / 0.5
            )
        )
        expected_yaw_step_rad = (
            None
            if not coordinated
            else hybrid.tail._clamp(
                float(yaw_error_rad),
                -hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
                hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
            )
        )
        actual_yaw_step_rad = hybrid._wrap_angle(
            current_yaw_rad - float(governed[3]) * math.pi
        )
        pose = sample.get("gate_pose")
        rows.append(
            {
                "elapsed_s": elapsed_s,
                "phase_elapsed_s": phase_elapsed_s,
                "raw_gate_vector_m": (
                    None if not pose else pose["body_vector_ned_m"]
                ),
                "anchor_absent_before": anchor_absent_before,
                "anchor_absent_after": anchor_absent_after,
                "coordinated_acquisition": coordinated,
                "expected_roll_norm": expected_roll_norm,
                "expected_yaw_step_rad": expected_yaw_step_rad,
                "actual_yaw_step_rad": actual_yaw_step_rad,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
                "reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
            }
        )

    coordinated_rows = [row for row in rows if row["coordinated_acquisition"]]
    dropout_rows = [row for row in rows if row["raw_gate_vector_m"] is None]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision": {
            "id": ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                "collision_id"
            ),
            "threat_level": (
                ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                    "collision_threat_level"
                )
            ),
            "impact": (
                ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                    "collision_impact"
                )
            ),
        },
        "gate4_samples": len(rows),
        "visible_samples": sum(
            row["raw_gate_vector_m"] is not None for row in rows
        ),
        "dropout_samples": len(dropout_rows),
        "coordinated_acquisition_count": len(coordinated_rows),
        "coordinated_pitch_or_thrust_violation_count": sum(
            abs(row["governed_action"][0] + hybrid.GATE4_MAX_PITCH_RAD / 0.5)
            > 1e-8
            or abs(row["governed_action"][2]) > 1e-8
            for row in coordinated_rows
        ),
        "coordinated_roll_mismatch_count": sum(
            abs(row["governed_action"][1] - row["expected_roll_norm"]) > 1e-7
            for row in coordinated_rows
        ),
        "coordinated_yaw_mismatch_count": sum(
            abs(
                hybrid._wrap_angle(
                    row["actual_yaw_step_rad"] - row["expected_yaw_step_rad"]
                )
            )
            > 1e-7
            for row in coordinated_rows
        ),
        "max_abs_coordinated_roll_norm": max(
            (abs(row["governed_action"][1]) for row in coordinated_rows),
            default=0.0,
        ),
        "dropout_roll_or_thrust_violation_count": sum(
            abs(row["governed_action"][1]) > 1e-8
            or abs(row["governed_action"][2]) > 1e-8
            for row in dropout_rows
        ),
        "dropout_yaw_non_hold_count": sum(
            abs(row["actual_yaw_step_rad"]) > 1e-7 for row in dropout_rows
        ),
        "first_anchor_elapsed_s": first_anchor_elapsed_s,
        "final_reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("candidate18", type=Path)
    parser.add_argument("candidate20", type=Path)
    parser.add_argument("candidate21", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    analyses = {
        "candidate16": _analyze(args.candidate16),
        "candidate18": _analyze(args.candidate18),
        "candidate20": _analyze(args.candidate20),
        "candidate21": _analyze(args.candidate21),
    }
    minimum_counts = {
        "candidate16": 10,
        "candidate18": 1,
        "candidate20": 20,
        "candidate21": 15,
    }
    expected_first_anchors = {
        "candidate16": 13.629,
        "candidate18": 12.656,
        "candidate20": None,
        "candidate21": 14.422,
    }
    expected_reanchors = {
        "candidate16": 2,
        "candidate18": 0,
        "candidate20": 0,
        "candidate21": 1,
    }
    blockers: list[str] = []
    for name, analysis in analyses.items():
        if analysis["official_active_gate_index"] != 3:
            blockers.append(f"{name}_did_not_reach_gate4")
        if analysis["coordinated_acquisition_count"] < minimum_counts[name]:
            blockers.append(f"{name}_insufficient_coordinated_acquisition")
        if analysis["coordinated_pitch_or_thrust_violation_count"]:
            blockers.append(f"{name}_acquisition_changes_pitch_or_thrust")
        if analysis["coordinated_roll_mismatch_count"]:
            blockers.append(f"{name}_acquisition_roll_mismatch")
        if analysis["coordinated_yaw_mismatch_count"]:
            blockers.append(f"{name}_acquisition_yaw_mismatch")
        if analysis["max_abs_coordinated_roll_norm"] > 0.200001:
            blockers.append(f"{name}_acquisition_roll_exceeds_point_one_rad")
        if analysis["dropout_roll_or_thrust_violation_count"]:
            blockers.append(f"{name}_dropout_not_level_hover")
        if analysis["dropout_yaw_non_hold_count"]:
            blockers.append(f"{name}_dropout_yaw_not_held")
        expected_anchor = expected_first_anchors[name]
        actual_anchor = analysis["first_anchor_elapsed_s"]
        if expected_anchor is None:
            if actual_anchor is not None:
                blockers.append(f"{name}_unexpected_initial_anchor")
        elif actual_anchor is None or abs(actual_anchor - expected_anchor) > 1e-6:
            blockers.append(f"{name}_initial_anchor_changed")
        if analysis["final_reanchor_count"] != expected_reanchors[name]:
            blockers.append(f"{name}_reanchor_count_changed")

    if analyses["candidate20"]["collision"]["id"] is not None:
        blockers.append("candidate20_not_collision_free")
    collision21 = analyses["candidate21"]["collision"]
    if collision21["id"] != 1002 or collision21["threat_level"] != 1:
        blockers.append("candidate21_failure_family_changed")
    if collision21["impact"] is None or collision21["impact"] >= 0.05:
        blockers.append("candidate21_not_low_speed_contact")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 3,
            "before_first_anchor_only": True,
            "delay_s": hybrid.GATE4_ACQUISITION_YAW_DELAY_S,
            "roll_kp": hybrid.GATE4_LATERAL_ROLL_KP,
            "max_roll_rad": hybrid.GATE4_MAX_LATERAL_ROLL_RAD,
            "max_yaw_step_rad": hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
            "pitch_norm": -hybrid.GATE4_MAX_PITCH_RAD / 0.5,
            "thrust_norm": hybrid._encode_gate4_thrust(
                hybrid.GATE4_HOVER_THRUST
            ),
            "dropout_level_hover_yaw_hold": True,
        },
        **analyses,
        "policy_callable": str(Path(hybrid.__file__).resolve()),
        "policy_callable_sha256": _sha256(Path(hybrid.__file__).resolve()),
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
