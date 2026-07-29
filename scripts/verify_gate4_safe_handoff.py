#!/usr/bin/env python3
"""Replay Candidate 016 raw Gate-4 evidence through the N156 safety handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_observation(sample: dict) -> np.ndarray | None:
    values = np.asarray(sample.get("observation") or [], dtype=np.float32)
    if values.shape != (32,):
        return None
    values = values.copy()
    values[0:3] = np.float32(0.0)
    pose = sample.get("gate_pose")
    if not pose:
        values[10:17] = np.float32(0.0)
        return values
    forward_m, right_m, down_m = (
        float(value) for value in pose["body_vector_ned_m"]
    )
    values[10] = np.float32(1.0)
    values[11] = np.tanh(np.float32(forward_m / 10.0))
    values[12] = np.tanh(np.float32(right_m / 5.0))
    values[13] = np.tanh(np.float32(down_m / 5.0))
    values[14] = np.float32(float(pose["yaw_error_rad"]) / (np.pi / 4.0))
    return values


def analyze(candidate_path: Path) -> dict:
    report = json.loads(candidate_path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    gate4_samples = [
        sample
        for sample in samples
        if int(sample.get("official_active_gate_index", -1)) == 3
    ]
    first_elapsed_s = float(gate4_samples[0]["elapsed_s"])
    hybrid._clear_gate4_state()
    rows: list[dict] = []
    for sample in gate4_samples:
        values = _raw_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        elapsed_s = float(sample["elapsed_s"])
        action = np.asarray(
            hybrid._gate4_safe_handoff(
                values,
                recorded.tolist(),
                now_s=100.0 + elapsed_s - first_elapsed_s,
            ),
            dtype=float,
        )
        pose = sample.get("gate_pose")
        rows.append(
            {
                "elapsed_s": elapsed_s,
                "raw_gate_vector_m": None if not pose else pose["body_vector_ned_m"],
                "recorded_action": recorded.tolist(),
                "governed_action": action.tolist(),
                "reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
            }
        )

    visible_rows = [row for row in rows if row["raw_gate_vector_m"] is not None]
    dropout_rows = [row for row in rows if row["raw_gate_vector_m"] is None]
    largest_recorded_roll = max(visible_rows, key=lambda row: abs(row["recorded_action"][1]))
    closest_row = min(
        visible_rows,
        key=lambda row: float(row["raw_gate_vector_m"][0]),
    )
    return {
        "path": str(candidate_path),
        "sha256": _sha256(candidate_path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "ordered_gate_passes": report.get("ordered_gate_passes"),
        "crash_detected": report.get("crash_detected"),
        "collision": {
            "id": ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                "collision_id"
            ),
            "impact": ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                "collision_impact"
            ),
        },
        "gate4_samples": len(rows),
        "visible_samples": len(visible_rows),
        "dropout_samples": len(dropout_rows),
        "final_reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
        "max_abs_governed_roll": max(abs(row["governed_action"][1]) for row in rows),
        "min_governed_thrust": min(row["governed_action"][2] for row in rows),
        "max_governed_pitch": max(row["governed_action"][0] for row in rows),
        "dropout_max_abs_roll": max(
            abs(row["governed_action"][1]) for row in dropout_rows
        ),
        "dropout_max_abs_thrust": max(
            abs(row["governed_action"][2]) for row in dropout_rows
        ),
        "largest_recorded_roll_sample": largest_recorded_roll,
        "closest_raw_sample": closest_row,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("--closest-frame", type=Path)
    parser.add_argument("--latest-frame", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze(args.candidate16)
    blockers: list[str] = []
    if analysis["official_active_gate_index"] != 3:
        blockers.append("candidate16_did_not_reach_gate4")
    if analysis["ordered_gate_passes"] != 3:
        blockers.append("candidate16_did_not_pass_exactly_three_gates")
    if not analysis["crash_detected"] or analysis["collision"]["id"] != 1002:
        blockers.append("candidate16_failure_family_changed")
    if analysis["gate4_samples"] < 50 or analysis["visible_samples"] < 50:
        blockers.append("insufficient_gate4_replay_samples")
    if analysis["final_reanchor_count"] != hybrid.GATE4_REANCHOR_MAX_COUNT:
        blockers.append("measured_close_families_not_reanchored")
    if analysis["max_abs_governed_roll"] > 0.200001:
        blockers.append("gate4_roll_exceeds_point_one_rad")
    if analysis["min_governed_thrust"] < hybrid._encode_gate4_thrust(0.22) - 1e-9:
        blockers.append("gate4_thrust_below_point_two_two")
    if analysis["max_governed_pitch"] > hybrid.GATE4_ALIGNED_FORWARD_PITCH_RAD / 0.5 + 1e-9:
        blockers.append("gate4_forward_pitch_exceeds_bound")
    if analysis["dropout_max_abs_roll"] != 0.0:
        blockers.append("dropout_roll_not_level")
    if analysis["dropout_max_abs_thrust"] != 0.0:
        blockers.append("dropout_thrust_not_hover")

    frames = {}
    for name, path in (("closest", args.closest_frame), ("latest", args.latest_frame)):
        if path is not None:
            if not path.is_file():
                blockers.append(f"missing_{name}_frame")
            else:
                frames[name] = {"path": str(path), "sha256": _sha256(path)}

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "handoff": {
            "gate_index": 3,
            "raw_observation_start_index": 3,
            "raw_observation_end_index": 3,
            "association_grace_s": hybrid.GATE4_ASSOCIATION_GRACE_S,
            "max_pose_jump_m": hybrid.GATE4_MAX_POSE_JUMP_M,
            "max_anchor_drift_m": hybrid.GATE4_MAX_ANCHOR_DRIFT_M,
            "reanchor_persistence_s": hybrid.GATE4_REANCHOR_PERSISTENCE_S,
            "reanchor_max_forward_m": hybrid.GATE4_REANCHOR_MAX_FORWARD_M,
            "reanchor_max_abs_right_m": hybrid.GATE4_REANCHOR_MAX_ABS_RIGHT_M,
            "reanchor_max_count": hybrid.GATE4_REANCHOR_MAX_COUNT,
            "max_roll_rad": hybrid.GATE4_MAX_LATERAL_ROLL_RAD,
            "descent_delay_s": hybrid.GATE4_DESCENT_DELAY_S,
            "descent_thrust": hybrid.GATE4_DESCENT_THRUST,
            "dropout_thrust": hybrid.GATE4_HOVER_THRUST,
        },
        "analysis": analysis,
        "frames": frames,
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
