#!/usr/bin/env python3
"""Verify N160 yaw-only Gate-4 acquisition on rejected live pose traces."""

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


def _is_fail_safe(action: np.ndarray) -> bool:
    return bool(
        abs(float(action[0]) + hybrid.GATE4_MAX_PITCH_RAD / 0.5) <= 1e-8
        and abs(float(action[1])) <= 1e-8
        and abs(float(action[2])) <= 1e-8
    )


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
    for sample in samples:
        values = _raw_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        elapsed_s = float(sample["elapsed_s"])
        phase_elapsed_s = elapsed_s - first_elapsed_s
        current_yaw_rad = hybrid._yaw_from_quaternion(values)
        yaw_error_rad = (
            hybrid.tail._clamp(float(values[14])) * (math.pi / 4.0)
            if float(values[10]) > 0.5
            else None
        )
        governed = np.asarray(
            hybrid._gate4_safe_handoff(
                values,
                recorded.tolist(),
                now_s=100.0 + phase_elapsed_s,
            ),
            dtype=float,
        )
        fail_safe = _is_fail_safe(governed)
        yaw_step_rad = hybrid._wrap_angle(
            current_yaw_rad - float(governed[3]) * math.pi
        )
        acquisition = bool(
            fail_safe
            and float(values[10]) > 0.5
            and phase_elapsed_s >= hybrid.GATE4_ACQUISITION_YAW_DELAY_S
            and abs(yaw_step_rad) > 1e-8
        )
        rows.append(
            {
                "elapsed_s": elapsed_s,
                "phase_elapsed_s": phase_elapsed_s,
                "raw_gate_vector_m": (
                    None
                    if not sample.get("gate_pose")
                    else sample["gate_pose"]["body_vector_ned_m"]
                ),
                "yaw_error_rad": yaw_error_rad,
                "current_yaw_rad": current_yaw_rad,
                "yaw_step_rad": yaw_step_rad,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
                "fail_safe": fail_safe,
                "acquisition": acquisition,
                "reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
            }
        )

    acquisitions = [row for row in rows if row["acquisition"]]
    dropouts = [row for row in rows if row["raw_gate_vector_m"] is None]
    associated = [row for row in rows if not row["fail_safe"]]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": (
            ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                "collision_id"
            )
        ),
        "gate4_samples": len(rows),
        "visible_samples": sum(
            row["raw_gate_vector_m"] is not None for row in rows
        ),
        "dropout_samples": len(dropouts),
        "acquisition_count": len(acquisitions),
        "first_acquisition": acquisitions[0] if acquisitions else None,
        "last_acquisition": acquisitions[-1] if acquisitions else None,
        "max_abs_acquisition_yaw_step_rad": max(
            (abs(row["yaw_step_rad"]) for row in acquisitions),
            default=0.0,
        ),
        "acquisition_wrong_direction_count": sum(
            row["yaw_error_rad"] is None
            or row["yaw_step_rad"] * row["yaw_error_rad"] <= 0.0
            for row in acquisitions
        ),
        "acquisition_non_yaw_violation_count": sum(
            not _is_fail_safe(np.asarray(row["governed_action"], dtype=float))
            for row in acquisitions
        ),
        "dropout_non_hold_count": sum(
            abs(row["yaw_step_rad"]) > 1e-8 for row in dropouts
        ),
        "first_associated": associated[0] if associated else None,
        "final_reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("candidate18", type=Path)
    parser.add_argument("candidate20", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    candidate16 = _analyze(args.candidate16)
    candidate18 = _analyze(args.candidate18)
    candidate20 = _analyze(args.candidate20)
    blockers: list[str] = []
    for name, analysis in (
        ("candidate16", candidate16),
        ("candidate18", candidate18),
        ("candidate20", candidate20),
    ):
        if analysis["official_active_gate_index"] != 3:
            blockers.append(f"{name}_did_not_reach_gate4")
        if analysis["acquisition_wrong_direction_count"] != 0:
            blockers.append(f"{name}_yaw_direction_wrong")
        if analysis["acquisition_non_yaw_violation_count"] != 0:
            blockers.append(f"{name}_acquisition_changes_non_yaw")
        if analysis["dropout_non_hold_count"] != 0:
            blockers.append(f"{name}_dropout_yaw_not_held")
        if (
            analysis["max_abs_acquisition_yaw_step_rad"]
            > hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD + 1e-8
        ):
            blockers.append(f"{name}_yaw_step_exceeds_bound")
    if candidate16["final_reanchor_count"] != hybrid.GATE4_REANCHOR_MAX_COUNT:
        blockers.append("candidate16_reanchor_coverage_changed")
    first18 = candidate18["first_associated"]
    if first18 is None or first18["elapsed_s"] != 12.656:
        blockers.append("candidate18_centered_anchor_changed")
    if candidate20["collision_id"] is not None:
        blockers.append("candidate20_not_collision_free")
    if candidate20["acquisition_count"] < 20:
        blockers.append("candidate20_far_target_not_yaw_acquired")
    first20 = candidate20["first_acquisition"]
    if (
        first20 is None
        or first20["phase_elapsed_s"]
        < hybrid.GATE4_ACQUISITION_YAW_DELAY_S
    ):
        blockers.append("candidate20_acquisition_before_delay")
    if candidate20["first_associated"] is not None:
        blockers.append("candidate20_archived_trace_unexpectedly_associated")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 3,
            "delay_s": hybrid.GATE4_ACQUISITION_YAW_DELAY_S,
            "max_yaw_step_rad": hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
            "rejected_visible_only": True,
            "fail_safe_pitch_norm": -hybrid.GATE4_MAX_PITCH_RAD / 0.5,
            "fail_safe_roll_norm": 0.0,
            "fail_safe_thrust_norm": hybrid._encode_gate4_thrust(
                hybrid.GATE4_HOVER_THRUST
            ),
            "dropout_holds_current_yaw": True,
        },
        "candidate16_archived_gate4": candidate16,
        "candidate18_archived_gate4": candidate18,
        "candidate20_collision_free_stall": candidate20,
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
