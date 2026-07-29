#!/usr/bin/env python3
"""Verify N162 half-second Gate-4 dropout yaw-target carry."""

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
    first_anchor_elapsed_s: float | None = None
    rows: list[dict] = []
    for sample in samples:
        values = _raw_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        elapsed_s = float(sample["elapsed_s"])
        phase_elapsed_s = elapsed_s - first_elapsed_s
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
        if first_anchor_elapsed_s is None and hybrid._GATE4_ANCHOR_VECTOR is not None:
            first_anchor_elapsed_s = elapsed_s
        visible = float(values[10]) > 0.5
        target_age_s = (
            None
            if hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_S is None
            else 100.0
            + phase_elapsed_s
            - hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_S
        )
        carry_expected = bool(
            not visible
            and hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD is not None
            and target_age_s is not None
            and target_age_s <= hybrid.GATE4_DROPOUT_YAW_CARRY_S
        )
        expected_yaw_rad = (
            hybrid._wrap_angle(
                current_yaw_rad
                + hybrid.tail._clamp(
                    hybrid._wrap_angle(
                        float(hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD)
                        - current_yaw_rad
                    ),
                    -hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
                    hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
                )
            )
            if carry_expected
            else current_yaw_rad
        )
        actual_yaw_rad = float(governed[3]) * math.pi
        pose = sample.get("gate_pose")
        rows.append(
            {
                "elapsed_s": elapsed_s,
                "phase_elapsed_s": phase_elapsed_s,
                "raw_gate_vector_m": (
                    None if not pose else pose["body_vector_ned_m"]
                ),
                "anchor_absent_after": anchor_absent_after,
                "current_yaw_rad": current_yaw_rad,
                "remembered_yaw_target_rad": (
                    hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD
                ),
                "remembered_target_age_s": target_age_s,
                "carry_expected": carry_expected,
                "expected_yaw_rad": expected_yaw_rad,
                "actual_yaw_rad": actual_yaw_rad,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
                "reanchor_count": hybrid._GATE4_REANCHOR_COUNT,
            }
        )

    dropout_rows = [row for row in rows if row["raw_gate_vector_m"] is None]
    carry_rows = [row for row in dropout_rows if row["carry_expected"]]
    expired_rows = [row for row in dropout_rows if not row["carry_expected"]]
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
        "carry_samples": len(carry_rows),
        "expired_hold_samples": len(expired_rows),
        "dropout_non_yaw_violation_count": sum(
            abs(row["governed_action"][0] + hybrid.GATE4_MAX_PITCH_RAD / 0.5)
            > 1e-8
            or abs(row["governed_action"][1]) > 1e-8
            or abs(row["governed_action"][2]) > 1e-8
            for row in dropout_rows
        ),
        "dropout_yaw_mismatch_count": sum(
            abs(
                hybrid._wrap_angle(
                    row["actual_yaw_rad"] - row["expected_yaw_rad"]
                )
            )
            > 1e-7
            for row in dropout_rows
        ),
        "max_carry_age_s": max(
            (float(row["remembered_target_age_s"]) for row in carry_rows),
            default=0.0,
        ),
        "max_abs_carried_yaw_delta_rad": max(
            (
                abs(
                    hybrid._wrap_angle(
                        row["actual_yaw_rad"] - row["current_yaw_rad"]
                    )
                )
                for row in carry_rows
            ),
            default=0.0,
        ),
        "pre_anchor_acquisition_roll_violation_count": sum(
            abs(row["governed_action"][1]) > 1e-8
            for row in rows
            if row["raw_gate_vector_m"] is not None
            and row["remembered_yaw_target_rad"] is not None
            and row["anchor_absent_after"]
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
    parser.add_argument("candidate22", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    analyses = {
        "candidate16": _analyze(args.candidate16),
        "candidate18": _analyze(args.candidate18),
        "candidate20": _analyze(args.candidate20),
        "candidate21": _analyze(args.candidate21),
        "candidate22": _analyze(args.candidate22),
    }
    expected_first_anchors = {
        "candidate16": 13.629,
        "candidate18": 12.656,
        "candidate20": None,
        "candidate21": 14.422,
        "candidate22": 14.406,
    }
    expected_reanchors = {
        "candidate16": 2,
        "candidate18": 0,
        "candidate20": 0,
        "candidate21": 1,
        "candidate22": 0,
    }
    blockers: list[str] = []
    for name, analysis in analyses.items():
        if analysis["official_active_gate_index"] != 3:
            blockers.append(f"{name}_did_not_reach_gate4")
        if analysis["dropout_non_yaw_violation_count"]:
            blockers.append(f"{name}_dropout_changes_non_yaw")
        if analysis["dropout_yaw_mismatch_count"]:
            blockers.append(f"{name}_dropout_yaw_mismatch")
        if analysis["max_carry_age_s"] > hybrid.GATE4_DROPOUT_YAW_CARRY_S + 1e-8:
            blockers.append(f"{name}_carry_exceeds_time_bound")
        if (
            analysis["max_abs_carried_yaw_delta_rad"]
            > hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD + 1e-8
        ):
            blockers.append(f"{name}_carry_exceeds_yaw_bound")
        if analysis["pre_anchor_acquisition_roll_violation_count"]:
            blockers.append(f"{name}_unsafe_n161_roll_not_removed")
        expected_anchor = expected_first_anchors[name]
        actual_anchor = analysis["first_anchor_elapsed_s"]
        if expected_anchor is None:
            if actual_anchor is not None:
                blockers.append(f"{name}_unexpected_anchor")
        elif actual_anchor is None or abs(actual_anchor - expected_anchor) > 1e-6:
            blockers.append(f"{name}_anchor_changed")
        if analysis["final_reanchor_count"] != expected_reanchors[name]:
            blockers.append(f"{name}_reanchor_count_changed")

    for name in ("candidate16", "candidate18", "candidate21"):
        if analyses[name]["carry_samples"] < 3:
            blockers.append(f"{name}_carry_not_exercised")
        if analyses[name]["expired_hold_samples"] < 3:
            blockers.append(f"{name}_expiry_hold_not_exercised")
    if analyses["candidate20"]["expired_hold_samples"] < 200:
        blockers.append("candidate20_long_dropout_hold_not_preserved")
    if analyses["candidate22"]["dropout_samples"] != 0:
        blockers.append("candidate22_unexpected_dropout_family")
    collision21 = analyses["candidate21"]["collision"]
    if collision21["id"] != 1002 or collision21["threat_level"] != 1:
        blockers.append("candidate21_low_speed_failure_changed")
    collision22 = analyses["candidate22"]["collision"]
    if collision22["id"] != 1002 or collision22["threat_level"] != 2:
        blockers.append("candidate22_high_speed_failure_changed")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 3,
            "carry_duration_s": hybrid.GATE4_DROPOUT_YAW_CARRY_S,
            "remembered_target_only": True,
            "dropout_pitch_norm": -hybrid.GATE4_MAX_PITCH_RAD / 0.5,
            "dropout_roll_norm": 0.0,
            "dropout_thrust_norm": hybrid._encode_gate4_thrust(
                hybrid.GATE4_HOVER_THRUST
            ),
            "expiry_holds_current_yaw": True,
            "n161_pre_anchor_roll_removed": True,
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
