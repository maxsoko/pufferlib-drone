#!/usr/bin/env python3
"""Verify the bounded Gate-2 governor on Candidates 013--015."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
    import policy_callable_six_gate_composite as tail
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts import policy_callable_six_gate_composite as tail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    eligible: list[dict] = []
    outside_gate2_changes = 0
    thrust_yaw_max_error = 0.0
    for sample_index, sample in enumerate(samples):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        gate = int(sample.get("official_active_gate_index", -1))
        projected = np.asarray(recorded, dtype=float)
        if gate == 1:
            projected = np.asarray(
                hybrid._gate2_projected_aperture_governor(
                    values, recorded.tolist()
                ),
                dtype=float,
            )
        elif not np.array_equal(projected, recorded):
            outside_gate2_changes += 1
        thrust_yaw_max_error = max(
            thrust_yaw_max_error,
            float(np.max(np.abs(projected[2:4] - recorded[2:4]))),
        )
        if gate == 1 and not np.array_equal(projected, recorded):
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
            forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
            right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
            time_to_plane_s = forward_m / max(
                -forward_rate_m_s,
                hybrid.GATE2_INTERCEPT_SPEED_FLOOR_M_S,
            )
            eligible.append(
                {
                    "sample": sample_index,
                    "elapsed_s": sample.get("elapsed_s"),
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "right_rate_m_s": right_rate_m_s,
                    "projected_right_at_plane_m": (
                        right_m + right_rate_m_s * time_to_plane_s
                    ),
                    "recorded_pitch_roll": recorded[:2].tolist(),
                    "governed_pitch_roll": projected[:2].tolist(),
                }
            )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "crash_detected": report.get("crash_detected"),
        "eligible_count": len(eligible),
        "eligible": eligible,
        "outside_gate2_changes": outside_gate2_changes,
        "thrust_yaw_max_error": thrust_yaw_max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate13", type=Path)
    parser.add_argument("candidate14", type=Path)
    parser.add_argument("candidate15", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(args.candidate13), analyze(args.candidate14), analyze(args.candidate15)]
    blockers: list[str] = []
    for number, report in zip((13, 14, 15), reports):
        if report["eligible_count"] <= 0:
            blockers.append(f"candidate{number}_has_no_governor_eligibility")
        if report["outside_gate2_changes"] != 0:
            blockers.append(f"candidate{number}_changes_outside_gate2")
        if report["thrust_yaw_max_error"] != 0.0:
            blockers.append(f"candidate{number}_changes_thrust_or_yaw")
    for number, report in zip((14, 15), reports[1:]):
        signs = {
            int(np.sign(item["projected_right_at_plane_m"]))
            for item in report["eligible"]
        }
        if not signs:
            blockers.append(f"candidate{number}_missing_projected_miss_sign")
    if not any(
        item["projected_right_at_plane_m"] < 0.0 for item in reports[1]["eligible"]
    ):
        blockers.append("candidate14_does_not_cover_left_edge")
    if not any(
        item["projected_right_at_plane_m"] > 0.0 for item in reports[2]["eligible"]
    ):
        blockers.append("candidate15_does_not_cover_right_edge")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "governor": {
            "gate_index": 1,
            "forward_limit_m": hybrid.GATE2_GOVERNOR_FORWARD_M,
            "projected_miss_threshold_m": hybrid.GATE2_PROJECTED_MISS_THRESHOLD_M,
            "brake_pitch_norm": hybrid.GATE2_BRAKE_PITCH_NORM,
            "roll_kp": hybrid.GATE2_ROLL_KP,
            "roll_kd": hybrid.GATE2_ROLL_KD,
            "roll_limit": hybrid.GATE2_ROLL_LIMIT,
        },
        "reports": reports,
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
