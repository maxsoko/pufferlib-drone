#!/usr/bin/env python3
"""Verify N157's earlier latched Gate-2 governor on live pass/failure traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    hybrid._clear_gate2_state()
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 1:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,) or values[10] < 0.5:
            continue
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        if not 0.0 < forward_m <= hybrid.GATE2_GOVERNOR_FORWARD_M:
            continue
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        time_to_plane_s = forward_m / max(
            -forward_rate_m_s,
            hybrid.GATE2_INTERCEPT_SPEED_FLOOR_M_S,
        )
        projected_right_m = right_m + right_rate_m_s * time_to_plane_s
        latched_before = hybrid._GATE2_GOVERNOR_LATCHED
        governed = np.asarray(
            hybrid._gate2_projected_aperture_governor(
                values,
                recorded.tolist(),
            ),
            dtype=float,
        )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "projected_right_at_plane_m": projected_right_m,
                "old_point_five_eligible": abs(projected_right_m) > 0.5,
                "new_point_three_eligible": (
                    abs(projected_right_m)
                    > hybrid.GATE2_PROJECTED_MISS_THRESHOLD_M
                ),
                "latched_before": latched_before,
                "latched_after": hybrid._GATE2_GOVERNOR_LATCHED,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
                "changed": not np.array_equal(governed, recorded),
            }
        )

    old_first = next((row for row in rows if row["old_point_five_eligible"]), None)
    new_first = next((row for row in rows if row["new_point_three_eligible"]), None)
    retained_below_threshold = [
        row
        for row in rows
        if row["latched_before"]
        and not row["new_point_three_eligible"]
        and row["changed"]
    ]
    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "rows": rows,
        "old_first_activation": old_first,
        "new_first_activation": new_first,
        "activation_lead_s": (
            None
            if old_first is None or new_first is None
            else old_first["elapsed_s"] - new_first["elapsed_s"]
        ),
        "retained_below_threshold_count": len(retained_below_threshold),
        "retained_below_threshold": retained_below_threshold,
        "max_abs_governed_roll": max(
            abs(row["governed_action"][1]) for row in rows
        ),
        "thrust_yaw_max_error": max(
            max(
                abs(row["governed_action"][index] - row["recorded_action"][index])
                for index in (2, 3)
            )
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("candidate17", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    passing = analyze(args.candidate16)
    failing = analyze(args.candidate17)
    blockers: list[str] = []
    if passing["official_active_gate_index"] < 2:
        blockers.append("candidate16_does_not_prove_gate2_pass")
    if failing["official_active_gate_index"] != 1 or failing["collision_id"] != 1001:
        blockers.append("candidate17_failure_family_changed")
    for name, report in (("candidate16", passing), ("candidate17", failing)):
        first = report["new_first_activation"]
        if first is None:
            blockers.append(f"{name}_has_no_point_three_activation")
        elif first["old_point_five_eligible"]:
            blockers.append(f"{name}_does_not_activate_before_old_threshold")
        if report["activation_lead_s"] is None or report["activation_lead_s"] <= 0.0:
            blockers.append(f"{name}_has_no_measured_activation_lead")
        if report["max_abs_governed_roll"] > hybrid.GATE2_ROLL_LIMIT + 1e-9:
            blockers.append(f"{name}_roll_exceeds_bound")
        if report["thrust_yaw_max_error"] != 0.0:
            blockers.append(f"{name}_changes_thrust_or_yaw")
    if passing["retained_below_threshold_count"] <= 0:
        blockers.append("candidate16_does_not_exercise_latch_retention")
    expected_candidate17_projection = 0.331
    actual_candidate17_projection = failing["new_first_activation"][
        "projected_right_at_plane_m"
    ]
    if not math.isclose(
        actual_candidate17_projection,
        expected_candidate17_projection,
        abs_tol=0.01,
    ):
        blockers.append("candidate17_first_projection_mismatch")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "governor": {
            "gate_index": 1,
            "forward_limit_m": hybrid.GATE2_GOVERNOR_FORWARD_M,
            "projected_miss_threshold_m": (
                hybrid.GATE2_PROJECTED_MISS_THRESHOLD_M
            ),
            "latched_until_official_transition": True,
            "brake_pitch_norm": hybrid.GATE2_BRAKE_PITCH_NORM,
            "roll_kp": hybrid.GATE2_ROLL_KP,
            "roll_kd": hybrid.GATE2_ROLL_KD,
            "roll_limit": hybrid.GATE2_ROLL_LIMIT,
        },
        "candidate16_clean_gate2": passing,
        "candidate17_gate2_collision": failing,
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
