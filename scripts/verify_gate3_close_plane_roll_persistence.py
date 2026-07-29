#!/usr/bin/env python3
"""Verify N159's Gate-3-only close-plane roll persistence on three live traces."""

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


def _roll_from_observation(values: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in values[6:10])
    return math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )


def _analyze(path: Path) -> dict:
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
        mode, _ = tail._promoted_gate3_roll_command(values)
        governed = np.asarray(
            hybrid._gate3_close_severe_boost(values, recorded.tolist()),
            dtype=float,
        )
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        closing_m_s = -forward_rate_m_s
        projected_right_m = right_m
        if closing_m_s >= 1.0:
            projected_right_m += (
                right_rate_m_s * max(forward_m - 2.0, 0.0) / closing_m_s
            )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "projected_right_m": projected_right_m,
                "mode": mode,
                "measured_roll_rad": _roll_from_observation(values),
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
                "roll_changed": float(governed[1]) != float(recorded[1]),
            }
        )
    changed = [row for row in rows if row["roll_changed"]]
    counter_suppressed = [
        row
        for row in changed
        if row["mode"] == "counter" and row["governed_action"][1] == 0.0
    ]
    severe_boosted = [
        row
        for row in changed
        if row["mode"] == "adaptive" and row["governed_action"][1] == 1.0
    ]
    non_roll_max_error = max(
        (
            max(
                abs(row["governed_action"][index] - row["recorded_action"][index])
                for index in (0, 2, 3)
            )
            for row in rows
        ),
        default=0.0,
    )
    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "gate3_samples": len(rows),
        "changed_samples": len(changed),
        "counter_suppressed_count": len(counter_suppressed),
        "severe_boosted_count": len(severe_boosted),
        "counter_suppressed": counter_suppressed,
        "severe_boosted": severe_boosted,
        "non_roll_max_error": non_roll_max_error,
        "last_gate3_row": rows[-1] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate16", type=Path)
    parser.add_argument("candidate18", type=Path)
    parser.add_argument("candidate19", type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    candidate16 = _analyze(args.candidate16)
    candidate18 = _analyze(args.candidate18)
    candidate19 = _analyze(args.candidate19)
    blockers: list[str] = []
    if int(candidate16["official_active_gate_index"] or -1) < 3:
        blockers.append("candidate16_not_clean_gate3_pass")
    if int(candidate18["official_active_gate_index"] or -1) < 3:
        blockers.append("candidate18_not_clean_gate3_pass")
    if candidate19["official_active_gate_index"] != 2:
        blockers.append("candidate19_not_gate3_failure")
    if candidate19["collision_id"] != 1001:
        blockers.append("candidate19_collision_family_changed")
    for name, analysis in (
        ("candidate16", candidate16),
        ("candidate18", candidate18),
        ("candidate19", candidate19),
    ):
        if analysis["non_roll_max_error"] != 0.0:
            blockers.append(f"{name}_changes_non_roll_action")
    if candidate16["counter_suppressed_count"] < 1:
        blockers.append("candidate16_does_not_exercise_counter_suppression")
    if candidate18["counter_suppressed_count"] != 0:
        blockers.append("candidate18_clean_severe_path_changed")
    if candidate19["counter_suppressed_count"] < 3:
        blockers.append("candidate19_close_counter_reversals_not_suppressed")
    final19 = candidate19["last_gate3_row"] or {}
    if abs(float(final19.get("projected_right_m", math.inf))) > 0.3:
        blockers.append("candidate19_final_projection_not_centered")
    if (final19.get("governed_action") or [0.0, -1.0])[1] < 0.0:
        blockers.append("candidate19_final_roll_still_negative")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "max_forward_m": hybrid.GATE3_CLOSE_SEVERE_FORWARD_M,
            "counter_roll_floor": hybrid.GATE3_CLOSE_COUNTER_ROLL_FLOOR,
            "left_side_only": True,
            "changed_action": "roll only",
            "adaptive_severe_roll": 1.0,
        },
        "candidate16_clean_gate3": candidate16,
        "candidate18_clean_gate3": candidate18,
        "candidate19_gate3_collision": candidate19,
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
