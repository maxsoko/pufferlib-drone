#!/usr/bin/env python3
"""Summarize observable Gate-3 control modes from recorded policy traces."""

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
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        action = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or action.shape != (4,):
            continue
        visible = float(values[10]) >= 0.5
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        time_to_terminal_s = (
            max(forward_m - 2.0, 0.0) / closing_m_s
            if closing_m_s >= 1.0
            else 0.0
        )
        projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
        projected_down_m = down_m + down_rate_m_s * time_to_terminal_s
        mode, promoted_roll = tail._promoted_gate3_roll_command(values)
        margin_condition = bool(
            visible
            and hybrid.GATE3_TERMINAL_MARGIN_MIN_FORWARD_M
            < forward_m
            <= hybrid.GATE3_TERMINAL_MARGIN_FORWARD_M
            and closing_m_s >= 1.0
            and projected_right_m
            < hybrid.GATE3_TERMINAL_MARGIN_PROJECTED_RIGHT_TRIGGER_M
        )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "visible": visible,
                "forward_m": forward_m,
                "right_m": right_m,
                "down_m": down_m,
                "closing_m_s": closing_m_s,
                "right_rate_m_s": right_rate_m_s,
                "down_rate_m_s": down_rate_m_s,
                "projected_right_at_terminal_m": projected_right_m,
                "projected_down_at_terminal_m": projected_down_m,
                "promoted_mode": mode,
                "promoted_roll": promoted_roll,
                "margin_condition": margin_condition,
                "normalized_action": action.tolist(),
            }
        )

    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": report.get(
            "official_active_gate_index"
        ),
        "collision_id": latest.get("collision_id"),
        "collision_threat_level": latest.get("collision_threat_level"),
        "collision_impact": latest.get("collision_impact"),
        "gate3_samples": len(rows),
        "visible_samples": sum(row["visible"] for row in rows),
        "margin_condition_samples": sum(row["margin_condition"] for row in rows),
        "full_negative_roll_samples": sum(
            row["normalized_action"][1] <= -0.999 for row in rows
        ),
        "margin_floor_roll_samples": sum(
            abs(
                row["normalized_action"][1]
                - hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM
            )
            <= 1e-6
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "contract": {
            "margin_max_forward_m": hybrid.GATE3_TERMINAL_MARGIN_FORWARD_M,
            "margin_min_forward_m_exclusive": (
                hybrid.GATE3_TERMINAL_MARGIN_MIN_FORWARD_M
            ),
            "margin_projected_right_trigger_m": (
                hybrid.GATE3_TERMINAL_MARGIN_PROJECTED_RIGHT_TRIGGER_M
            ),
            "margin_roll_floor_norm": (
                hybrid.GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM
            ),
        },
        "reports": [analyze(path) for path in args.traces],
        "policy_callable_sha256": _sha256(Path(hybrid.__file__).resolve()),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
