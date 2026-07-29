#!/usr/bin/env python3
"""Compare observable Gate-2-to-Gate-3 handoffs in official traces.

This diagnostic intentionally uses only fields available to the deployed
controller: the common first 23 values of the historical/current observation,
normalized action, official gate index, and elapsed time. Camera poses and
estimator landmarks are not used to decide whether a transition is recoverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inverse_tanh(value: float, scale: float) -> float:
    clipped = max(-0.999999, min(0.999999, float(value)))
    return math.atanh(clipped) * scale


def _decode(sample: dict) -> dict | None:
    observation = sample.get("observation") or []
    action = sample.get("normalized_action") or []
    if len(observation) not in (23, 32) or len(action) != 4:
        return None
    forward_rate_m_s = _inverse_tanh(observation[0], 5.0)
    return {
        "elapsed_s": float(sample.get("elapsed_s") or 0.0),
        "gate": int(sample.get("official_active_gate_index", -1)),
        "visible": float(observation[10]) >= 0.5,
        "forward_m": _inverse_tanh(observation[11], 10.0),
        "right_m": _inverse_tanh(observation[12], 5.0),
        "down_m": _inverse_tanh(observation[13], 5.0),
        "yaw_error_rad": max(-1.0, min(1.0, float(observation[14])))
        * (math.pi / 4.0),
        "forward_rate_m_s": forward_rate_m_s,
        "closing_m_s": -forward_rate_m_s,
        "right_rate_m_s": _inverse_tanh(observation[1], 3.0),
        "down_rate_m_s": _inverse_tanh(observation[2], 3.0),
        "pitch_norm": float(action[0]),
        "roll_norm": float(action[1]),
        "thrust_norm": float(action[2]),
        "yaw_norm": float(action[3]),
    }


def _public_row(row: dict | None) -> dict | None:
    return None if row is None else dict(row)


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        decoded
        for sample in sorted(
            (payload.get("policy_trace") or {}).get("samples") or [],
            key=lambda item: float(item.get("elapsed_s") or 0.0),
        )
        if (decoded := _decode(sample)) is not None
    ]
    gate2 = [row for row in rows if row["gate"] == 1]
    gate3 = [row for row in rows if row["gate"] == 2]
    transition = gate3[0] if gate3 else None
    visible = [
        row for row in gate3 if row["visible"] and row["forward_m"] > 0.0
    ]
    first_visible = visible[0] if visible else None
    last_gate2_visible = next(
        (row for row in reversed(gate2) if row["visible"]), None
    )
    admission = next(
        (
            row
            for row in visible
            if row["forward_m"] <= 12.0 and row["closing_m_s"] >= 1.0
        ),
        None,
    )
    severe = [
        row
        for row in visible
        if 15.0 <= row["forward_m"] <= 35.0
        and abs(row["right_m"]) >= 8.0
    ]
    receding = [row for row in visible if row["closing_m_s"] <= -1.0]
    official_index = int(payload.get("official_active_gate_index") or 0)
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "passed_gate3": official_index >= 3,
        "reported_official_active_gate_index": official_index,
        "collision_id": latest.get("collision_id"),
        "gate2_samples": len(gate2),
        "gate3_samples": len(gate3),
        "gate3_visible_samples": len(visible),
        "transition": _public_row(transition),
        "last_gate2_visible": _public_row(last_gate2_visible),
        "first_gate3_visible": _public_row(first_visible),
        "first_gate3_visible_delay_s": (
            None
            if transition is None or first_visible is None
            else first_visible["elapsed_s"] - transition["elapsed_s"]
        ),
        "first_close_range_admission": _public_row(admission),
        "severe_far_off_axis_samples": len(severe),
        "first_severe_far_off_axis": _public_row(severe[0] if severe else None),
        "receding_visible_samples": len(receding),
        "first_receding_visible": _public_row(receding[0] if receding else None),
        "minimum_visible_forward_m": (
            None if not visible else min(row["forward_m"] for row in visible)
        ),
        "minimum_visible_abs_right_m": (
            None if not visible else min(abs(row["right_m"]) for row in visible)
        ),
        "full_positive_roll_samples": sum(
            row["roll_norm"] >= 0.999 for row in visible
        ),
        "full_negative_roll_samples": sum(
            row["roll_norm"] <= -0.999 for row in visible
        ),
    }


def build_report(paths: list[Path]) -> dict:
    reports = [analyze(path) for path in paths]
    passing = [report for report in reports if report["passed_gate3"]]
    failing = [report for report in reports if not report["passed_gate3"]]

    def _count(group: list[dict], key: str) -> int:
        return sum(bool(report[key]) for report in group)

    return {
        "contract": {
            "inputs": (
                "common first 23 deployed observation values and normalized "
                "action only"
            ),
            "close_range_admission": "visible, forward <= 12 m, closing >= 1 m/s",
            "severe_far_off_axis": (
                "visible, 15 <= forward <= 35 m, abs(right) >= 8 m"
            ),
        },
        "source_count": len(reports),
        "passing_source_count": len(passing),
        "failing_source_count": len(failing),
        "summary": {
            "passing_with_close_range_admission": _count(
                passing, "first_close_range_admission"
            ),
            "failing_with_close_range_admission": _count(
                failing, "first_close_range_admission"
            ),
            "passing_with_severe_far_off_axis": _count(
                passing, "first_severe_far_off_axis"
            ),
            "failing_with_severe_far_off_axis": _count(
                failing, "first_severe_far_off_axis"
            ),
            "passing_with_receding_visible": _count(
                passing, "first_receding_visible"
            ),
            "failing_with_receding_visible": _count(
                failing, "first_receding_visible"
            ),
        },
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.reports)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
