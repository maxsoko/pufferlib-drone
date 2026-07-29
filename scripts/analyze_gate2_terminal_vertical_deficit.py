#!/usr/bin/env python3
"""Compare observable terminal Gate-2 vertical deficits in official traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inverse_tanh(value: float, scale: float) -> float:
    return math.atanh(max(-0.999999, min(0.999999, float(value)))) * scale


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _decode(sample: dict) -> dict | None:
    observation = sample.get("observation") or []
    action = sample.get("normalized_action") or []
    if len(observation) not in (23, 32) or len(action) != 4:
        return None
    if int(sample.get("official_active_gate_index", -1)) != 1:
        return None
    if float(observation[10]) < 0.5:
        return None
    forward_m = _inverse_tanh(observation[11], 10.0)
    closing_m_s = -_inverse_tanh(observation[0], 5.0)
    if not (0.0 < forward_m <= 12.0 and closing_m_s >= 1.0):
        return None
    down_m = _inverse_tanh(observation[13], 5.0)
    down_rate_m_s = _inverse_tanh(observation[2], 3.0)
    horizon_s = forward_m / max(closing_m_s, 3.0)
    return {
        "elapsed_s": float(sample.get("elapsed_s") or 0.0),
        "forward_m": forward_m,
        "closing_m_s": closing_m_s,
        "right_m": _inverse_tanh(observation[12], 5.0),
        "right_rate_m_s": _inverse_tanh(observation[1], 3.0),
        "down_m": down_m,
        "down_rate_m_s": down_rate_m_s,
        "projected_down_m": down_m + down_rate_m_s * horizon_s,
        "pitch_norm": float(action[0]),
        "roll_norm": float(action[1]),
        "thrust_norm": float(action[2]),
        "yaw_norm": float(action[3]),
    }


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for sample in sorted(
            (payload.get("policy_trace") or {}).get("samples") or [],
            key=lambda item: float(item.get("elapsed_s") or 0.0),
        ) if (row := _decode(sample)) is not None
    ]
    rapid = [
        row for row in rows
        if row["down_rate_m_s"] < -2.8 and row["projected_down_m"] < -0.75
    ]
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    official_index = int(payload.get("official_active_gate_index") or 0)
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "passed_gate2": official_index >= 2,
        "official_active_gate_index": official_index,
        "collision_id": latest.get("collision_id"),
        "terminal_samples": len(rows),
        "minimum_projected_down_m": min(
            (row["projected_down_m"] for row in rows), default=None
        ),
        "minimum_down_rate_m_s": min(
            (row["down_rate_m_s"] for row in rows), default=None
        ),
        "rapid_deficit_samples": len(rapid),
        "rapid_deficit_changed_if_full_thrust": sum(
            row["thrust_norm"] < 1.0 for row in rapid
        ),
        "first_rapid_deficit": rapid[0] if rapid else None,
        "last_rapid_deficit": rapid[-1] if rapid else None,
        "last_terminal_sample": rows[-1] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    payload = {
        "contract": {
            "official_gate_index": 1,
            "max_forward_m": 12.0,
            "closing_speed_floor_m_s": 3.0,
            "rapid_down_rate_trigger_m_s": -2.8,
            "projected_down_trigger_m": -0.75,
            "candidate_action": "floor normalized thrust at +1.0",
        },
        "passing_count": sum(report["passed_gate2"] for report in reports),
        "failing_count": sum(not report["passed_gate2"] for report in reports),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
