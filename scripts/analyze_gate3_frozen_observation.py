#!/usr/bin/env python3
"""Find frozen near-plane Gate-3 observation runs in official policy traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


FEATURE_INDICES = (0, 1, 2, 11, 12, 13, 14)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inverse_tanh(value: float, scale: float) -> float:
    return math.atanh(max(-0.999999, min(0.999999, float(value)))) * scale


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _public(sample_index: int, sample: dict, run_length: int) -> dict:
    obs = sample["observation"]
    return {
        "sample": sample_index,
        "elapsed_s": float(sample.get("elapsed_s") or 0.0),
        "run_length": run_length,
        "forward_m": _inverse_tanh(obs[11], 10.0),
        "closing_m_s": -_inverse_tanh(obs[0], 5.0),
        "right_m": _inverse_tanh(obs[12], 5.0),
        "right_rate_m_s": _inverse_tanh(obs[1], 3.0),
        "down_m": _inverse_tanh(obs[13], 5.0),
        "down_rate_m_s": _inverse_tanh(obs[2], 3.0),
        "yaw_error_rad": float(obs[14]) * (math.pi / 4.0),
        "normalized_action": sample.get("normalized_action"),
    }


def analyze(path: Path, trigger_samples: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )
    last_key = None
    run_length = 0
    longest = None
    first_trigger = None
    eligible_samples = 0
    for sample_index, sample in enumerate(samples):
        obs = sample.get("observation") or []
        if (
            int(sample.get("official_active_gate_index", -1)) != 2
            or len(obs) != 32
            or float(obs[10]) < 0.5
        ):
            last_key = None
            run_length = 0
            continue
        forward_m = _inverse_tanh(obs[11], 10.0)
        closing_m_s = -_inverse_tanh(obs[0], 5.0)
        if not (0.0 < forward_m <= 3.0 and closing_m_s >= 1.0):
            last_key = None
            run_length = 0
            continue
        eligible_samples += 1
        key = tuple(float(obs[index]) for index in FEATURE_INDICES)
        run_length = run_length + 1 if key == last_key else 1
        last_key = key
        row = _public(sample_index, sample, run_length)
        if longest is None or run_length > longest["run_length"]:
            longest = row
        if first_trigger is None and run_length >= trigger_samples:
            first_trigger = row
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "eligible_samples": eligible_samples,
        "longest_frozen_run": longest,
        "first_trigger": first_trigger,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--trigger-samples", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.trigger_samples < 2:
        raise ValueError("--trigger-samples must be at least two")
    reports = [analyze(path, args.trigger_samples) for path in args.traces]
    payload = {
        "contract": {
            "official_gate_index": 2,
            "visible": True,
            "max_forward_m": 3.0,
            "minimum_closing_m_s": 1.0,
            "feature_indices": FEATURE_INDICES,
            "consecutive_equal_samples": args.trigger_samples,
        },
        "triggered_candidates": [
            report["candidate"] for report in reports
            if report["first_trigger"] is not None
        ],
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
