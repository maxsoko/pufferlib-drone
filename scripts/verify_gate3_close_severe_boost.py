#!/usr/bin/env python3
"""Verify the N150 close severe Gate-3 roll boost on archived/live traces."""

from __future__ import annotations

import argparse
from collections import Counter
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_observation(observation: list[float], gate: int) -> np.ndarray:
    if len(observation) == 32:
        return np.asarray(observation, dtype=np.float32)
    if len(observation) != 23:
        raise ValueError(f"expected 23 or 32 values, got {len(observation)}")
    values = np.zeros(32, dtype=np.float32)
    values[:23] = np.asarray(observation, dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = np.float32(1.0)
    return values


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    modes: Counter[str] = Counter()
    eligible = []
    non_roll_max_error = 0.0
    gate3_samples = 0
    for sample_index, sample in enumerate(samples):
        gate = int(sample.get("official_active_gate_index", -1))
        if gate != 2:
            continue
        values = convert_observation(sample.get("observation") or [], gate)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if recorded.shape != (4,):
            raise ValueError(f"invalid action shape in {path}:{sample_index}")
        mode, _ = tail._promoted_gate3_roll_command(values)
        modes[mode] += 1
        gate3_samples += 1
        projected = np.asarray(
            hybrid._gate3_close_severe_boost(values, recorded.tolist()),
            dtype=float,
        )
        non_roll_max_error = max(
            non_roll_max_error,
            float(np.max(np.abs(projected[[0, 2, 3]] - recorded[[0, 2, 3]]))),
        )
        if float(projected[1]) > float(recorded[1]) + 1e-9:
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
            forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
            right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
            closing_m_s = -forward_rate_m_s
            predicted_right_m = right_m
            if closing_m_s >= 1.0:
                predicted_right_m += (
                    right_rate_m_s * max(forward_m - 2.0, 0.0) / closing_m_s
                )
            eligible.append(
                {
                    "sample": sample_index,
                    "elapsed_s": sample.get("elapsed_s"),
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "predicted_right_m": predicted_right_m,
                    "recorded_roll": float(recorded[1]),
                    "boosted_roll": float(projected[1]),
                }
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "official_active_gate_index": int(report.get("official_active_gate_index") or 0),
        "collisions": int(
            (((report.get("sitl") or {}).get("telemetry") or {}).get("collisions") or 0)
        ),
        "gate3_samples": gate3_samples,
        "mode_counts": dict(sorted(modes.items())),
        "eligible_count": len(eligible),
        "eligible": eligible,
        "non_roll_max_error": non_roll_max_error,
    }


def verify(historical: list[Path], failed_candidate: Path, callable_path: Path) -> dict:
    blockers: list[str] = []
    history = [analyze(path) for path in historical]
    failed = analyze(failed_candidate)
    historical_fail_eligible = sum(
        item["eligible_count"]
        for item in history
        if item["official_active_gate_index"] < 3
    )
    historical_success_eligible = sum(
        item["eligible_count"]
        for item in history
        if item["official_active_gate_index"] >= 3
    )
    if len(history) != 10:
        blockers.append(f"historical_count:{len(history)}!=10")
    if sum(item["official_active_gate_index"] >= 3 for item in history) != 8:
        blockers.append("historical_gate3_reliability_not_8_of_10")
    if historical_fail_eligible <= 0:
        blockers.append("no_historical_lateral_failure_eligibility")
    if failed["official_active_gate_index"] != 2 or failed["collisions"] != 1:
        blockers.append("candidate011_not_gate3_collision")
    if failed["eligible_count"] <= 0:
        blockers.append("candidate011_close_severe_not_eligible")
    if failed["non_roll_max_error"] != 0.0 or any(
        item["non_roll_max_error"] != 0.0 for item in history
    ):
        blockers.append("proposal_changes_non_roll_action")
    if any(
        event["boosted_roll"] != 1.0
        for item in [*history, failed]
        for event in item["eligible"]
    ):
        blockers.append("eligible_roll_not_full_scale")
    source = callable_path.read_text(encoding="utf-8")
    required = [
        "GATE3_CLOSE_SEVERE_FORWARD_M = 4.0",
        'if mode != "adaptive":',
        "actions[1] = 1.0",
    ]
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        blockers.append(f"callable_contract_missing:{missing}")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "gate": 3,
            "max_forward_m": hybrid.GATE3_CLOSE_SEVERE_FORWARD_M,
            "required_promoted_mode": "adaptive",
            "changed_action": "roll only",
            "boosted_roll": 1.0,
        },
        "policy_callable": str(callable_path),
        "policy_callable_sha256": sha256_file(callable_path),
        "historical_failed_eligible_count": historical_fail_eligible,
        "historical_success_eligible_count": historical_success_eligible,
        "failed_candidate": failed,
        "historical": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", nargs="+", type=Path, required=True)
    parser.add_argument("--failed-candidate", type=Path, required=True)
    parser.add_argument("--policy-callable", type=Path, required=True)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify(args.historical, args.failed_candidate, args.policy_callable)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
