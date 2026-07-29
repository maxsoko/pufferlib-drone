#!/usr/bin/env python3
"""Verify the promoted Gate-3 adapter against archived Candidate-129 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import policy_callable_six_gate_composite as composite
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_composite as composite


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert_observation(old: list[float], gate: int) -> np.ndarray:
    if len(old) != 23:
        raise ValueError(f"expected legacy 23-value observation, got {len(old)}")
    values = np.zeros(32, dtype=np.float32)
    values[:22] = np.asarray(old[:22], dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = 1.0
    return values


def verify_traces(
    paths: Iterable[Path],
    *,
    action_atol: float = 2e-4,
    required_modes: tuple[str, ...] = ("adaptive", "early", "counter"),
) -> dict:
    mode_counts: Counter[str] = Counter()
    violations: list[dict] = []
    projection_max_error = 0.0
    gate3_samples = 0
    accepted_reports = 0
    source_records = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        accepted_reports += int(bool(report.get("acceptance_passed")))
        source_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "acceptance_passed": bool(report.get("acceptance_passed")),
                "official_active_gate_index": report.get(
                    "official_active_gate_index"
                ),
            }
        )
        samples = (report.get("policy_trace") or {}).get("samples") or []
        for sample_index, sample in enumerate(samples):
            gate = int(sample.get("official_active_gate_index", -1))
            if gate != 2:
                continue
            expected = np.asarray(sample.get("normalized_action", []), dtype=float)
            if expected.shape != (4,):
                violations.append(
                    {
                        "path": str(path),
                        "sample": sample_index,
                        "reason": "invalid_action_shape",
                    }
                )
                continue
            observation = _convert_observation(sample.get("observation", []), gate)
            mode, roll_command = composite._promoted_gate3_roll_command(observation)
            mode_counts[mode] += 1
            gate3_samples += 1
            projected = np.asarray(
                composite._promoted_gate3_intercept(
                    observation, expected.astype(float).tolist()
                ),
                dtype=float,
            )
            projection_max_error = max(
                projection_max_error,
                float(np.max(np.abs(projected - expected))),
            )
            if mode in {"adaptive", "early", "close"}:
                if roll_command is None or expected[1] + action_atol < roll_command:
                    violations.append(
                        {
                            "path": str(path),
                            "sample": sample_index,
                            "reason": "recorded_roll_below_frozen_floor",
                            "mode": mode,
                            "command": roll_command,
                            "recorded_roll": float(expected[1]),
                        }
                    )
            elif mode == "counter" and abs(float(expected[1]) + 1.0) > action_atol:
                violations.append(
                    {
                        "path": str(path),
                        "sample": sample_index,
                        "reason": "recorded_counter_roll_mismatch",
                        "recorded_roll": float(expected[1]),
                    }
                )

    missing_modes = [mode for mode in required_modes if mode_counts[mode] <= 0]
    paths_count = len(source_records)
    return {
        "passed": (
            paths_count > 0
            and gate3_samples > 0
            and not violations
            and not missing_modes
            and projection_max_error <= action_atol
        ),
        "reports": paths_count,
        "accepted_reports": accepted_reports,
        "gate3_samples": gate3_samples,
        "mode_counts": dict(sorted(mode_counts.items())),
        "required_modes": list(required_modes),
        "missing_modes": missing_modes,
        "action_atol": action_atol,
        "projection_max_error": projection_max_error,
        "violations": violations,
        "sources": source_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--action-atol", type=float, default=2e-4)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify_traces(args.traces, action_atol=args.action_atol)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
