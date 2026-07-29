#!/usr/bin/env python3
"""Verify Gate-1 positive-floor separation on passed and failed live traces."""

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


POSITIVE_MODES = frozenset({"adaptive", "early", "close"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert_observation(raw: list[float], gate: int) -> tuple[np.ndarray, str]:
    if len(raw) == 32:
        return np.asarray(raw, dtype=np.float32), "current32"
    if len(raw) != 23:
        raise ValueError(f"expected 23- or 32-value observation, got {len(raw)}")
    values = np.zeros(32, dtype=np.float32)
    values[:22] = np.asarray(raw[:22], dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = 1.0
    return values, "legacy23"


def _source_record(path: Path, report: dict) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "acceptance_passed": bool(report.get("acceptance_passed")),
        "official_active_gate_index": report.get("official_active_gate_index"),
    }


def _project_group(paths: Iterable[Path]) -> dict:
    mode_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    sources: list[dict] = []
    samples = 0
    positive_changes = 0
    positive_projection_max_error = 0.0
    positive_non_roll_max_error = 0.0
    full_adapter_changes = 0
    full_adapter_counter_changes = 0
    changed_samples: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append(_source_record(path, report))
        trace = (report.get("policy_trace") or {}).get("samples") or []
        for sample_index, sample in enumerate(trace):
            gate = int(sample.get("official_active_gate_index", -1))
            if gate != 0:
                continue
            expected = np.asarray(sample.get("normalized_action", []), dtype=float)
            if expected.shape != (4,):
                raise ValueError(
                    f"{path}: sample {sample_index}: expected four actions"
                )
            observation, trace_format = _convert_observation(
                sample.get("observation", []), gate
            )
            format_counts[trace_format] += 1
            mode, roll_command = composite._promoted_gate3_roll_command(observation)
            mode_counts[mode] += 1
            samples += 1

            positive = np.asarray(
                composite._promoted_gate1_positive_intercept(
                    observation, expected.tolist()
                ),
                dtype=float,
            )
            positive_error = float(np.max(np.abs(positive - expected)))
            positive_projection_max_error = max(
                positive_projection_max_error, positive_error
            )
            positive_non_roll_max_error = max(
                positive_non_roll_max_error,
                float(np.max(np.abs(positive[[0, 2, 3]] - expected[[0, 2, 3]]))),
            )
            if positive_error > 0.0:
                positive_changes += 1
                changed_samples.append(
                    {
                        "path": str(path),
                        "sample": sample_index,
                        "elapsed_s": sample.get("elapsed_s"),
                        "mode": mode,
                        "recorded_roll": float(expected[1]),
                        "projected_roll": float(positive[1]),
                        "roll_floor": roll_command,
                    }
                )

            full = np.asarray(
                composite._promoted_gate3_intercept(
                    observation, expected.tolist()
                ),
                dtype=float,
            )
            if float(np.max(np.abs(full - expected))) > 0.0:
                full_adapter_changes += 1
                if mode == "counter":
                    full_adapter_counter_changes += 1

    return {
        "reports": len(sources),
        "gate1_samples": samples,
        "mode_counts": dict(sorted(mode_counts.items())),
        "trace_format_counts": dict(sorted(format_counts.items())),
        "positive_projection_changes": positive_changes,
        "positive_projection_max_error": positive_projection_max_error,
        "positive_non_roll_max_error": positive_non_roll_max_error,
        "full_adapter_changes": full_adapter_changes,
        "full_adapter_counter_changes": full_adapter_counter_changes,
        "changed_samples": changed_samples,
        "sources": sources,
    }


def verify_separation(
    accepted_paths: Iterable[Path],
    failed_path: Path,
    *,
    expected_failed_changes: int = 6,
    required_failed_modes: tuple[str, ...] = ("adaptive", "close"),
) -> dict:
    accepted = _project_group(accepted_paths)
    failed = _project_group([failed_path])
    accepted_progress_violations = [
        source
        for source in accepted["sources"]
        if int(source.get("official_active_gate_index") or 0) < 1
    ]
    failed_source = failed["sources"][0] if failed["sources"] else {}
    failed_source_valid = (
        failed_source.get("acceptance_passed") is False
        and int(failed_source.get("official_active_gate_index") or 0) == 0
    )
    missing_failed_modes = [
        mode
        for mode in required_failed_modes
        if int(failed["mode_counts"].get(mode, 0)) <= 0
    ]
    changed_outside_positive_modes = [
        sample
        for sample in failed["changed_samples"]
        if sample["mode"] not in POSITIVE_MODES
    ]
    formats = set(accepted["trace_format_counts"]) | set(
        failed["trace_format_counts"]
    )
    missing_trace_formats = sorted({"legacy23", "current32"} - formats)
    blockers = []
    if accepted["reports"] <= 0 or accepted["gate1_samples"] <= 0:
        blockers.append("no_accepted_gate1_evidence")
    if accepted_progress_violations:
        blockers.append("accepted_source_without_official_gate1_pass")
    if accepted["positive_projection_changes"] != 0:
        blockers.append("accepted_trace_action_changed")
    if accepted["positive_non_roll_max_error"] != 0.0:
        blockers.append("accepted_non_roll_action_changed")
    if accepted["full_adapter_counter_changes"] <= 0:
        blockers.append("counter_exclusion_not_exercised")
    if not failed_source_valid:
        blockers.append("failed_source_contract_invalid")
    if failed["positive_projection_changes"] != expected_failed_changes:
        blockers.append("unexpected_failed_trace_change_count")
    if failed["positive_non_roll_max_error"] != 0.0:
        blockers.append("failed_non_roll_action_changed")
    if missing_failed_modes:
        blockers.append("failed_positive_modes_not_exercised")
    if changed_outside_positive_modes:
        blockers.append("failed_change_outside_positive_modes")
    if missing_trace_formats:
        blockers.append("trace_format_coverage_missing")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate": 0,
            "positive_modes": sorted(POSITIVE_MODES),
            "counter_mode_behavior": "leave_recurrent_roll_unchanged",
            "expected_failed_changes": expected_failed_changes,
            "required_failed_modes": list(required_failed_modes),
            "required_trace_formats": ["legacy23", "current32"],
        },
        "accepted": accepted,
        "failed": failed,
        "accepted_progress_violations": accepted_progress_violations,
        "failed_source_valid": failed_source_valid,
        "missing_failed_modes": missing_failed_modes,
        "changed_outside_positive_modes": changed_outside_positive_modes,
        "missing_trace_formats": missing_trace_formats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", nargs="+", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--expected-failed-changes", type=int, default=6)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify_separation(
        args.accepted,
        args.failed,
        expected_failed_changes=args.expected_failed_changes,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
