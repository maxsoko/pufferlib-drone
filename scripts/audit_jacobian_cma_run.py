#!/usr/bin/env python3
"""Build a machine-verifiable rejected-candidate index for a CMA run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coefficient_sha256(coefficients: list[float]) -> str:
    values = np.asarray(coefficients, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(report: dict[str, Any] | None) -> dict[str, float] | None:
    if report is None:
        return None
    values = report["metrics"]
    return {
        "success": float(values.get("env/success_rate", 0.0)),
        "gates": float(values.get("env/gates_passed", 0.0)),
        "crash": float(values.get("env/crash", 0.0)),
        "gate2_crossing": float(values.get("env/gate2_crossing_sampled", 0.0)),
        "gate2_terminal_radial": float(
            values.get("env/avg_gate2_terminal_crossing_radial", 0.0)
        ),
        "final_progress": float(values.get("env/final_progress", 0.0)),
    }


def build_rejected_candidate_index(
    run_dir: Path,
    *,
    expected_generations: int = 8,
    expected_population: int = 24,
    expected_dimension: int = 12,
) -> dict[str, Any]:
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    summaries = {
        int(summary["generation"]): summary
        for summary in final["generation_summaries"]
    }
    expected = set(range(expected_generations))
    if set(summaries) != expected:
        raise ValueError(
            f"expected generation summaries {sorted(expected)}, got {sorted(summaries)}"
        )
    if final.get("promoted") is not None:
        raise ValueError("run contains a promoted candidate; it is not a fully rejected run")

    records: list[dict[str, Any]] = []
    coefficient_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    evaluation_cache_keys: set[str] = set()
    development_evaluations = 0
    actual_anchor_evaluations = 0
    exact_evaluations = 0
    evaluation_cache_references = 0
    for generation in range(expected_generations):
        generation_dir = run_dir / f"generation_{generation:03d}"
        coefficients = np.load(generation_dir / "coefficients.npy")
        if coefficients.shape != (expected_population, expected_dimension):
            raise ValueError(f"generation {generation} has invalid population shape")
        summary = summaries[generation]
        winner_index = int(summary["winner_index"])
        for index in range(expected_population):
            candidate_path = generation_dir / f"candidate_{index:02d}.json"
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            stored_coefficients = candidate["checkpoint"]["coefficients"]
            stored = np.asarray(stored_coefficients, dtype=np.float32)
            if not np.array_equal(stored, coefficients[index].astype(np.float32)):
                raise ValueError(
                    f"generation {generation} candidate {index} coefficient mismatch"
                )
            coefficient_hash = _coefficient_sha256(stored_coefficients)
            checkpoint_hash = candidate["checkpoint"]["sha256"]
            checkpoint_path = Path(candidate["checkpoint"]["path"])
            if _file_sha256(checkpoint_path) != checkpoint_hash:
                raise ValueError(
                    f"generation {generation} candidate {index} checkpoint hash mismatch"
                )
            if coefficient_hash in coefficient_hashes:
                raise ValueError(f"duplicate coefficient vector {coefficient_hash}")
            if checkpoint_hash in checkpoint_hashes:
                raise ValueError(f"duplicate checkpoint {checkpoint_hash}")
            coefficient_hashes.add(coefficient_hash)
            checkpoint_hashes.add(checkpoint_hash)

            for report_name in ("development_target", "development_anchor"):
                report = candidate.get(report_name)
                if report is not None:
                    key = report.get("cache", {}).get("key")
                    if key:
                        evaluation_cache_references += 1
                        evaluation_cache_keys.add(key)
            if candidate.get("development_target") is not None:
                development_evaluations += 1
            if candidate.get("development_anchor") is not None:
                actual_anchor_evaluations += 1

            is_winner = index == winner_index
            heldout = summary["winner_validation"] if is_winner else None
            exact = summary["winner_exact"] if is_winner else None
            if heldout is not None:
                key = heldout.get("cache", {}).get("key")
                if key:
                    evaluation_cache_references += 1
                    evaluation_cache_keys.add(key)
            if exact is not None:
                exact_evaluations += 2
                for report in exact.values():
                    key = report.get("cache", {}).get("key")
                    if key:
                        evaluation_cache_references += 1
                        evaluation_cache_keys.add(key)

            if not candidate["trace_feasible"]:
                reason = "anchor_trace_trust_region_rejection"
            elif candidate.get("anchor_actual_feasible") is False:
                reason = "actual_floor4_anchor_rejection"
            elif is_winner:
                reason = "generation_winner_failed_floor4p5_promotion_contract"
            else:
                reason = "not_generation_winner"
            records.append(
                {
                    "generation": generation,
                    "index": index,
                    "coefficient_sha256": coefficient_hash,
                    "coefficients": stored_coefficients,
                    "checkpoint_sha256": checkpoint_hash,
                    "actual_delta_l2": candidate["checkpoint"]["actual_delta_l2"],
                    "trace_feasible": candidate["trace_feasible"],
                    "anchor_drift": candidate["anchor_drift"],
                    "failure_drift": candidate["failure_drift"],
                    "development_target": _metrics(candidate.get("development_target")),
                    "development_anchor": _metrics(candidate.get("development_anchor")),
                    "heldout_target": _metrics(heldout),
                    "exact_target": _metrics(exact["target"]) if exact else None,
                    "exact_anchor": _metrics(exact["anchor"]) if exact else None,
                    "rejection_reason": reason,
                }
            )

    expected_candidates = expected_generations * expected_population
    return {
        "run_dir": str(run_dir),
        "manifest_sha256": final["manifest_sha256"],
        "basis_sha256": final["basis_sha256"],
        "parent_sha256": final["parent_sha256"],
        "stop_reason": final["stop_reason"],
        "generations": expected_generations,
        "population_per_generation": expected_population,
        "candidate_count": len(records),
        "unique_coefficient_vectors": len(coefficient_hashes),
        "unique_candidate_checkpoints": len(checkpoint_hashes),
        "development_evaluations": development_evaluations,
        "actual_anchor_evaluations": actual_anchor_evaluations,
        "exact_evaluations": exact_evaluations,
        "unique_evaluation_cache_keys": len(evaluation_cache_keys),
        "evaluation_cache_references": evaluation_cache_references,
        "all_candidates_accounted_for": len(records) == expected_candidates,
        "no_duplicate_coefficients": len(coefficient_hashes) == expected_candidates,
        "no_duplicate_candidate_checkpoints": len(checkpoint_hashes)
        == expected_candidates,
        "no_duplicate_evaluation_references": len(evaluation_cache_keys)
        == evaluation_cache_references,
        "candidates": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-generations", type=int, default=8)
    parser.add_argument("--expected-population", type=int, default=24)
    parser.add_argument("--expected-dimension", type=int, default=12)
    args = parser.parse_args()
    report = build_rejected_candidate_index(
        args.run_dir,
        expected_generations=args.expected_generations,
        expected_population=args.expected_population,
        expected_dimension=args.expected_dimension,
    )
    _write_json(args.output, report)
    summary = {key: value for key, value in report.items() if key != "candidates"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
