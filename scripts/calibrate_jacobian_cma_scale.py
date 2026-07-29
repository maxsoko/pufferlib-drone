#!/usr/bin/env python3
"""Calibrate the pre-registered CMA coefficient scale before outcome evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from constrained_cma import FullCovarianceCMA
    from jacobian_cma_subspace import materialize_candidate, trace_action_drift
except ModuleNotFoundError:
    from scripts.constrained_cma import FullCovarianceCMA
    from scripts.jacobian_cma_subspace import materialize_candidate, trace_action_drift


def calibrate_scale(
    parent: Path,
    basis_path: Path,
    failure_trace: Path,
    anchor_trace: Path,
    output_dir: Path,
    *,
    dimension: int = 12,
    population_size: int = 24,
    parent_count: int = 12,
    seed: int = 3385,
    target_failure_rms: float = 0.005,
    maximum_anchor_rms: float = 0.001,
    maximum_anchor_max: float = 0.01,
    agents: int = 2,
) -> dict:
    data = np.load(basis_path, allow_pickle=False)
    arrays = {key: np.asarray(data[key]).copy() for key in data.files if key != "metadata"}
    metadata = json.loads(str(data["metadata"]))
    initial_scale = float(metadata["coefficient_scale"])
    if metadata.get("empirical_scale_calibration"):
        raise ValueError("basis has already been empirically calibrated")

    cma = FullCovarianceCMA(
        dimension,
        population_size=population_size,
        parent_count=parent_count,
        seed=seed,
    )
    coefficients = cma.ask()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, coefficient in enumerate(coefficients):
        candidate = output_dir / f"candidate_{index:02d}.bin"
        checkpoint = materialize_candidate(
            parent, basis_path, coefficient, candidate
        )
        failure_drift = trace_action_drift(
            parent, candidate, failure_trace, agents=agents
        )
        anchor_drift = trace_action_drift(
            parent, candidate, anchor_trace, agents=agents
        )
        records.append(
            {
                "index": index,
                "checkpoint": checkpoint,
                "failure_drift": failure_drift,
                "anchor_drift": anchor_drift,
            }
        )

    failure_rms = np.asarray([r["failure_drift"]["focus_rms"] for r in records])
    anchor_rms = np.asarray([
        max(r["anchor_drift"]["full_rms"], r["anchor_drift"]["focus_rms"])
        for r in records
    ])
    anchor_max = np.asarray([
        max(r["anchor_drift"]["full_max"], r["anchor_drift"]["focus_max"])
        for r in records
    ])
    failure_reference = float(np.median(failure_rms))
    anchor_rms_reference = float(np.quantile(anchor_rms, 0.95))
    anchor_max_reference = float(np.quantile(anchor_max, 0.95))
    ratios = {
        "failure_target": target_failure_rms / max(failure_reference, 1e-20),
        "anchor_rms_limit": maximum_anchor_rms / max(anchor_rms_reference, 1e-20),
        "anchor_max_limit": maximum_anchor_max / max(anchor_max_reference, 1e-20),
    }
    scale_factor = float(min(ratios.values()))
    calibrated_scale = initial_scale * scale_factor
    calibration = {
        "method": "generation_zero_parent_trace_only",
        "outcomes_observed": False,
        "dimension": dimension,
        "population_size": population_size,
        "parent_count": parent_count,
        "seed": seed,
        "agents": agents,
        "initial_coefficient_scale": initial_scale,
        "scale_factor": scale_factor,
        "calibrated_coefficient_scale": calibrated_scale,
        "target_failure_rms": target_failure_rms,
        "maximum_anchor_rms": maximum_anchor_rms,
        "maximum_anchor_max": maximum_anchor_max,
        "median_failure_focus_rms_before": failure_reference,
        "p95_anchor_rms_before": anchor_rms_reference,
        "p95_anchor_max_before": anchor_max_reference,
        "predicted_median_failure_focus_rms_after": failure_reference * scale_factor,
        "predicted_p95_anchor_rms_after": anchor_rms_reference * scale_factor,
        "predicted_p95_anchor_max_after": anchor_max_reference * scale_factor,
        "limiting_constraint": min(ratios, key=ratios.get),
        "ratios": ratios,
    }
    metadata["coefficient_scale"] = calibrated_scale
    metadata["empirical_scale_calibration"] = calibration
    np.savez_compressed(
        basis_path,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {"calibration": calibration, "records": records}
    (output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("basis", type=Path)
    parser.add_argument("failure_trace", type=Path)
    parser.add_argument("anchor_trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dimension", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--parent-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=3385)
    parser.add_argument("--target-failure-rms", type=float, default=0.005)
    parser.add_argument("--maximum-anchor-rms", type=float, default=0.001)
    parser.add_argument("--maximum-anchor-max", type=float, default=0.01)
    parser.add_argument("--agents", type=int, default=2)
    args = parser.parse_args()
    report = calibrate_scale(
        args.parent,
        args.basis,
        args.failure_trace,
        args.anchor_trace,
        args.output_dir,
        dimension=args.dimension,
        population_size=args.population_size,
        parent_count=args.parent_count,
        seed=args.seed,
        target_failure_rms=args.target_failure_rms,
        maximum_anchor_rms=args.maximum_anchor_rms,
        maximum_anchor_max=args.maximum_anchor_max,
        agents=args.agents,
    )
    print(json.dumps(report["calibration"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
