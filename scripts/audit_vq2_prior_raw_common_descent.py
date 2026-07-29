#!/usr/bin/env python3
"""Solve the N728 horizon Gram matrix for a raw common descent direction."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_prior_horizon_gradient_conflict import (
    AUDIT_SCHEMA as N728_SCHEMA,
    _weight_candidates,
)
from scripts.audit_vq2_probe_gradient_alignment import _sha256


AUDIT_SCHEMA = "vq2_prior_raw_minimum_norm_common_descent_v1"
N728_REPORT_SHA256 = "c9af8630ea20ec87fa21d0216957700eb27ef26cbb7378639d58013f01fba376"
HORIZONS = (1, 2, 4, 8, 16)


def _gram_from_report(report: dict[str, object]) -> np.ndarray:
    norms = np.asarray(
        [float(report["gradient_norms"][str(horizon)]) for horizon in HORIZONS],
        dtype=np.float64,
    )
    cosine = np.asarray(report["gradient_cosine_matrix"], dtype=np.float64)
    return cosine * norms[:, None] * norms[None, :]


def _minimum_norm_weights(
    gram: np.ndarray, *, maximum_iterations: int = 100_000, tolerance: float = 1e-13
) -> tuple[np.ndarray, int, float]:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError("minimum-norm Gram shape mismatch")
    weight = np.full(gram.shape[0], 1.0 / gram.shape[0], dtype=np.float64)
    gap = math.inf
    for iteration in range(1, maximum_iterations + 1):
        product = gram @ weight
        vertex = int(np.argmin(product))
        gap = float(weight @ product - product[vertex])
        scale = max(1.0, abs(float(weight @ product)))
        if gap <= tolerance * scale:
            return weight, iteration, gap
        direction = -weight
        direction[vertex] += 1.0
        denominator = float(direction @ gram @ direction)
        if denominator <= 0.0:
            raise RuntimeError("minimum-norm line search is not positive")
        step = float(np.clip(-direction @ product / denominator, 0.0, 1.0))
        weight += step * direction
    return weight, maximum_iterations, gap


def _raw_direction_row(gram: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    weight = np.asarray(weights, dtype=np.float64)
    weight = weight / weight.sum()
    product = gram @ weight
    squared_norm = float(weight @ product)
    norm = math.sqrt(max(squared_norm, 0.0))
    derivatives = -product / max(norm, 1e-30)
    return {
        "weights": weight.tolist(),
        "combined_gradient_norm": norm,
        "unit_raw_directional_derivative": {
            str(horizon): float(derivatives[index])
            for index, horizon in enumerate(HORIZONS)
        },
        "common_raw_descent": bool(np.all(derivatives < 0.0)),
        "worst_unit_raw_derivative": float(np.max(derivatives)),
        "mean_unit_raw_derivative": float(np.mean(derivatives)),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    source_before = {
        "executable": _sha256(Path(__file__)),
        "n728_report": _sha256(args.n728_report),
    }
    if source_before["n728_report"] != N728_REPORT_SHA256:
        raise RuntimeError("raw common-descent source hash mismatch")
    n728 = json.loads(args.n728_report.read_text(encoding="utf-8"))
    if (
        n728.get("contract") != N728_SCHEMA
        or n728.get("process_passed") is not True
        or n728.get("selection_informative") is not False
        or n728.get("common_adam_descent_candidates") != 0
        or tuple(n728.get("gradient_cosine_matrix_horizon_order", ())) != HORIZONS
    ):
        raise RuntimeError("raw common-descent predecessor contract mismatch")
    gram = _gram_from_report(n728)
    weights, iterations, final_gap = _minimum_norm_weights(gram)
    minimum_norm = _raw_direction_row(gram, weights)
    sampled_weights, canonical_count = _weight_candidates()
    sampled_rows = [_raw_direction_row(gram, row) for row in sampled_weights]
    sampled_common = [row for row in sampled_rows if row["common_raw_descent"]]
    source_after = {
        "executable": _sha256(Path(__file__)),
        "n728_report": _sha256(args.n728_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "n728_source_exact": source_after["n728_report"] == N728_REPORT_SHA256,
        "gram_symmetric": bool(np.allclose(gram, gram.T, rtol=0.0, atol=1e-6)),
        "gram_positive_semidefinite": float(np.linalg.eigvalsh(gram).min()) >= -1e-5,
        "minimum_norm_weights_valid": (
            bool(np.all(weights >= 0.0)) and abs(float(weights.sum()) - 1.0) <= 1e-12
        ),
        "fixed_search_reproduced": (
            sampled_weights.shape == (8220, 5) and canonical_count == 28
        ),
        "finite_diagnostics": all(
            math.isfinite(value)
            for value in [
                final_gap,
                *weights,
                *minimum_norm["unit_raw_directional_derivative"].values(),
            ]
        ),
        "no_dataset_or_model_path": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
        "minimum_norm_iterations": iterations,
        "minimum_norm_final_frank_wolfe_gap": final_gap,
        "minimum_norm_candidate": minimum_norm,
        "sampled_common_raw_descent_candidates": len(sampled_common),
        "selection_informative": bool(minimum_norm["common_raw_descent"]),
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "train_dataset_path_received": 0,
        "validation_dataset_path_received": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "model_path_received": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if not report["process_passed"]:
        raise RuntimeError("raw common-descent process gates failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n728-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("raw common-descent audit refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
