#!/usr/bin/env python3
"""Read-only capacity audit for legal VQ2 gate-mask geometry.

The native visual observation is a 64x64 confidence mask made from projected
physical gate edges. This audit derives deterministic image statistics and
coarse pooled pixels from that legal mask. Privileged active-gate position is
used only as an offline target after a fixed group-disjoint split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_boundary_plane_capacity import _load_boundary_event_dataset
from scripts.audit_vq2_event_latent_separability import (
    _phase_split_counts,
    _stratified_group_split,
)
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.audit_vq2_plane_representation_capacity import (
    _fit_continuous_ridge,
    _passes_capacity,
    _regression_result,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_legal_mask_geometry_plane_capacity_v1"
THRESHOLDS = (0.05, 0.25, 0.50, 0.75)


def _safe_weighted_moments(
    weight: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> list[np.ndarray]:
    mass = weight.sum((-2, -1))
    safe = np.maximum(mass, 1.0e-8)
    center_x = (weight * x).sum((-2, -1)) / safe
    center_y = (weight * y).sum((-2, -1)) / safe
    dx = x - center_x[..., None, None]
    dy = y - center_y[..., None, None]
    variance_x = (weight * np.square(dx)).sum((-2, -1)) / safe
    variance_y = (weight * np.square(dy)).sum((-2, -1)) / safe
    covariance = (weight * dx * dy).sum((-2, -1)) / safe
    return [mass / float(MASK_SIZE), center_x, center_y, variance_x, variance_y, covariance]


def _support_spans(support: np.ndarray) -> list[np.ndarray]:
    row = support.any(-1)
    column = support.any(-2)
    any_support = support.any((-2, -1))
    first_row = row.argmax(-1)
    last_row = VISUAL_HEIGHT - 1 - row[..., ::-1].argmax(-1)
    first_column = column.argmax(-1)
    last_column = VISUAL_WIDTH - 1 - column[..., ::-1].argmax(-1)
    height = np.where(
        any_support,
        (last_row - first_row + 1) / float(VISUAL_HEIGHT),
        0.0,
    )
    width = np.where(
        any_support,
        (last_column - first_column + 1) / float(VISUAL_WIDTH),
        0.0,
    )
    row_fraction = row.mean(-1)
    column_fraction = column.mean(-1)
    return [width, height, row_fraction, column_fraction]


def _mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return deterministic geometry for masks shaped [..., 64, 64]."""

    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape[-2:] != (VISUAL_HEIGHT, VISUAL_WIDTH):
        raise ValueError("mask geometry expects 64x64 images")
    if not np.isfinite(mask).all() or mask.min() < 0.0 or mask.max() > 1.0:
        raise ValueError("mask values must be finite and normalized")
    x = np.linspace(-1.0, 1.0, VISUAL_WIDTH, dtype=np.float64)[None, :]
    y = np.linspace(-1.0, 1.0, VISUAL_HEIGHT, dtype=np.float64)[:, None]
    values: list[np.ndarray] = []
    names: list[str] = []
    weighted = _safe_weighted_moments(mask, x, y)
    values.extend(weighted)
    names.extend(("mass", "center_x", "center_y", "variance_x", "variance_y", "covariance"))
    values.extend(
        (
            np.square(mask).mean((-2, -1)),
            mask.max((-2, -1)),
            mask.sum(-1).max(-1) / float(VISUAL_WIDTH),
            mask.sum(-2).max(-1) / float(VISUAL_HEIGHT),
            mask[..., 16:48, 16:48].mean((-2, -1)),
            np.concatenate(
                (
                    mask[..., :2, :].reshape(*mask.shape[:-2], -1),
                    mask[..., -2:, :].reshape(*mask.shape[:-2], -1),
                    mask[..., 2:-2, :2].reshape(*mask.shape[:-2], -1),
                    mask[..., 2:-2, -2:].reshape(*mask.shape[:-2], -1),
                ),
                -1,
            ).mean(-1),
        )
    )
    names.extend(("mean_square", "maximum", "max_row_mass", "max_column_mass", "center_mass", "border_mass"))
    for threshold in THRESHOLDS:
        support = mask >= threshold
        tag = str(threshold).replace(".", "p")
        values.append(support.mean((-2, -1)))
        names.append(f"support_{tag}")
        spans = _support_spans(support)
        values.extend(spans)
        names.extend(
            (
                f"width_{tag}",
                f"height_{tag}",
                f"row_fraction_{tag}",
                f"column_fraction_{tag}",
            )
        )
        binary_moments = _safe_weighted_moments(support.astype(np.float64), x, y)
        values.extend(binary_moments[1:])
        names.extend(
            (
                f"center_x_{tag}",
                f"center_y_{tag}",
                f"variance_x_{tag}",
                f"variance_y_{tag}",
                f"covariance_{tag}",
            )
        )
    base = np.stack(values, -1)
    # Projected edge length is approximately inverse range. These transforms
    # let a linear probe test that relationship without fitting a nonlinear net.
    positive = np.clip(base, 0.0, None)
    expanded = np.concatenate(
        (
            base,
            np.sign(base) * np.sqrt(np.abs(base)),
            np.square(base),
            np.log1p(64.0 * positive),
        ),
        -1,
    )
    expanded_names = (
        names
        + [f"signed_sqrt_{name}" for name in names]
        + [f"square_{name}" for name in names]
        + [f"log1p64_{name}" for name in names]
    )
    return expanded.astype(np.float32), expanded_names


def _pooled_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if VISUAL_HEIGHT % size or VISUAL_WIDTH % size:
        raise ValueError("pooled mask size must divide both visual dimensions")
    height_bin = VISUAL_HEIGHT // size
    width_bin = VISUAL_WIDTH // size
    return mask.reshape(
        *mask.shape[:-2], size, height_bin, size, width_bin
    ).mean((-3, -1))


def _temporal_features(
    series: np.ndarray,
    targets: tuple[int, ...],
    *,
    depth: int,
) -> np.ndarray:
    if depth <= 0 or min(targets) < depth:
        raise ValueError("temporal depth exceeds available target history")
    output: list[np.ndarray] = []
    for target in targets:
        indices = [target - 1 - lag for lag in range(depth)]
        values = series[:, indices]
        deltas = values[:, :-1] - values[:, 1:]
        output.append(
            np.concatenate((values.reshape(values.shape[0], -1), deltas.reshape(values.shape[0], -1)), -1)
        )
    return np.stack(output, 1)


def _current_features(series: np.ndarray, targets: tuple[int, ...]) -> np.ndarray:
    return np.stack([series[:, target - 1] for target in targets], 1)


def audit(args: argparse.Namespace) -> dict[str, object]:
    source_paths = {
        "event_dataset": args.event_dataset,
        "n644_report": args.n644_report,
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    dataset = _load_boundary_event_dataset(
        args.event_dataset,
        context_length=args.context_length,
        event_threshold=args.event_threshold,
        deterministic_size=args.deterministic_size,
        stochastic_groups=args.stochastic_groups,
        stochastic_classes=args.stochastic_classes,
    )
    negative_offsets = tuple(
        sorted((int(value) for value in args.negative_offsets.split(",")), reverse=True)
    )
    if not negative_offsets or any(value <= 0 for value in negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    if len(set(negative_offsets)) != len(negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    targets = tuple(args.context_length - value for value in negative_offsets) + (
        args.context_length,
    )
    mask = dataset["mask"].astype(np.float32).reshape(
        dataset["mask"].shape[0], args.context_length + 1, VISUAL_HEIGHT, VISUAL_WIDTH
    ) / 255.0
    geometry, geometry_names = _mask_geometry(mask)
    pooled = _pooled_mask(mask, args.pool_size).reshape(
        mask.shape[0], mask.shape[1], -1
    )
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    legal_tail = dataset["tail"][..., :legal_tail_size].astype(np.float32)
    current_tail = _current_features(legal_tail, targets)
    current_geometry = _current_features(geometry, targets)
    temporal_geometry = _temporal_features(
        geometry, targets, depth=args.temporal_depth
    )
    current_pooled = _current_features(pooled, targets)
    temporal_pooled = _temporal_features(pooled, targets, depth=args.temporal_depth)
    variants = {
        "mask_geometry_current": current_geometry,
        "mask_geometry_temporal": np.concatenate((temporal_geometry, current_tail), -1),
        "mask_geometry_pooled_current": np.concatenate((current_geometry, current_pooled, current_tail), -1),
        "mask_geometry_pooled_temporal": np.concatenate((temporal_geometry, temporal_pooled, current_tail), -1),
    }
    target_forward = np.stack(
        [
            dataset["tail"][:, target - 1, legal_tail_size + 3].astype(np.float32)
            for target in targets
        ],
        1,
    )
    labels = np.stack(
        [dataset["reward"][:, target] > args.event_threshold for target in targets], 1
    )
    if not labels[:, -1].all() or labels[:, :-1].any():
        raise RuntimeError("audited targets are not four hard negatives and one event")
    groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(np.int64)
    event_split = _stratified_group_split(
        groups,
        phases,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    ridge_values = tuple(float(value) for value in args.ridge_values.split(","))
    results: dict[str, object] = {}
    passes: dict[str, bool] = {}
    for name, feature in variants.items():
        prediction, fit = _fit_continuous_ridge(
            feature, target_forward, event_split, ridge_values=ridge_values
        )
        regression = _regression_result(target_forward, prediction, event_split)
        results[name] = {"fit": fit, "test": regression}
        passes[name] = _passes_capacity(regression)

    test = event_split == 2
    scalar_diagnostics = {
        name: _correlation(
            target_forward[test],
            _current_features(geometry[..., index : index + 1], targets)[test],
        )
        for index, name in enumerate(geometry_names)
    }
    strongest_scalar = max(
        scalar_diagnostics,
        key=lambda name: abs(float(scalar_diagnostics[name])),
    )
    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    split_summary = _phase_split_counts(event_split, phases, groups)
    best_name = max(
        results,
        key=lambda name: float(results[name]["test"]["correlation"]),
    )
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "all_splits_cover_all_phases": all(
            all(int(count) > 0 for count in split["phase_counts"].values())
            for split in split_summary.values()
        ),
        "legal_input_only": True,
        "finite_features": all(np.isfinite(feature).all() for feature in variants.values()),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "targets": list(targets),
        "negative_offsets": list(negative_offsets),
        "temporal_depth": args.temporal_depth,
        "pool_size": args.pool_size,
        "geometry_base_feature_names": geometry_names,
        "variant_input_sizes": {name: int(value.shape[-1]) for name, value in variants.items()},
        "event_split": split_summary,
        "variant_results": results,
        "variant_passes_capacity_gate": passes,
        "best_variant": best_name,
        "best_test_correlation": float(results[best_name]["test"]["correlation"]),
        "any_variant_passes_capacity_gate": any(passes.values()),
        "strongest_direct_scalar": strongest_scalar,
        "strongest_direct_scalar_test_correlation": float(scalar_diagnostics[strongest_scalar]),
        "direct_scalar_test_correlations": scalar_diagnostics,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "probe_fits": len(variants) * len(ridge_values),
        "optimizer_steps": 0,
        "model_loaded": 0,
        "model_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--n644-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=644)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-values", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--deterministic-size", type=int, default=256)
    parser.add_argument("--stochastic-groups", type=int, default=16)
    parser.add_argument("--stochastic-classes", type=int, default=16)
    args = parser.parse_args()
    if args.context_length <= 0 or args.temporal_depth <= 0 or args.pool_size <= 0:
        parser.error("context, temporal depth, and pool size must be positive")
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
