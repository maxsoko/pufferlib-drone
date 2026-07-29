#!/usr/bin/env python3
"""Audit small-corpus minibatch variance without loading a model or sealed test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_event_partition_support import _crop_exposure_counts
from scripts.audit_vq2_event_pair_support import _crop_pair_exposure_counts
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index
from scripts.continue_vq2_success_prefix_representation_offline import _target_batch


AUDIT_SCHEMA = "vq2_donor_sampling_variance_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_sampling_counts(
    item_count: int,
    batch_size: int,
    steps: int,
    *,
    seed: int,
    dense_pool_size: int,
    dense_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if min(item_count, batch_size, steps, dense_pool_size, dense_batch_size) <= 0:
        raise ValueError("sampling geometry must be positive")
    if batch_size > item_count or dense_batch_size > dense_pool_size:
        raise ValueError("audit expects without-replacement minibatches")
    rng = np.random.default_rng(seed)
    item_counts = np.zeros(item_count, dtype=np.int64)
    dense_counts = np.zeros(dense_pool_size, dtype=np.int64)
    for _ in range(steps):
        chosen = rng.choice(item_count, batch_size, replace=False)
        dense = rng.choice(dense_pool_size, dense_batch_size, replace=False)
        item_counts[chosen] += 1
        dense_counts[dense] += 1
    return item_counts, dense_counts


def _balanced_sampling_counts(
    item_count: int, batch_size: int, steps: int, *, seed: int
) -> np.ndarray:
    if min(item_count, batch_size, steps) <= 0 or batch_size > item_count:
        raise ValueError("balanced sampling geometry is invalid")
    if item_count % batch_size:
        raise ValueError("balanced audit requires whole minibatches per epoch")
    rng = np.random.default_rng(seed)
    counts = np.zeros(item_count, dtype=np.int64)
    batches_per_epoch = item_count // batch_size
    completed = 0
    while completed < steps:
        order = rng.permutation(item_count)
        for batch in range(batches_per_epoch):
            if completed == steps:
                break
            chosen = order[batch * batch_size : (batch + 1) * batch_size]
            counts[chosen] += 1
            completed += 1
    return counts


def _complete_progress_steps(report: dict, thresholds: dict[str, float]) -> list[int]:
    result: list[int] = []
    for candidate in report.get("validation_history", []):
        metrics = candidate["validation_progress"]
        if (
            metrics["correlation"] >= thresholds["minimum_correlation"]
            and metrics["mae_m"] <= thresholds["maximum_mae_m"]
            and metrics["near_plane_mae_m"]
            <= thresholds["maximum_near_plane_mae_m"]
            and metrics["direction_accuracy"]
            >= thresholds["minimum_direction_accuracy"]
            and all(candidate["preservation_gates"].values())
        ):
            result.append(int(candidate["step"]))
    return result


def _support_geometry(
    arrays: dict[str, np.ndarray],
    crop_starts: tuple[int, ...],
    *,
    sequence_length: int,
    burn_in: int,
    near_plane_m: float,
    near_plane_multiplier: float,
    minimum_pair_delta_m: float,
) -> dict[str, np.ndarray | float | int]:
    events = len(arrays["vector_step"])
    original_event, crop_start = _crop_index(events, crop_starts)
    exposure = _crop_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=sequence_length,
        burn_in=burn_in,
    )
    pair_exposure = _crop_pair_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=sequence_length,
        burn_in=burn_in,
    )
    target = _target_batch(
        arrays, original_event, crop_start, sequence_length=sequence_length
    )[:, burn_in:]
    time_index = crop_start[:, None] + np.arange(
        burn_in, sequence_length, dtype=np.int64
    )[None, :]
    near = np.abs(target) <= near_plane_m
    plane_weight = (1.0 / exposure[time_index]) * np.where(
        near, near_plane_multiplier, 1.0
    )
    pair_index = crop_start[:, None] + np.arange(
        burn_in + 1, sequence_length, dtype=np.int64
    )[None, :]
    pair_near = near[:, :-1] | near[:, 1:]
    pair_weight = (1.0 / pair_exposure[pair_index]) * np.where(
        pair_near, near_plane_multiplier, 1.0
    )
    qualified_pair = np.abs(np.diff(target, axis=1)) >= minimum_pair_delta_m
    return {
        "events": events,
        "original_event": original_event,
        "crop_start": crop_start,
        "time_index": time_index,
        "near": near,
        "plane_weight": plane_weight,
        "pair_weight": pair_weight,
        "qualified_pair": qualified_pair,
        "crop_plane_weight": plane_weight.sum(axis=1),
        "crop_near_weight": (plane_weight * near).sum(axis=1),
        "crop_qualified_pair_weight": (pair_weight * qualified_pair).sum(axis=1),
    }


def _schedule_metrics(
    counts: np.ndarray,
    dense_counts: np.ndarray,
    support: dict[str, np.ndarray | float | int],
) -> dict[str, float | int]:
    original_event = np.asarray(support["original_event"])
    time_index = np.asarray(support["time_index"])
    plane_weight = np.asarray(support["plane_weight"])
    crop_plane_weight = np.asarray(support["crop_plane_weight"])
    crop_near_weight = np.asarray(support["crop_near_weight"])
    crop_pair_weight = np.asarray(support["crop_qualified_pair_weight"])
    events = int(support["events"])
    desired = np.zeros((events, int(time_index.max()) + 1), dtype=np.float64)
    actual = np.zeros_like(desired)
    for crop, multiplicity in enumerate(counts):
        desired[original_event[crop], time_index[crop]] += plane_weight[crop]
        actual[original_event[crop], time_index[crop]] += (
            multiplicity * plane_weight[crop]
        )
    active = desired > 0.0
    exposure_ratio = actual[active] / desired[active]
    event_draws = np.bincount(original_event, weights=counts, minlength=events)
    return {
        "draws": int(counts.sum()),
        "unseen_crops": int((counts == 0).sum()),
        "crop_count_minimum": int(counts.min()),
        "crop_count_maximum": int(counts.max()),
        "crop_count_std": float(counts.std()),
        "event_draw_minimum": int(event_draws.min()),
        "event_draw_maximum": int(event_draws.max()),
        "event_draw_std": float(event_draws.std()),
        "unique_timestep_zero_exposure": int((actual[active] == 0.0).sum()),
        "unique_timestep_exposure_ratio_minimum": float(exposure_ratio.min()),
        "unique_timestep_exposure_ratio_maximum": float(exposure_ratio.max()),
        "unique_timestep_exposure_ratio_std": float(exposure_ratio.std()),
        "near_plane_weight_share": float(
            np.dot(counts, crop_near_weight) / np.dot(counts, crop_plane_weight)
        ),
        "qualifying_pair_exposure_factor": float(
            np.dot(counts, crop_pair_weight) / crop_pair_weight.sum()
        ),
        "dense_draws": int(dense_counts.sum()),
        "dense_count_minimum": int(dense_counts.min()),
        "dense_count_maximum": int(dense_counts.max()),
        "dense_count_std": float(dense_counts.std()),
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    sources_before = {
        "train_dataset": _sha256(args.train_dataset),
        "n690_report": _sha256(args.n690_report),
        "n692_report": _sha256(args.n692_report),
        "executable": _sha256(Path(__file__)),
    }
    with np.load(args.train_dataset) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    n690 = json.loads(args.n690_report.read_text(encoding="utf-8"))
    n692 = json.loads(args.n692_report.read_text(encoding="utf-8"))
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    support = _support_geometry(
        arrays,
        crop_starts,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        near_plane_m=args.near_plane_m,
        near_plane_multiplier=args.near_plane_multiplier,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    crop_count = len(np.asarray(support["original_event"]))
    schedules: dict[str, dict[str, float | int]] = {}
    reference_rows: list[tuple[float, float]] = []
    for seed in range(args.reference_seed_count):
        counts, dense = _current_sampling_counts(
            crop_count,
            args.batch_size,
            args.steps,
            seed=seed,
            dense_pool_size=args.dense_pool_size,
            dense_batch_size=args.dense_batch_size,
        )
        metrics = _schedule_metrics(counts, dense, support)
        reference_rows.append(
            (
                float(metrics["near_plane_weight_share"]),
                float(metrics["qualifying_pair_exposure_factor"]),
            )
        )
    reference = np.asarray(reference_rows, dtype=np.float64)
    for name, seed in (("n690", args.n690_seed), ("n692", args.n692_seed)):
        counts, dense = _current_sampling_counts(
            crop_count,
            args.batch_size,
            args.steps,
            seed=seed,
            dense_pool_size=args.dense_pool_size,
            dense_batch_size=args.dense_batch_size,
        )
        metrics = _schedule_metrics(counts, dense, support)
        metrics["seed"] = seed
        metrics["near_plane_share_lower_percentile"] = float(
            (reference[:, 0] < metrics["near_plane_weight_share"]).mean()
        )
        metrics["qualifying_pair_lower_percentile"] = float(
            (reference[:, 1] < metrics["qualifying_pair_exposure_factor"]).mean()
        )
        schedules[name] = metrics
    balanced_steps = args.balanced_epochs * (crop_count // args.batch_size)
    balanced_counts = _balanced_sampling_counts(
        crop_count, args.batch_size, balanced_steps, seed=args.balanced_seed
    )
    balanced_dense = _balanced_sampling_counts(
        args.dense_pool_size,
        args.dense_batch_size,
        balanced_steps,
        seed=args.balanced_seed + 1,
    )
    balanced = _schedule_metrics(balanced_counts, balanced_dense, support)
    balanced["seed"] = args.balanced_seed
    balanced["steps"] = balanced_steps
    thresholds = {
        "minimum_correlation": args.minimum_validation_correlation,
        "maximum_mae_m": args.maximum_validation_mae_m,
        "maximum_near_plane_mae_m": args.maximum_validation_near_plane_mae_m,
        "minimum_direction_accuracy": args.minimum_validation_direction_accuracy,
    }
    n690_pass_steps = _complete_progress_steps(n690, thresholds)
    n692_pass_steps = _complete_progress_steps(n692, thresholds)
    ideal_near_share = float(
        np.asarray(support["crop_near_weight"]).sum()
        / np.asarray(support["crop_plane_weight"]).sum()
    )
    source_contracts = bool(
        n690.get("contract") == "vq2_success_prefix_joint_representation_v1"
        and n692.get("contract") == "vq2_success_prefix_joint_representation_v1"
        and n690.get("train_events") == len(arrays["vector_step"])
        and n692.get("train_events") == len(arrays["vector_step"])
        and n690.get("inverse_crop_pair_exposure_weighting") is True
        and n692.get("inverse_crop_pair_exposure_weighting") is True
        and n690.get("sealed_test_dataset_opened") == 0
        and n692.get("sealed_test_dataset_opened") == 0
    )
    sources_after = {
        "train_dataset": _sha256(args.train_dataset),
        "n690_report": _sha256(args.n690_report),
        "n692_report": _sha256(args.n692_report),
        "executable": _sha256(Path(__file__)),
    }
    gates = {
        "sources_unchanged": sources_before == sources_after,
        "source_contracts_exact": source_contracts,
        "n690_has_complete_progress_pass": n690_pass_steps == [350],
        "n692_has_no_complete_progress_pass": n692_pass_steps == [],
        "n692_near_share_bottom_fifteen_percent": schedules["n692"][
            "near_plane_share_lower_percentile"
        ]
        <= 0.15,
        "n692_pair_exposure_bottom_fifteen_percent": schedules["n692"][
            "qualifying_pair_lower_percentile"
        ]
        <= 0.15,
        "schedule_measures_correlate": float(
            np.corrcoef(reference[:, 0], reference[:, 1])[0, 1]
        )
        >= 0.98,
        "balanced_three_epochs_exact": bool(
            balanced_steps == 357
            and balanced["crop_count_minimum"] == args.balanced_epochs
            and balanced["crop_count_maximum"] == args.balanced_epochs
            and balanced["unseen_crops"] == 0
            and balanced["event_draw_std"] == 0.0
            and balanced["unique_timestep_zero_exposure"] == 0
            and abs(balanced["near_plane_weight_share"] - ideal_near_share)
            <= 1e-12
            and abs(
                balanced["qualifying_pair_exposure_factor"]
                - args.balanced_epochs
            )
            <= 1e-12
        ),
        "balanced_dense_spread_at_most_one": (
            balanced["dense_count_maximum"] - balanced["dense_count_minimum"]
            <= 1
        ),
    }
    return {
        "contract": AUDIT_SCHEMA,
        "source_hashes": sources_before,
        "sources_unchanged": sources_before == sources_after,
        "crop_starts": list(crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "train_events": len(arrays["vector_step"]),
        "train_crops": crop_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "dense_pool_size": args.dense_pool_size,
        "dense_batch_size": args.dense_batch_size,
        "reference_seed_count": args.reference_seed_count,
        "reference_near_pair_correlation": float(
            np.corrcoef(reference[:, 0], reference[:, 1])[0, 1]
        ),
        "ideal_near_plane_weight_share": ideal_near_share,
        "reference_distribution": {
            "near_plane_weight_share_mean": float(reference[:, 0].mean()),
            "near_plane_weight_share_std": float(reference[:, 0].std()),
            "qualifying_pair_exposure_factor_mean": float(reference[:, 1].mean()),
            "qualifying_pair_exposure_factor_std": float(reference[:, 1].std()),
        },
        "source_validation_pass_steps": {
            "n690": n690_pass_steps,
            "n692": n692_pass_steps,
        },
        "current_schedules": schedules,
        "balanced_schedule": balanced,
        "process_gates": gates,
        "process_passed": all(gates.values()),
        "model_loaded": 0,
        "optimizer_steps": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "flightsim_packets": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--n690-report", type=Path, required=True)
    parser.add_argument("--n692-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-pool-size", type=int, default=128)
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--n690-seed", type=int, default=690)
    parser.add_argument("--n692-seed", type=int, default=692)
    parser.add_argument("--reference-seed-count", type=int, default=200)
    parser.add_argument("--balanced-epochs", type=int, default=3)
    parser.add_argument("--balanced-seed", type=int, default=693)
    parser.add_argument("--minimum-validation-correlation", type=float, default=0.8)
    parser.add_argument("--maximum-validation-mae-m", type=float, default=0.5)
    parser.add_argument(
        "--maximum-validation-near-plane-mae-m", type=float, default=0.6
    )
    parser.add_argument(
        "--minimum-validation-direction-accuracy", type=float, default=0.75
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("audit refuses to overwrite its report")
    if args.reference_seed_count < 2 or args.balanced_epochs <= 0:
        parser.error("audit support counts are invalid")
    report = run_audit(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["process_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
