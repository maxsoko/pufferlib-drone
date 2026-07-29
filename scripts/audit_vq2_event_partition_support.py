#!/usr/bin/env python3
"""Audit plane-distance support in physically isolated VQ2 train/validation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_success_prefix_sequence_capacity import _forward_metres
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_event_partition_plane_support_audit_v1"
TARGET_TAIL_INDEX = LEGAL_OBS_SIZE - MASK_SIZE + 3
DEFAULT_BIN_EDGES_M = (-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)


def _plane_distance_m(tail: np.ndarray) -> np.ndarray:
    tail = np.asarray(tail)
    if tail.ndim != 3 or tail.shape[-1] <= TARGET_TAIL_INDEX:
        raise ValueError("tail does not contain the privileged plane target")
    target = _forward_metres(
        np.asarray(tail[..., TARGET_TAIL_INDEX], dtype=np.float32)
    ).astype(np.float32)
    if not np.isfinite(target).all():
        raise RuntimeError("plane target contains non-finite values")
    return target


def _crop_exposure_counts(
    history_length: int,
    crop_starts: tuple[int, ...],
    *,
    sequence_length: int,
    burn_in: int,
) -> np.ndarray:
    if history_length <= 0 or sequence_length <= 0 or burn_in < 0:
        raise ValueError("history, sequence length, and burn-in are invalid")
    if burn_in >= sequence_length:
        raise ValueError("burn-in must be shorter than the sequence")
    exposure = np.zeros(history_length, dtype=np.int64)
    for start in crop_starts:
        if start < 0 or start + sequence_length > history_length:
            raise ValueError("crop falls outside the source history")
        exposure[start + burn_in : start + sequence_length] += 1
    return exposure


def _histogram(values: np.ndarray, edges: tuple[float, ...]) -> list[dict[str, object]]:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=np.float64))
    rows: list[dict[str, object]] = []
    for index, count in enumerate(counts):
        lower = edges[index]
        upper = edges[index + 1]
        rows.append(
            {
                "lower_m": None if not np.isfinite(lower) else float(lower),
                "upper_m": None if not np.isfinite(upper) else float(upper),
                "count": int(count),
            }
        )
    return rows


def _summary(
    target: np.ndarray,
    exposure: np.ndarray,
    *,
    near_plane_m: float,
) -> dict[str, object]:
    if target.ndim != 2 or target.shape[1] != len(exposure):
        raise ValueError("target and exposure do not align")
    unique_mask = exposure > 0
    unique_values = target[:, unique_mask]
    expanded_exposure = np.broadcast_to(exposure, target.shape)
    repeated_values = np.repeat(target.ravel(), expanded_exposure.ravel())
    near_unique = np.abs(unique_values) <= near_plane_m
    near_repeated = np.abs(repeated_values) <= near_plane_m
    per_event_near = near_unique.sum(axis=1)
    quantile_levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    quantiles = np.quantile(unique_values, quantile_levels)
    return {
        "events": int(target.shape[0]),
        "history_length": int(target.shape[1]),
        "trainable_unique_samples": int(unique_values.size),
        "crop_sample_exposures": int(repeated_values.size),
        "near_plane_m": near_plane_m,
        "near_plane_unique_samples": int(near_unique.sum()),
        "near_plane_unique_fraction": float(near_unique.mean()),
        "near_plane_crop_exposures": int(near_repeated.sum()),
        "near_plane_crop_fraction": float(near_repeated.mean()),
        "per_event_near_unique_min": int(per_event_near.min()),
        "per_event_near_unique_median": float(np.median(per_event_near)),
        "per_event_near_unique_max": int(per_event_near.max()),
        "events_with_near_plane_support": int(np.count_nonzero(per_event_near)),
        "unique_target_quantiles_m": {
            str(level): float(value) for level, value in zip(quantile_levels, quantiles)
        },
        "unique_target_histogram": _histogram(unique_values, DEFAULT_BIN_EDGES_M),
        "crop_exposure_target_histogram": _histogram(
            repeated_values, DEFAULT_BIN_EDGES_M
        ),
        "pre_event_index": int(target.shape[1] - 2),
        "pre_event_target_m": {
            "min": float(target[:, -2].min()),
            "median": float(np.median(target[:, -2])),
            "max": float(target[:, -2].max()),
        },
        "event_index": int(target.shape[1] - 1),
        "event_target_m": {
            "min": float(target[:, -1].min()),
            "median": float(np.median(target[:, -1])),
            "max": float(target[:, -1].max()),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    source_hashes_before = {
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    if materialization.get("contract") != MATERIALIZATION_SCHEMA or not materialization.get(
        "process_passed"
    ):
        raise RuntimeError("materialization contract did not pass")
    expected_hashes = materialization.get("output_sha256", {})
    if expected_hashes.get("train") != source_hashes_before["train"]:
        raise RuntimeError("train dataset hash mismatch")
    if expected_hashes.get("validation") != source_hashes_before["validation"]:
        raise RuntimeError("validation dataset hash mismatch")

    summaries: dict[str, object] = {}
    exposure: np.ndarray | None = None
    source_shapes: dict[str, object] = {}
    for name, path in (
        ("train", args.train_dataset),
        ("validation", args.validation_dataset),
    ):
        with np.load(path) as loaded:
            if "tail" not in loaded.files or "vector_step" not in loaded.files:
                raise RuntimeError(f"{name} dataset lacks support-audit arrays")
            tail = loaded["tail"]
            vector_step = loaded["vector_step"]
        target = _plane_distance_m(tail)
        if exposure is None:
            exposure = _crop_exposure_counts(
                target.shape[1],
                args.crop_starts,
                sequence_length=args.sequence_length,
                burn_in=args.burn_in,
            )
        elif target.shape[1] != len(exposure):
            raise RuntimeError("train and validation history lengths differ")
        summaries[name] = _summary(
            target,
            exposure,
            near_plane_m=args.near_plane_m,
        )
        summaries[name]["distinct_vector_steps"] = int(len(np.unique(vector_step)))
        source_shapes[name] = {
            "tail": list(tail.shape),
            "vector_step": list(vector_step.shape),
        }
    assert exposure is not None

    source_hashes_after = {
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_contract": MATERIALIZATION_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "crop_starts": list(args.crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "exposure_by_history_index": exposure.tolist(),
        "source_shapes": source_shapes,
        "support": summaries,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "process_passed": bool(source_hashes_after == source_hashes_before),
        "model_loaded": 0,
        "model_tensor_changes": [],
        "optimizer_steps": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    args = parser.parse_args()
    try:
        args.crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    except ValueError as error:
        parser.error(f"invalid --crop-starts: {error}")
    if not args.crop_starts:
        parser.error("--crop-starts cannot be empty")
    if args.sequence_length <= 0 or args.burn_in < 0:
        parser.error("sequence length and burn-in are invalid")
    if args.near_plane_m <= 0.0:
        parser.error("near-plane bound must be positive")
    if args.report.exists():
        parser.error("support audit refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
