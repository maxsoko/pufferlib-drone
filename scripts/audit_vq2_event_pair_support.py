#!/usr/bin/env python3
"""Audit temporal-pair exposure in isolated VQ2 train/validation histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_event_partition_support import (
    AUDIT_SCHEMA as SAMPLE_SUPPORT_SCHEMA,
    _plane_distance_m,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


PAIR_AUDIT_SCHEMA = "vq2_event_partition_plane_pair_support_audit_v1"


def _crop_pair_exposure_counts(
    history_length: int,
    crop_starts: tuple[int, ...],
    *,
    sequence_length: int,
    burn_in: int,
) -> np.ndarray:
    """Count crops contributing each pair ending at a history index."""
    if history_length <= 0 or sequence_length <= 1 or burn_in < 0:
        raise ValueError("history, sequence length, and burn-in are invalid")
    if burn_in >= sequence_length - 1:
        raise ValueError("burn-in leaves no temporal pair")
    exposure = np.zeros(history_length, dtype=np.int64)
    for start in crop_starts:
        if start < 0 or start + sequence_length > history_length:
            raise ValueError("crop falls outside the source history")
        exposure[start + burn_in + 1 : start + sequence_length] += 1
    return exposure


def _pair_summary(
    target: np.ndarray,
    pair_exposure: np.ndarray,
    *,
    near_plane_m: float,
    minimum_pair_delta_m: float,
    near_plane_multiplier: float,
) -> dict[str, object]:
    target = np.asarray(target, dtype=np.float64)
    pair_exposure = np.asarray(pair_exposure, dtype=np.int64)
    if target.ndim != 2 or target.shape[1] != len(pair_exposure):
        raise ValueError("target and pair exposure do not align")
    endpoint_exposure = pair_exposure[1:]
    unique_endpoint = endpoint_exposure > 0
    delta = np.diff(target, axis=1)
    near_pair = (np.abs(target[:, :-1]) <= near_plane_m) | (
        np.abs(target[:, 1:]) <= near_plane_m
    )
    unique_delta = delta[:, unique_endpoint]
    unique_near = near_pair[:, unique_endpoint]
    repeated_delta = np.repeat(
        delta.ravel(), np.broadcast_to(endpoint_exposure, delta.shape).ravel()
    )
    repeated_near = np.repeat(
        near_pair.ravel(), np.broadcast_to(endpoint_exposure, delta.shape).ravel()
    )
    directional_unique = np.abs(unique_delta) >= minimum_pair_delta_m
    near_count = int(unique_near.sum())
    far_count = int(unique_near.size - near_count)
    weighted_near_fraction = float(
        near_plane_multiplier * near_count
        / (near_plane_multiplier * near_count + far_count)
    )
    values, counts = np.unique(endpoint_exposure[unique_endpoint], return_counts=True)
    return {
        "events": int(target.shape[0]),
        "unique_pairs": int(unique_delta.size),
        "crop_pair_exposures": int(repeated_delta.size),
        "near_plane_unique_pairs": near_count,
        "near_plane_unique_fraction": float(unique_near.mean()),
        "near_plane_crop_pair_exposures": int(repeated_near.sum()),
        "near_plane_crop_fraction": float(repeated_near.mean()),
        "directional_unique_pairs": int(directional_unique.sum()),
        "directional_unique_fraction": float(directional_unique.mean()),
        "inverse_exposure_near_multiplier": near_plane_multiplier,
        "expected_weighted_near_fraction": weighted_near_fraction,
        "pair_endpoint_exposure_histogram": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
        "pair_delta_m": {
            "min": float(unique_delta.min()),
            "median": float(np.median(unique_delta)),
            "max": float(unique_delta.max()),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    source_before = {
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "sample_support_report": _sha256(args.sample_support_report),
    }
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    sample_support = json.loads(
        args.sample_support_report.read_text(encoding="utf-8")
    )
    if (
        materialization.get("contract") != MATERIALIZATION_SCHEMA
        or not materialization.get("process_passed")
        or materialization.get("output_sha256", {}).get("train")
        != source_before["train"]
        or materialization.get("output_sha256", {}).get("validation")
        != source_before["validation"]
    ):
        raise RuntimeError("materialization contract mismatch")
    if (
        sample_support.get("contract") != SAMPLE_SUPPORT_SCHEMA
        or not sample_support.get("process_passed")
        or sample_support.get("source_hashes")
        != {
            "train": source_before["train"],
            "validation": source_before["validation"],
            "materialization_report": source_before["materialization_report"],
        }
        or sample_support.get("test_dataset_opened") != 0
    ):
        raise RuntimeError("sample support contract mismatch")

    pair_exposure: np.ndarray | None = None
    summaries: dict[str, object] = {}
    for name, path in (
        ("train", args.train_dataset),
        ("validation", args.validation_dataset),
    ):
        with np.load(path) as loaded:
            if "tail" not in loaded.files:
                raise RuntimeError(f"{name} corpus lacks tail")
            target = _plane_distance_m(loaded["tail"])
        if pair_exposure is None:
            pair_exposure = _crop_pair_exposure_counts(
                target.shape[1],
                args.crop_starts,
                sequence_length=args.sequence_length,
                burn_in=args.burn_in,
            )
        elif target.shape[1] != len(pair_exposure):
            raise RuntimeError("train and validation history lengths differ")
        summaries[name] = _pair_summary(
            target,
            pair_exposure,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
            near_plane_multiplier=args.near_plane_multiplier,
        )
    assert pair_exposure is not None

    source_after = {
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "sample_support_report": _sha256(args.sample_support_report),
    }
    report: dict[str, object] = {
        "contract": PAIR_AUDIT_SCHEMA,
        "source_contract": SAMPLE_SUPPORT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "crop_starts": list(args.crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "near_plane_m": args.near_plane_m,
        "minimum_pair_delta_m": args.minimum_pair_delta_m,
        "near_plane_multiplier": args.near_plane_multiplier,
        "pair_exposure_by_endpoint_history_index": pair_exposure.tolist(),
        "pair_support": summaries,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "process_passed": bool(source_after == source_before),
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
    parser.add_argument("--sample-support-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    args = parser.parse_args()
    try:
        args.crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    except ValueError as error:
        parser.error(f"invalid crop starts: {error}")
    if (
        not args.crop_starts
        or args.sequence_length <= 1
        or args.burn_in < 0
        or args.near_plane_m <= 0.0
        or args.minimum_pair_delta_m < 0.0
        or args.near_plane_multiplier <= 0.0
    ):
        parser.error("pair-support parameters are invalid")
    if args.report.exists():
        parser.error("pair support audit refuses to overwrite report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))

