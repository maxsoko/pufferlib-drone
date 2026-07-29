#!/usr/bin/env python3
"""Freeze label-blind whole-vector-step partitions for VQ2 event windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_vq2_gate_event_windows import _sha256_bytes
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.filter_vq2_event_windows_replay import FILTER_SCHEMA


PARTITION_SCHEMA = "vq2_event_window_whole_vector_step_partitions_v1"
TRAIN = np.uint8(0)
VALIDATION = np.uint8(1)
TEST = np.uint8(2)
SPLIT_NAMES = {int(TRAIN): "train", int(VALIDATION): "validation", int(TEST): "test"}


def _assign_group_partitions(
    vector_step: np.ndarray,
    *,
    seed: int,
    minimum_train_events: int,
    minimum_validation_events: int,
    minimum_test_events: int,
) -> np.ndarray:
    """Assign whole vector-step groups without consulting any event label."""
    vector_step = np.asarray(vector_step)
    if vector_step.ndim != 1 or vector_step.size == 0:
        raise ValueError("vector_step must be a nonempty vector")
    if not np.issubdtype(vector_step.dtype, np.integer):
        raise ValueError("vector_step must be integral")
    minima = (minimum_train_events, minimum_validation_events, minimum_test_events)
    if any(value <= 0 for value in minima):
        raise ValueError("all split minima must be positive")
    if sum(minima) > vector_step.size:
        raise RuntimeError("event corpus cannot satisfy requested split minima")

    groups, inverse, counts = np.unique(
        vector_step.astype(np.int64, copy=False), return_inverse=True, return_counts=True
    )
    order = np.random.default_rng(seed).permutation(len(groups))
    group_split = np.full(len(groups), TRAIN, dtype=np.uint8)
    cursor = 0
    for split_id, target in (
        (VALIDATION, minimum_validation_events),
        (TEST, minimum_test_events),
    ):
        assigned = 0
        while assigned < target:
            if cursor >= len(order):
                raise RuntimeError("whole-group assignment exhausted the corpus")
            group_index = int(order[cursor])
            cursor += 1
            group_split[group_index] = split_id
            assigned += int(counts[group_index])

    event_split = group_split[inverse]
    train_count = int(np.count_nonzero(event_split == TRAIN))
    if train_count < minimum_train_events:
        raise RuntimeError(
            f"whole-group assignment leaves only {train_count} train events"
        )
    return event_split


def freeze(args: argparse.Namespace) -> dict[str, object]:
    source_hashes_before = {
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("contract") != FILTER_SCHEMA:
        raise RuntimeError("source report is not the replay-filter contract")
    if not source_report.get("process_passed"):
        raise RuntimeError("source replay filter did not pass")
    if source_report.get("output_dataset_sha256") != source_hashes_before["dataset"]:
        raise RuntimeError("source report dataset hash mismatch")

    with np.load(args.dataset) as loaded:
        if "vector_step" not in loaded.files:
            raise RuntimeError("source corpus lacks vector_step")
        source_array_names = sorted(loaded.files)
        vector_step = loaded["vector_step"].copy()
    expected_events = int(source_report.get("selected_events", -1))
    if vector_step.shape != (expected_events,):
        raise RuntimeError("vector_step does not align with selected event count")

    event_split = _assign_group_partitions(
        vector_step,
        seed=args.seed,
        minimum_train_events=args.minimum_train_events,
        minimum_validation_events=args.minimum_validation_events,
        minimum_test_events=args.minimum_test_events,
    )
    source_event_index = np.arange(len(vector_step), dtype=np.int32)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.output_manifest.with_suffix(
        args.output_manifest.suffix + ".tmp"
    )
    with temporary_manifest.open("wb") as stream:
        np.savez_compressed(
            stream,
            source_event_index=source_event_index,
            vector_step=vector_step.astype(np.int32, copy=False),
            event_split=event_split,
        )
    temporary_manifest.replace(args.output_manifest)

    source_hashes_after = {
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    event_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    group_sets: dict[str, set[int]] = {}
    for split_id, name in SPLIT_NAMES.items():
        selected = event_split == split_id
        event_counts[name] = int(np.count_nonzero(selected))
        groups = set(int(value) for value in np.unique(vector_step[selected]))
        group_sets[name] = groups
        group_counts[name] = len(groups)
    overlap = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    minima_pass = bool(
        event_counts["train"] >= args.minimum_train_events
        and event_counts["validation"] >= args.minimum_validation_events
        and event_counts["test"] >= args.minimum_test_events
    )
    report: dict[str, object] = {
        "contract": PARTITION_SCHEMA,
        "source_contract": FILTER_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "source_events": int(len(vector_step)),
        "source_distinct_vector_steps": int(len(np.unique(vector_step))),
        "seed": args.seed,
        "minimum_event_counts": {
            "train": args.minimum_train_events,
            "validation": args.minimum_validation_events,
            "test": args.minimum_test_events,
        },
        "event_counts": event_counts,
        "group_counts": group_counts,
        "group_overlap_counts": overlap,
        "partition_inputs_loaded": ["vector_step"],
        "source_array_names_not_loaded_for_partitioning": [
            name for name in source_array_names if name != "vector_step"
        ],
        "geometry_or_target_quality_inspected": 0,
        "reward_inspected": 0,
        "observation_inspected": 0,
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": _sha256_bytes(args.output_manifest),
        "output_manifest_bytes": args.output_manifest.stat().st_size,
        "process_passed": bool(
            source_hashes_after == source_hashes_before
            and minima_pass
            and all(value == 0 for value in overlap.values())
        ),
        "model_loaded": 0,
        "model_tensor_changes": [],
        "optimizer_steps": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-train-events", type=int, default=256)
    parser.add_argument("--minimum-validation-events", type=int, default=64)
    parser.add_argument("--minimum-test-events", type=int, default=64)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.report.exists():
        parser.error("partition freeze refuses to overwrite manifest or report")
    return args


if __name__ == "__main__":
    print(json.dumps(freeze(parse_args()), indent=2, sort_keys=True))
