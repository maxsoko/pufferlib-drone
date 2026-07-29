#!/usr/bin/env python3
"""Mechanically materialize frozen VQ2 event-window partitions.

This tool interprets no observation, reward, action, geometry, or privileged
target. It copies rows according to a previously frozen whole-vector-step
manifest so future development can open train/validation files without opening
the sealed test file.
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

from scripts.collect_vq2_gate_event_windows import _sha256_bytes
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.filter_vq2_event_windows_replay import FILTER_SCHEMA
from scripts.freeze_vq2_event_window_partitions import (
    PARTITION_SCHEMA,
    SPLIT_NAMES,
    TEST,
    TRAIN,
    VALIDATION,
)


MATERIALIZATION_SCHEMA = "vq2_event_window_partition_materialization_v1"


def _validated_split_indices(
    source_vector_step: np.ndarray,
    source_event_index: np.ndarray,
    manifest_vector_step: np.ndarray,
    event_split: np.ndarray,
) -> dict[int, np.ndarray]:
    source_vector_step = np.asarray(source_vector_step)
    source_event_index = np.asarray(source_event_index)
    manifest_vector_step = np.asarray(manifest_vector_step)
    event_split = np.asarray(event_split)
    event_count = len(source_vector_step)
    if any(
        value.ndim != 1 or len(value) != event_count
        for value in (source_event_index, manifest_vector_step, event_split)
    ):
        raise ValueError("manifest arrays must align with source events")
    if not all(
        np.issubdtype(value.dtype, np.integer)
        for value in (source_vector_step, source_event_index, manifest_vector_step, event_split)
    ):
        raise ValueError("source and manifest routing arrays must be integral")
    expected = np.arange(event_count, dtype=np.int64)
    if not np.array_equal(np.sort(source_event_index.astype(np.int64)), expected):
        raise RuntimeError("source_event_index is not a permutation of source rows")
    source_event_index_i64 = source_event_index.astype(np.int64, copy=False)
    if not np.array_equal(
        source_vector_step[source_event_index_i64], manifest_vector_step
    ):
        raise RuntimeError("manifest vector_step does not match source rows")
    allowed = np.asarray([TRAIN, VALIDATION, TEST], dtype=event_split.dtype)
    if not np.isin(event_split, allowed).all():
        raise RuntimeError("manifest contains an unknown split id")
    indices = {
        split_id: source_event_index_i64[event_split == split_id]
        for split_id in SPLIT_NAMES
    }
    if sum(len(value) for value in indices.values()) != event_count:
        raise RuntimeError("manifest does not assign every source event once")
    return indices


def _slice_aligned_arrays(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("source corpus is empty")
    event_count = int(next(iter(arrays.values())).shape[0])
    for name, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != event_count:
            raise ValueError(f"source array {name} does not align on axis zero")
    return {name: value[indices].copy() for name, value in arrays.items()}


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def materialize(args: argparse.Namespace) -> dict[str, object]:
    source_hashes_before = {
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
        "partition_manifest": _sha256_bytes(args.partition_manifest),
        "partition_report": _sha256(args.partition_report),
    }
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    partition_report = json.loads(
        args.partition_report.read_text(encoding="utf-8")
    )
    if source_report.get("contract") != FILTER_SCHEMA or not source_report.get(
        "process_passed"
    ):
        raise RuntimeError("source replay filter contract did not pass")
    if source_report.get("output_dataset_sha256") != source_hashes_before["dataset"]:
        raise RuntimeError("source dataset hash mismatch")
    if partition_report.get("contract") != PARTITION_SCHEMA or not partition_report.get(
        "process_passed"
    ):
        raise RuntimeError("partition freeze contract did not pass")
    if (
        partition_report.get("output_manifest_sha256")
        != source_hashes_before["partition_manifest"]
    ):
        raise RuntimeError("partition manifest hash mismatch")
    if partition_report.get("source_hashes") != {
        "dataset": source_hashes_before["dataset"],
        "source_report": source_hashes_before["source_report"],
    }:
        raise RuntimeError("partition report source hashes mismatch")

    with np.load(args.dataset) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if "vector_step" not in arrays:
        raise RuntimeError("source corpus lacks vector_step")
    with np.load(args.partition_manifest) as loaded:
        required = {"source_event_index", "vector_step", "event_split"}
        if set(loaded.files) != required:
            raise RuntimeError("partition manifest arrays differ from contract")
        source_event_index = loaded["source_event_index"].copy()
        manifest_vector_step = loaded["vector_step"].copy()
        event_split = loaded["event_split"].copy()
    split_indices = _validated_split_indices(
        arrays["vector_step"],
        source_event_index,
        manifest_vector_step,
        event_split,
    )

    output_paths = {
        int(TRAIN): args.train_dataset,
        int(VALIDATION): args.validation_dataset,
        int(TEST): args.test_dataset,
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    output_bytes: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    exact_copy_by_split: dict[str, bool] = {}
    for split_id, name in SPLIT_NAMES.items():
        selected = split_indices[split_id]
        split_arrays = _slice_aligned_arrays(arrays, selected)
        path = output_paths[split_id]
        _write_npz_once(path, split_arrays)
        exact_copy = True
        with np.load(path) as copied:
            if set(copied.files) != set(arrays):
                exact_copy = False
            else:
                for array_name, source_value in arrays.items():
                    if not np.array_equal(copied[array_name], source_value[selected]):
                        exact_copy = False
                        break
        exact_copy_by_split[name] = exact_copy
        event_counts[name] = int(len(selected))
        output_hashes[name] = _sha256_bytes(path)
        output_bytes[name] = path.stat().st_size

    source_hashes_after = {
        "dataset": _sha256_bytes(args.dataset),
        "source_report": _sha256(args.source_report),
        "partition_manifest": _sha256_bytes(args.partition_manifest),
        "partition_report": _sha256(args.partition_report),
    }
    expected_counts = partition_report.get("event_counts")
    report: dict[str, object] = {
        "contract": MATERIALIZATION_SCHEMA,
        "source_contract": FILTER_SCHEMA,
        "partition_contract": PARTITION_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "source_events": int(len(arrays["vector_step"])),
        "source_arrays_copied": sorted(arrays),
        "event_counts": event_counts,
        "expected_event_counts": expected_counts,
        "exact_copy_by_split": exact_copy_by_split,
        "output_paths": {
            SPLIT_NAMES[split_id]: str(path)
            for split_id, path in output_paths.items()
        },
        "output_sha256": output_hashes,
        "output_bytes": output_bytes,
        "content_metrics_computed": 0,
        "geometry_or_target_quality_interpreted": 0,
        "test_rows_scored": 0,
        "test_values_used_for_selection": 0,
        "process_passed": bool(
            source_hashes_after == source_hashes_before
            and event_counts == expected_counts
            and all(exact_copy_by_split.values())
            and sum(event_counts.values()) == len(arrays["vector_step"])
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
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--partition-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outputs = (
        args.train_dataset,
        args.validation_dataset,
        args.test_dataset,
        args.report,
    )
    if len(set(outputs)) != len(outputs):
        parser.error("materialization outputs must be distinct")
    if any(path.exists() for path in outputs):
        parser.error("materializer refuses to overwrite an output")
    return args


if __name__ == "__main__":
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))

