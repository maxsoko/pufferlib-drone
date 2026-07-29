#!/usr/bin/env python3
"""Audit CUDA batch-shape sensitivity of N587 recurrent prefix replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_event_partition_support import AUDIT_SCHEMA as SUPPORT_SCHEMA
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _load_partitioned_arrays,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _model_from_payload,
    _parent_reference,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


BATCH_AUDIT_SCHEMA = "vq2_partition_reference_batch_sensitivity_v1"


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"invalid batch size: {error}") from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("batch sizes must be unique positive integers")
    return sizes


def _fixed_batch_event_chunks(
    event_ids: np.ndarray, batch_size: int
) -> list[tuple[np.ndarray, int]]:
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if event_ids.ndim != 1 or len(event_ids) == 0 or batch_size <= 0:
        raise ValueError("event ids and fixed batch size are invalid")
    chunks: list[tuple[np.ndarray, int]] = []
    for offset in range(0, len(event_ids), batch_size):
        chosen = event_ids[offset : offset + batch_size]
        valid = len(chosen)
        if valid < batch_size:
            padding = np.resize(event_ids, batch_size - valid)
            chosen = np.concatenate((chosen, padding))
        chunks.append((chosen, valid))
    return chunks


def _event_max_abs(error: np.ndarray) -> np.ndarray:
    error = np.asarray(error)
    if error.ndim < 2 or len(error) == 0:
        raise ValueError("event error must have an event axis and features")
    return error.reshape(len(error), -1).max(axis=1)


def _fixed_batch_parent_reference(
    model,
    arrays: dict[str, np.ndarray],
    event_ids: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "deterministic": [],
        "logits": [],
        "action": [],
    }
    for chosen, valid in _fixed_batch_event_chunks(event_ids, batch_size):
        reference = _parent_reference(
            model,
            arrays,
            chosen,
            microbatch_size=batch_size,
            device=device,
        )
        for name in rows:
            rows[name].append(reference[name][:valid])
    return {name: np.concatenate(value) for name, value in rows.items()}


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "checkpoint": _sha256(args.checkpoint),
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
    }
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    support = json.loads(args.support_report.read_text(encoding="utf-8"))
    if (
        materialization.get("contract") != MATERIALIZATION_SCHEMA
        or not materialization.get("process_passed")
        or materialization.get("output_sha256", {}).get("train")
        != source_before["train"]
        or materialization.get("output_sha256", {}).get("validation")
        != source_before["validation"]
    ):
        raise RuntimeError("materialized partition source mismatch")
    if (
        support.get("contract") != SUPPORT_SCHEMA
        or not support.get("process_passed")
        or support.get("source_hashes")
        != {
            "train": source_before["train"],
            "validation": source_before["validation"],
            "materialization_report": source_before["materialization_report"],
        }
        or support.get("test_dataset_opened") != 0
    ):
        raise RuntimeError("support-audit source mismatch")

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    model.eval()
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    arrays, event_split = _load_partitioned_arrays(
        args.train_dataset, args.validation_dataset
    )
    event_ids = np.arange(len(event_split), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        reference = _fixed_batch_parent_reference(
            model,
            arrays,
            event_ids,
            batch_size=batch_size,
            device=device,
        )
        deterministic_error = np.abs(
            reference["deterministic"][:, -1] - arrays["preevent_deterministic"]
        )
        logits_error = np.abs(
            reference["logits"][:, -1] - arrays["preevent_logits"]
        )
        mismatch = (
            reference["logits"][:, -1].argmax(-1)
            != arrays["preevent_stochastic_index"]
        )
        rows.append(
            {
                "microbatch_size": batch_size,
                "deterministic_max_abs_error": float(deterministic_error.max()),
                "deterministic_p99_event_max_abs_error": float(
                    np.quantile(_event_max_abs(deterministic_error), 0.99)
                ),
                "logits_max_abs_error": float(logits_error.max()),
                "stochastic_index_mismatch_count": int(mismatch.sum()),
            }
        )

    source_after = {
        "checkpoint": _sha256(args.checkpoint),
        "train": _sha256(args.train_dataset),
        "validation": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
    }
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, parent_state[name])
    ]
    report: dict[str, object] = {
        "contract": BATCH_AUDIT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "events": int(len(event_split)),
        "train_events": int(np.count_nonzero(event_split == 0)),
        "validation_events": int(np.count_nonzero(event_split == 1)),
        "test_events": 0,
        "batch_results": rows,
        "fixed_batch_padding": True,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "deterministic_max_abs_error",
                "deterministic_p99_event_max_abs_error",
                "logits_max_abs_error",
            )
        ),
        "model_tensor_changes": model_changes,
        "process_passed": bool(
            source_after == source_before
            and not model_changes
            and all(
                math.isfinite(float(row[key]))
                for row in rows
                for key in (
                    "deterministic_max_abs_error",
                    "deterministic_p99_event_max_abs_error",
                    "logits_max_abs_error",
                )
            )
        ),
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "optimizer_steps": 0,
        "actor_updates": 0,
        "world_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "device": str(device),
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="4,64,336")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    try:
        args.batch_sizes = _parse_batch_sizes(args.batch_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.report.exists():
        parser.error("batch audit refuses to overwrite its report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
