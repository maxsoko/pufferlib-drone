#!/usr/bin/env python3
"""Read-only validation audit of the N702 direct tangent progress readout."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index, _metrics
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    JOINT_SCHEMA,
    _canonical_actor_tangent_null_direction,
    _fixed_parent_reference,
    _load_partitioned_arrays,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _changed_model_keys,
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _soft_posterior_features,
    _target_batch,
)


AUDIT_SCHEMA = "vq2_tangent_validation_readout_audit_v1"
N702_REPORT_SHA256 = (
    "6c19ee663411a8eb50298581dee380385ab7a8e3c430b3ced8607532663408f8"
)


def _n702_rejection_exact(report: dict[str, object], hashes: dict[str, str]) -> bool:
    progress = report.get("selected_validation_progress")
    return bool(
        report.get("contract") == JOINT_SCHEMA
        and report.get("process_passed") is True
        and report.get("validation_admitted") is False
        and report.get("output_checkpoint_sha256") == hashes["child_checkpoint"]
        and report.get("source_hashes", {}).get("checkpoint")
        == hashes["parent_checkpoint"]
        and report.get("source_hashes", {}).get("dataset") == hashes["train_dataset"]
        and report.get("source_hashes", {}).get("validation_dataset")
        == hashes["validation_dataset"]
        and report.get("tangent_residual_enabled") is True
        and report.get("tangent_residual_weight") == 0.01
        and report.get("tangent_residual_scale") == 0.05
        and isinstance(progress, dict)
        and progress.get("direction_accuracy", math.inf) < 0.75
        and report.get("progress_gates", {}).get("direction_at_least_minimum")
        is False
        and report.get("sealed_test_dataset_path_received") == 0
        and report.get("sealed_test_dataset_opened") == 0
        and report.get("test_evaluations") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


@torch.no_grad()
def _direct_validation_prediction(
    parent,
    child,
    arrays: dict[str, np.ndarray],
    validation_events: np.ndarray,
    reference: dict[str, np.ndarray],
    event_to_reference: np.ndarray,
    direction: torch.Tensor,
    *,
    scale: float,
    target_mean: float,
    target_std: float,
    microbatch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    child.eval()
    full_length = arrays["mask"].shape[1] - 1
    direction = direction.to(device)
    for offset in range(0, len(validation_events), microbatch_size):
        chosen = validation_events[offset : offset + microbatch_size]
        legal, input_action = _legal_action_batch(
            arrays,
            chosen,
            np.zeros(len(chosen), dtype=np.int64),
            sequence_length=full_length,
            device=device,
        )
        state = _observe_legal_sequence(
            child,
            legal,
            input_action,
            _initial_state(arrays, chosen, device=device),
        )
        ref = event_to_reference[chosen]
        parent_det = torch.from_numpy(reference["deterministic"][ref]).to(device)
        parent_logits = torch.from_numpy(reference["logits"][ref]).to(device)
        parent_probability = parent_logits.float().softmax(-1)
        parent_feature = torch.cat(
            (parent_det.float(), parent_probability.flatten(-2)), -1
        )
        normalized = (
            (_soft_posterior_features(state) - parent_feature) @ direction
        ) / scale
        rows.append((normalized * target_std + target_mean).cpu().numpy())
    return np.concatenate(rows)


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "child_checkpoint": _sha256(args.child_checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "n702_report": _sha256(args.n702_report),
    }
    n702 = json.loads(args.n702_report.read_text(encoding="utf-8"))
    if source_before["n702_report"] != N702_REPORT_SHA256:
        raise RuntimeError("N702 report hash mismatch")
    if not _n702_rejection_exact(n702, source_before):
        raise RuntimeError("N702 rejection contract mismatch")
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    if (
        materialization.get("output_sha256", {}).get("train")
        != source_before["train_dataset"]
        or materialization.get("output_sha256", {}).get("validation")
        != source_before["validation_dataset"]
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("materialization contract mismatch")

    arrays, event_split = _load_partitioned_arrays(
        args.train_dataset, args.validation_dataset
    )
    validation_events = np.flatnonzero(event_split == 1)
    allowed_events = np.arange(len(event_split), dtype=np.int64)
    event_to_reference = np.arange(len(event_split), dtype=np.int64)
    parent_payload = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False
    )
    child_payload = torch.load(
        args.child_checkpoint, map_location=device, weights_only=False
    )
    parent = _model_from_payload(parent_payload, device)
    child = _model_from_payload(child_payload, device)
    parent_before = {
        name: value.detach().clone() for name, value in parent.state_dict().items()
    }
    child_before = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    direction, geometry = _canonical_actor_tangent_null_direction(
        parent.actor[0].weight,
        parent.rssm.deterministic_size,
        parent.rssm.stochastic_groups,
        parent.rssm.stochastic_classes,
    )
    reference = _fixed_parent_reference(
        parent,
        arrays,
        allowed_events,
        batch_size=args.fixed_reference_batch_size,
        device=device,
    )
    prediction = _direct_validation_prediction(
        parent,
        child,
        arrays,
        validation_events,
        reference,
        event_to_reference,
        direction,
        scale=float(n702["tangent_residual_scale"]),
        target_mean=float(n702["target_mean"]),
        target_std=float(n702["target_std"]),
        microbatch_size=args.microbatch_size,
        device=device,
    )
    crop_starts = tuple(int(value) for value in n702["crop_starts"])
    local_event, starts = _crop_index(len(validation_events), crop_starts)
    time_index = starts[:, None] + np.arange(
        int(n702["sequence_length"]), dtype=np.int64
    )[None, :]
    crop_prediction = prediction[local_event[:, None], time_index]
    target = _target_batch(
        arrays,
        validation_events[local_event],
        starts,
        sequence_length=int(n702["sequence_length"]),
    )
    metrics = _metrics(
        target,
        crop_prediction,
        evaluation_indices=np.asarray(n702["evaluation_indices"], dtype=np.int64),
        near_plane_m=1.0,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    progress_gates = {
        "correlation_at_least_minimum": metrics["correlation"] >= 0.80,
        "mae_at_most_maximum": metrics["mae_m"] <= 0.50,
        "near_plane_mae_at_most_maximum": metrics["near_plane_mae_m"] <= 0.60,
        "direction_at_least_minimum": metrics["direction_accuracy"] >= 0.75,
    }
    parent_changed, _ = _changed_model_keys(parent_before, parent.state_dict())
    child_changed, _ = _changed_model_keys(child_before, child.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "child_checkpoint": _sha256(args.child_checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "n702_report": _sha256(args.n702_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "models_bit_exact": parent_changed == [] and child_changed == [],
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "geometry_exact": (
            geometry["actor_first_layer_residual_max_abs"] <= 1e-8
            and geometry["simplex_tangent_residual_max_abs"] <= 1e-8
            and geometry["stochastic_direction_norm"] > 0.0
        ),
        "validation_only": int((event_split == 1).sum()) == 64,
        "no_test_path": True,
    }
    process_passed = all(process_gates.values())
    direct_readout_admitted = process_passed and all(progress_gates.values())
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "direct_validation_progress": metrics,
        "joint_probe_validation_progress": n702["selected_validation_progress"],
        "progress_gates": progress_gates,
        "direct_readout_admitted": direct_readout_admitted,
        "tangent_geometry": geometry,
        "parent_model_tensor_changes": parent_changed,
        "child_model_tensor_changes": child_changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "train_rows_scored": 0,
        "validation_events": int(len(validation_events)),
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "native_environment_created": 0,
        "flightsim_packets": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
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
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--child-checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--n702-report", type=Path, required=True)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("tangent readout audit refuses to overwrite report")
    if args.fixed_reference_batch_size != 416:
        parser.error("fixed reference batch size must remain 416")
    if args.microbatch_size <= 0:
        parser.error("microbatch size must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
