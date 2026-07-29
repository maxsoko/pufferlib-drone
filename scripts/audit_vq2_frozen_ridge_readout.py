#!/usr/bin/env python3
"""Fit one fixed train-only ridge on a frozen VQ2 representation and score validation."""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.audit_vq2_event_partition_support import _crop_exposure_counts
from scripts.audit_vq2_probe_gradient_alignment import _sha256, _target_normalization
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index, _metrics
from scripts.continue_vq2_success_prefix_representation_joint_offline import JOINT_SCHEMA
from scripts.continue_vq2_success_prefix_representation_offline import (
    _changed_model_keys,
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _soft_posterior_features,
    _target_batch,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA


AUDIT_SCHEMA = "vq2_frozen_train_ridge_readout_audit_v1"
N705_REPORT_SHA256 = (
    "c3e7fdb5da897c1042e388bc007a9cea6fb42fc1df2e84b07daa3af99bb63f75"
)


def _weighted_standardized_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    ridge: float,
    minimum_feature_std: float,
) -> tuple[dict[str, np.ndarray | float], dict[str, object]]:
    feature = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if (
        feature.ndim != 2
        or len(feature) != len(target)
        or len(target) != len(weight)
        or len(target) == 0
        or np.any(weight <= 0.0)
        or ridge <= 0.0
        or minimum_feature_std <= 0.0
    ):
        raise ValueError("weighted ridge inputs are invalid")
    normalized_weight = weight / weight.sum()
    feature_mean = (normalized_weight[:, None] * feature).sum(0)
    centered = feature - feature_mean
    feature_std = np.sqrt(
        (normalized_weight[:, None] * np.square(centered)).sum(0)
    )
    active = feature_std >= minimum_feature_std
    if not active.any():
        raise RuntimeError("ridge has no active feature")
    standardized = centered[:, active] / feature_std[active]
    target_mean = float(np.dot(normalized_weight, target))
    target_centered = target - target_mean
    target_std = float(
        np.sqrt(np.dot(normalized_weight, np.square(target_centered)))
    )
    if not math.isfinite(target_std) or target_std <= 0.0:
        raise RuntimeError("ridge target variance is invalid")
    normalized_target = target_centered / target_std
    design = np.concatenate(
        (standardized, np.ones((len(standardized), 1), dtype=np.float64)), 1
    )
    weighted_design = design * normalized_weight[:, None]
    gram = design.T @ weighted_design
    rhs = design.T @ (normalized_weight * normalized_target)
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
    regularizer[-1, -1] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, rhs)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "active": active,
        "target_mean": target_mean,
        "target_std": target_std,
        "coefficient": coefficient,
    }, {
        "rows": int(len(feature)),
        "input_features": int(feature.shape[1]),
        "active_features": int(active.sum()),
        "weight_sum": float(weight.sum()),
        "target_mean": target_mean,
        "target_std": target_std,
        "ridge": ridge,
        "minimum_feature_std": minimum_feature_std,
        "gram_condition_number": float(np.linalg.cond(gram + regularizer)),
    }


def _ridge_predict(fit: dict[str, np.ndarray | float], feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(fit["feature_mean"])
    std = np.asarray(fit["feature_std"])
    active = np.asarray(fit["active"], dtype=bool)
    coefficient = np.asarray(fit["coefficient"])
    standardized = (feature[:, active] - mean[active]) / std[active]
    design = np.concatenate(
        (standardized, np.ones((len(standardized), 1), dtype=np.float64)), 1
    )
    return (
        design @ coefficient * float(fit["target_std"])
        + float(fit["target_mean"])
    )


def _fit_sha256(fit: dict[str, np.ndarray | float]) -> str:
    digest = hashlib.sha256()
    for name in ("feature_mean", "feature_std", "active", "coefficient"):
        value = np.ascontiguousarray(fit[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    digest.update(np.asarray([fit["target_mean"], fit["target_std"]], dtype=np.float64).tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _continuous_features(
    model,
    arrays: dict[str, np.ndarray],
    event_ids: np.ndarray,
    *,
    microbatch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    full_length = arrays["mask"].shape[1] - 1
    model.eval()
    for offset in range(0, len(event_ids), microbatch_size):
        chosen = event_ids[offset : offset + microbatch_size]
        legal, action = _legal_action_batch(
            arrays,
            chosen,
            np.zeros(len(chosen), dtype=np.int64),
            sequence_length=full_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model,
            legal,
            action,
            _initial_state(arrays, chosen, device=device),
        )
        rows.append(_soft_posterior_features(state).float().cpu().numpy())
    return np.concatenate(rows)


def _n705_contract_exact(report: dict[str, object], hashes: dict[str, str]) -> bool:
    invariant_gradient = report.get("maximum_invariant_gradient_abs_by_component")
    return bool(
        hashes["n705_report"] == N705_REPORT_SHA256
        and report.get("contract") == JOINT_SCHEMA
        and report.get("process_passed") is True
        and report.get("output_checkpoint_sha256") == hashes["checkpoint"]
        and report.get("invariant_progress_enabled") is True
        and report.get("invariant_progress_weight") == 1.0
        and report.get("detach_probe_representation") is True
        and isinstance(invariant_gradient, dict)
        and all(float(invariant_gradient.get(name, 0.0)) > 0.0 for name in ("encoder", "sequence", "posterior"))
        and report.get("sealed_test_dataset_path_received") == 0
        and report.get("sealed_test_dataset_opened") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n705_report": _sha256(args.n705_report),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    n705 = json.loads(args.n705_report.read_text(encoding="utf-8"))
    if not _n705_contract_exact(n705, source_before):
        raise RuntimeError("N705 smoke contract mismatch")
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("output_sha256", {}).get("train") != source_before["train_dataset"]
        or materialization.get("output_sha256", {}).get("validation") != source_before["validation_dataset"]
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("materialization contract mismatch")
    with np.load(args.train_dataset) as loaded:
        train = {name: loaded[name].copy() for name in loaded.files}
    with np.load(args.validation_dataset) as loaded:
        validation = {name: loaded[name].copy() for name in loaded.files}
    if set(train) != set(validation):
        raise RuntimeError("train and validation arrays differ")
    arrays = {name: np.concatenate((train[name], validation[name]), 0) for name in train}
    train_events = np.arange(len(train["vector_step"]), dtype=np.int64)
    validation_events = np.arange(len(train_events), len(arrays["vector_step"]), dtype=np.int64)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train_feature = _continuous_features(
        model, arrays, train_events, microbatch_size=args.microbatch_size, device=device
    )
    validation_feature = _continuous_features(
        model, arrays, validation_events, microbatch_size=args.microbatch_size, device=device
    )
    full_length = arrays["mask"].shape[1] - 1
    train_target = _target_batch(
        arrays,
        train_events,
        np.zeros(len(train_events), dtype=np.int64),
        sequence_length=full_length,
    )
    validation_target = _target_batch(
        arrays,
        validation_events,
        np.zeros(len(validation_events), dtype=np.int64),
        sequence_length=full_length,
    )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    exposure = _crop_exposure_counts(
        arrays["mask"].shape[1], crop_starts,
        sequence_length=args.sequence_length, burn_in=args.burn_in,
    )
    covered = np.flatnonzero(exposure > 0)
    covered = covered[covered < full_length]
    fit_feature = train_feature[:, covered].reshape(-1, train_feature.shape[-1])
    fit_target = train_target[:, covered].reshape(-1)
    fit_weight = np.where(
        np.abs(fit_target) <= args.near_plane_m,
        args.near_plane_multiplier,
        1.0,
    ).astype(np.float64)
    fit_one, fit_summary = _weighted_standardized_ridge(
        fit_feature, fit_target, fit_weight,
        ridge=args.ridge, minimum_feature_std=args.minimum_feature_std,
    )
    fit_two, _ = _weighted_standardized_ridge(
        fit_feature, fit_target, fit_weight,
        ridge=args.ridge, minimum_feature_std=args.minimum_feature_std,
    )
    fit_hash_one = _fit_sha256(fit_one)
    fit_hash_two = _fit_sha256(fit_two)
    train_prediction = _ridge_predict(fit_one, fit_feature)
    validation_prediction = _ridge_predict(
        fit_one, validation_feature.reshape(-1, validation_feature.shape[-1])
    ).reshape(validation_feature.shape[:2])
    local_event, starts = _crop_index(len(validation_events), crop_starts)
    time_index = starts[:, None] + np.arange(args.sequence_length, dtype=np.int64)[None, :]
    crop_prediction = validation_prediction[local_event[:, None], time_index]
    crop_target = validation_target[local_event[:, None], time_index]
    metrics = _metrics(
        crop_target,
        crop_prediction,
        evaluation_indices=np.arange(
            args.burn_in - 1, args.sequence_length, args.evaluation_stride,
            dtype=np.int64,
        ),
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    train_error = train_prediction - fit_target
    changed, _ = _changed_model_keys(model_before, model.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "n705_report": _sha256(args.n705_report),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "model_tensors_exact": changed == [],
        "fit_deterministic": fit_hash_one == fit_hash_two,
        "fixed_single_lambda": args.ridge == 0.001,
        "float64_solver": np.asarray(fit_one["coefficient"]).dtype == np.float64,
        "train_rows_nonempty": len(fit_target) > 0,
        "validation_prediction_finite": bool(np.isfinite(validation_prediction).all()),
        "validation_metrics_finite": all(
            math.isfinite(float(metrics[name]))
            for name in ("correlation", "mae_m", "near_plane_mae_m", "direction_accuracy")
        ),
        "no_test_path": True,
    }
    cuda_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    process_passed = all(process_gates.values()) and resource_gate
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "readout_smoke_passed": process_passed,
        "fit": fit_summary,
        "fit_sha256": fit_hash_one,
        "covered_history_indices": covered.tolist(),
        "covered_history_index_count": int(len(covered)),
        "train_fit_rmse_m": float(np.sqrt(np.mean(np.square(train_error)))),
        "train_fit_mae_m": float(np.abs(train_error).mean()),
        "validation_progress_diagnostic_only": metrics,
        "validation_used_for_fit_or_selection": 0,
        "ridge_candidates": [args.ridge],
        "model_tensor_changes": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "train_events": int(len(train_events)),
        "validation_events": int(len(validation_events)),
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "test_rows_scored": 0,
        "native_environment_created": 0,
        "flightsim_packets": 0,
        "resource_gate": resource_gate,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = cuda_peak
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n705-report", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--minimum-feature-std", type=float, default=1e-6)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("frozen ridge audit refuses to overwrite report")
    if args.ridge != 0.001:
        parser.error("ridge must remain fixed at 0.001")
    if args.microbatch_size <= 0 or args.minimum_feature_std <= 0.0:
        parser.error("microbatch and feature std must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
