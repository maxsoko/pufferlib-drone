#!/usr/bin/env python3
"""Fit the fixed train-only ridge and admit or reject frozen N707 features."""

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

from scripts.audit_vq2_event_partition_support import _crop_exposure_counts
from scripts.audit_vq2_frozen_ridge_readout import (
    _continuous_features,
    _fit_sha256,
    _ridge_predict,
    _weighted_standardized_ridge,
)
from scripts.audit_vq2_probe_gradient_alignment import _sha256
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


AUDIT_SCHEMA = "vq2_invariant_donor_frozen_ridge_admission_v1"
N707_REPORT_SHA256 = (
    "2bb3f31cf135af1315fa651055b5ba4bb2b7046191c4f84ce1322c76097ad181"
)
N707_CHECKPOINT_SHA256 = (
    "8f415f85086a2779335523cd77e7ad6400a85182b6954a1ccc08cb042c1d7253"
)
N704_REPORT_SHA256 = (
    "73e701069af3d25c9bc6578602871fda787f9523be734f09fe46386251aed733"
)
N706_REPORT_SHA256 = (
    "b5f9a26dda344486595fdd85f0f4a80242e887f36d362d6c92a9bc52a303e3de"
)
N706_EXECUTABLE_SHA256 = (
    "cb99bdd8b9ad231c9e1e2becc9cd3bd1e2776d4deabc85dc0c7e6e707f7e0d33"
)
N707_TRAINER_SHA256 = (
    "0338fb128ef47cf0bf7d818583b603ae9e2b4aad8c4377d59dfd924efe96f23b"
)
TRAIN_SHA256 = "6bb788c0ae27264fa8a68b75500228d9be3269e80bfa454c8d01f8bb654d91ec"
VALIDATION_SHA256 = (
    "c8ded2bd9191a5c548b3bfc1f0a7142b66e0d75558ff80c883819075a5f48ad1"
)
MATERIALIZATION_SHA256 = (
    "7e2dd5c1b5cefb6f15fca7109c35ddb7115a7fc192b69a0186599218727778ac"
)

MINIMUM_CORRELATION = 0.80
MAXIMUM_MAE_M = 0.50
MAXIMUM_NEAR_PLANE_MAE_M = 0.60
MINIMUM_DIRECTION_ACCURACY = 0.75
MAXIMUM_ACTION_RMSE = 0.01
MAXIMUM_ACTION_MAX_ABS = 0.02
MAXIMUM_PARAMETER_DELTA = 0.005001


def _n707_contract_gates(
    report: dict[str, object], hashes: dict[str, str]
) -> dict[str, bool]:
    sampling = report.get("sampling_summary")
    preservation = report.get("selected_preservation_gates")
    validation_action = report.get("selected_validation_action_drift")
    dense_action = report.get("selected_dense_drift")
    invariant_gradient = report.get("maximum_invariant_gradient_abs_by_component")
    source = report.get("source_hashes")
    if not isinstance(sampling, dict):
        sampling = {}
    if not isinstance(preservation, dict):
        preservation = {}
    if not isinstance(validation_action, dict):
        validation_action = {}
    if not isinstance(dense_action, dict):
        dense_action = {}
    dense_hard_action = dense_action.get("hard_action", {})
    if not isinstance(dense_hard_action, dict):
        dense_hard_action = {}
    if not isinstance(invariant_gradient, dict):
        invariant_gradient = {}
    if not isinstance(source, dict):
        source = {}
    return {
        "n707_report_hash_exact": hashes["n707_report"] == N707_REPORT_SHA256,
        "n707_checkpoint_hash_exact": (
            hashes["checkpoint"] == N707_CHECKPOINT_SHA256
            and report.get("output_checkpoint_sha256") == hashes["checkpoint"]
        ),
        "n707_contract_exact": report.get("contract") == JOINT_SCHEMA,
        "n707_process_passed": report.get("process_passed") is True,
        "n707_representation_training_passed": (
            report.get("representation_training_passed") is True
        ),
        "n707_final_step_exact": (
            report.get("fixed_final_selection") is True
            and report.get("selection_mode") == "fixed_final_step"
            and report.get("selected_step") == 357
            and report.get("optimizer_steps") == 357
            and report.get("representation_updates") == 357
        ),
        "n707_sampling_exact": (
            report.get("balanced_epoch_sampling") is True
            and sampling.get("train_draws") == 5712
            and sampling.get("train_count_minimum") == 3
            and sampling.get("train_count_maximum") == 3
            and sampling.get("train_event_draw_minimum") == 21
            and sampling.get("train_event_draw_maximum") == 21
            and sampling.get("dense_draws") == 2856
            and sampling.get("dense_count_minimum") == 22
            and sampling.get("dense_count_maximum") == 23
            and sampling.get("train_unseen_crops") == 0
        ),
        "n707_invariant_contract_exact": (
            report.get("invariant_progress_enabled") is True
            and report.get("invariant_progress_weight") == 1.0
            and report.get("invariant_level_coefficient") == 1.0
            and report.get("invariant_delta_coefficient") == 1.0
            and report.get("detach_probe_representation") is True
            and report.get("invariant_source_contract_sha256")
            == N704_REPORT_SHA256
            and report.get("frozen_readout_smoke_sha256") == N706_REPORT_SHA256
            and all(
                float(invariant_gradient.get(name, 0.0)) > 0.0
                for name in ("encoder", "sequence", "posterior")
            )
        ),
        "n707_sources_exact": (
            source.get("executable") == N707_TRAINER_SHA256
            and source.get("dataset") == TRAIN_SHA256
            and source.get("validation_dataset") == VALIDATION_SHA256
            and source.get("materialization_report") == MATERIALIZATION_SHA256
            and source.get("invariant_source_contract_report")
            == N704_REPORT_SHA256
            and source.get("frozen_readout_smoke_report") == N706_REPORT_SHA256
        ),
        "n707_preservation_exact": bool(preservation)
        and all(value is True for value in preservation.values()),
        "n707_action_drift_bounded": (
            float(validation_action.get("rmse", math.inf)) <= MAXIMUM_ACTION_RMSE
            and float(validation_action.get("max_abs", math.inf))
            <= MAXIMUM_ACTION_MAX_ABS
            and float(dense_hard_action.get("rmse", math.inf)) <= MAXIMUM_ACTION_RMSE
            and float(dense_hard_action.get("max_abs", math.inf))
            <= MAXIMUM_ACTION_MAX_ABS
        ),
        "n707_parameter_delta_bounded": (
            float(report.get("final_maximum_parameter_delta", math.inf))
            <= MAXIMUM_PARAMETER_DELTA
        ),
        "n707_runtime_isolation_exact": (
            report.get("actor_updates") == 0
            and report.get("critic_updates") == 0
            and report.get("world_updates") == 0
            and report.get("reward_updates") == 0
            and report.get("collected_transitions") == 0
            and report.get("native_environment_created") == 0
            and report.get("flightsim_packets") == 0
            and report.get("sealed_test_dataset_path_received") == 0
            and report.get("sealed_test_dataset_opened") == 0
        ),
    }


def _fixed_admission_gates(metrics: dict[str, object]) -> dict[str, bool]:
    return {
        "correlation_at_least_0p80": (
            float(metrics.get("correlation", -math.inf)) >= MINIMUM_CORRELATION
        ),
        "mae_at_most_0p50_m": (
            float(metrics.get("mae_m", math.inf)) <= MAXIMUM_MAE_M
        ),
        "near_plane_mae_at_most_0p60_m": (
            float(metrics.get("near_plane_mae_m", math.inf))
            <= MAXIMUM_NEAR_PLANE_MAE_M
        ),
        "direction_accuracy_at_least_0p75": (
            float(metrics.get("direction_accuracy", -math.inf))
            >= MINIMUM_DIRECTION_ACCURACY
        ),
    }


def _covered_history_exact(covered: np.ndarray) -> bool:
    return bool(
        len(covered) == 288
        and int(covered[0]) == 32
        and int(covered[-1]) == 319
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    helper_path = ROOT / "scripts/audit_vq2_frozen_ridge_readout.py"
    source_before = {
        "executable": _sha256(Path(__file__)),
        "ridge_helper": _sha256(helper_path),
        "checkpoint": _sha256(args.checkpoint),
        "n707_report": _sha256(args.n707_report),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    n707 = json.loads(args.n707_report.read_text(encoding="utf-8"))
    n707_gates = _n707_contract_gates(n707, source_before)
    if not all(n707_gates.values()):
        failed = sorted(name for name, passed in n707_gates.items() if not passed)
        raise RuntimeError(f"N707 donor contract mismatch: {failed}")
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if (
        source_before["ridge_helper"] != N706_EXECUTABLE_SHA256
        or source_before["train_dataset"] != TRAIN_SHA256
        or source_before["validation_dataset"] != VALIDATION_SHA256
        or source_before["materialization_report"] != MATERIALIZATION_SHA256
        or materialization.get("contract") != MATERIALIZATION_SCHEMA
        or materialization.get("process_passed") is not True
        or materialization.get("output_sha256", {}).get("train")
        != source_before["train_dataset"]
        or materialization.get("output_sha256", {}).get("validation")
        != source_before["validation_dataset"]
        or materialization.get("test_rows_scored") != 0
    ):
        raise RuntimeError("materialization or fixed ridge helper contract mismatch")

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
        arrays, train_events, np.zeros(len(train_events), dtype=np.int64),
        sequence_length=full_length,
    )
    validation_target = _target_batch(
        arrays, validation_events, np.zeros(len(validation_events), dtype=np.int64),
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
        np.abs(fit_target) <= args.near_plane_m, args.near_plane_multiplier, 1.0
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
        crop_target, crop_prediction,
        evaluation_indices=np.arange(
            args.burn_in - 1, args.sequence_length, args.evaluation_stride,
            dtype=np.int64,
        ),
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    admission_gates = _fixed_admission_gates(metrics)
    train_error = train_prediction - fit_target
    changed, _ = _changed_model_keys(model_before, model.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "ridge_helper": _sha256(helper_path),
        "checkpoint": _sha256(args.checkpoint),
        "n707_report": _sha256(args.n707_report),
        "train_dataset": _sha256(args.train_dataset),
        "validation_dataset": _sha256(args.validation_dataset),
        "materialization_report": _sha256(args.materialization_report),
    }
    process_gates = {
        "sources_unchanged": source_after == source_before,
        "n707_contract_exact": all(n707_gates.values()),
        "fixed_ridge_helper_exact": source_after["ridge_helper"] == N706_EXECUTABLE_SHA256,
        "model_tensors_exact": changed == [],
        "fit_deterministic": fit_hash_one == fit_hash_two,
        "fixed_single_lambda": args.ridge == 0.001,
        "float64_solver": np.asarray(fit_one["coefficient"]).dtype == np.float64,
        "all_512_features_active": fit_summary["active_features"] == 512,
        "exact_train_fit_rows": fit_summary["rows"] == 78336,
        "exact_event_counts": len(train_events) == 272 and len(validation_events) == 64,
        "covered_history_exact": _covered_history_exact(covered),
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
    readout_admitted = process_passed and all(admission_gates.values())
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "n707_contract_gates": n707_gates,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "fixed_admission_thresholds": {
            "minimum_correlation": MINIMUM_CORRELATION,
            "maximum_mae_m": MAXIMUM_MAE_M,
            "maximum_near_plane_mae_m": MAXIMUM_NEAR_PLANE_MAE_M,
            "minimum_direction_accuracy": MINIMUM_DIRECTION_ACCURACY,
        },
        "admission_gates": admission_gates,
        "readout_admitted": readout_admitted,
        "fit": fit_summary,
        "fit_sha256": fit_hash_one,
        "covered_history_indices": covered.tolist(),
        "covered_history_index_count": int(len(covered)),
        "train_fit_rmse_m": float(np.sqrt(np.mean(np.square(train_error)))),
        "train_fit_mae_m": float(np.abs(train_error).mean()),
        "validation_progress": metrics,
        "validation_used_for_fit_or_hyperparameter_selection": 0,
        "validation_used_for_fixed_donor_admission": 1,
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
    parser.add_argument("--n707-report", type=Path, required=True)
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
        parser.error("invariant donor ridge audit refuses to overwrite report")
    if args.ridge != 0.001:
        parser.error("ridge must remain fixed at 0.001")
    if args.microbatch_size <= 0 or args.minimum_feature_std <= 0.0:
        parser.error("microbatch and feature std must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
