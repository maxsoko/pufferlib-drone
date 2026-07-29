#!/usr/bin/env python3
"""Audit decoder-free rotation/sign-invariant progress geometry on N681 train."""

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

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_partition_support import _crop_exposure_counts
from scripts.audit_vq2_event_pair_support import _crop_pair_exposure_counts
from scripts.audit_vq2_probe_gradient_alignment import (
    COMPONENTS,
    _component_parameters,
    _sha256,
    _source_contracts,
    _target_normalization,
)
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _fixed_parent_reference,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    _changed_model_keys,
    _crop_initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _soft_posterior_features,
    _target_batch,
)


AUDIT_SCHEMA = "vq2_rotation_invariant_progress_geometry_audit_v1"
N696_REPORT_SHA256 = (
    "6cab01feb0cbebdd9f0778c58c8b427b7afc5a764dc66fdbd62336542e74ca9d"
)
N703_REPORT_SHA256 = (
    "48772b4d5b08432b173e4935cc565a80d7673c2a73af0f25b51e6ca8b29ba2c4"
)


def _weighted_covariance_score(
    feature: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Squared vector covariance normalized by target and total feature variance."""
    if feature.shape[:-1] != target.shape or target.shape != weight.shape:
        raise ValueError("covariance feature, target, and weight shapes must align")
    if feature.shape[-1] <= 0 or bool((weight <= 0.0).any()):
        raise ValueError("covariance features and weights are invalid")
    flat_feature = feature.float().reshape(-1, feature.shape[-1])
    flat_target = target.float().reshape(-1)
    flat_weight = weight.float().reshape(-1)
    total_weight = flat_weight.sum()
    normalized_weight = flat_weight / total_weight
    feature_mean = (normalized_weight[:, None] * flat_feature).sum(0)
    target_mean = (normalized_weight * flat_target).sum()
    centered_feature = flat_feature - feature_mean
    centered_target = flat_target - target_mean
    cross = (
        normalized_weight[:, None]
        * centered_feature
        * centered_target[:, None]
    ).sum(0)
    feature_variance = (
        normalized_weight[:, None] * centered_feature.square()
    ).sum()
    target_variance = (normalized_weight * centered_target.square()).sum()
    denominator = feature_variance * target_variance + epsilon
    score = cross.square().sum() / denominator
    return score, {
        "score": float(score.detach().cpu()),
        "feature_trace_variance": float(feature_variance.detach().cpu()),
        "target_variance": float(target_variance.detach().cpu()),
        "cross_covariance_norm": float(
            torch.linalg.vector_norm(cross).detach().cpu()
        ),
        "weight_sum": float(total_weight.detach().cpu()),
        "samples": int(flat_target.numel()),
    }


def _householder_rotate(feature: torch.Tensor) -> torch.Tensor:
    """Apply a fixed dense orthogonal reflection without materializing a matrix."""
    direction = torch.arange(
        1, feature.shape[-1] + 1, dtype=feature.dtype, device=feature.device
    )
    direction = direction / torch.linalg.vector_norm(direction)
    return feature - 2.0 * (feature @ direction)[..., None] * direction


def _progress_geometry_loss(
    feature: torch.Tensor,
    normalized_target: torch.Tensor,
    level_weight: torch.Tensor,
    pair_weight: torch.Tensor,
    *,
    level_weight_coefficient: float,
    delta_weight_coefficient: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    if feature.shape[:-1] != normalized_target.shape:
        raise ValueError("progress feature and target shapes must align")
    if feature.shape[1] < 2 or pair_weight.shape != normalized_target[:, 1:].shape:
        raise ValueError("progress temporal-pair geometry is invalid")
    if level_weight_coefficient <= 0.0 or delta_weight_coefficient <= 0.0:
        raise ValueError("progress geometry coefficients must be positive")
    level_score, level_metrics = _weighted_covariance_score(
        feature, normalized_target, level_weight
    )
    feature_delta = torch.diff(feature, dim=1)
    target_delta = torch.diff(normalized_target, dim=1)
    delta_score, delta_metrics = _weighted_covariance_score(
        feature_delta, target_delta, pair_weight
    )
    loss = (
        level_weight_coefficient * (1.0 - level_score)
        + delta_weight_coefficient * (1.0 - delta_score)
    )
    return loss, {
        "level": level_metrics,
        "delta": delta_metrics,
        "loss": float(loss.detach().cpu()),
    }


def _loss_weights(
    target_m: torch.Tensor,
    crop_starts: np.ndarray,
    exposure: np.ndarray,
    pair_exposure: np.ndarray,
    *,
    burn_in: int,
    near_plane_m: float,
    near_plane_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    level_index = crop_starts[:, None] + np.arange(
        burn_in, burn_in + target_m.shape[1], dtype=np.int64
    )[None, :]
    level_weight = torch.from_numpy(
        (1.0 / np.asarray(exposure, dtype=np.float64)[level_index]).astype(
            np.float32
        )
    ).to(target_m.device)
    level_weight = level_weight * torch.where(
        target_m.abs() <= near_plane_m,
        torch.as_tensor(near_plane_multiplier, device=target_m.device),
        torch.ones((), device=target_m.device),
    )
    endpoint_index = crop_starts[:, None] + np.arange(
        burn_in + 1, burn_in + target_m.shape[1], dtype=np.int64
    )[None, :]
    pair_weight = torch.from_numpy(
        (1.0 / np.asarray(pair_exposure, dtype=np.float64)[endpoint_index]).astype(
            np.float32
        )
    ).to(target_m.device)
    near_pair = (target_m[:, :-1].abs() <= near_plane_m) | (
        target_m[:, 1:].abs() <= near_plane_m
    )
    pair_weight = pair_weight * torch.where(
        near_pair,
        torch.as_tensor(near_plane_multiplier, device=target_m.device),
        torch.ones((), device=target_m.device),
    )
    return level_weight, pair_weight


def _rejection_contracts(args: argparse.Namespace) -> tuple[bool, bool]:
    n696 = json.loads(args.n696_report.read_text(encoding="utf-8"))
    n703 = json.loads(args.n703_report.read_text(encoding="utf-8"))
    n696_exact = bool(
        _sha256(args.n696_report) == N696_REPORT_SHA256
        and n696.get("contract") == "vq2_probe_representation_gradient_alignment_v1"
        and n696.get("process_passed") is True
        and n696.get("alignment_admitted") is False
        and n696.get("model_tensor_changes") == []
        and n696.get("test_dataset_opened") == 0
    )
    n703_exact = bool(
        _sha256(args.n703_report) == N703_REPORT_SHA256
        and n703.get("contract") == "vq2_tangent_validation_readout_audit_v1"
        and n703.get("process_passed") is True
        and n703.get("direct_readout_admitted") is False
        and n703.get("parent_model_tensor_changes") == []
        and n703.get("child_model_tensor_changes") == []
        and n703.get("test_dataset_path_received") == 0
        and n703.get("test_dataset_opened") == 0
        and n703.get("optimizer_steps") == 0
        and n703.get("checkpoint_written") == 0
        and n703.get("flightsim_packets") == 0
    )
    return n696_exact, n703_exact


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
        "pair_support_report": _sha256(args.pair_support_report),
        "reference_batch_audit_report": _sha256(
            args.reference_batch_audit_report
        ),
        "n696_report": _sha256(args.n696_report),
        "n703_report": _sha256(args.n703_report),
    }
    contracts_exact = _source_contracts(args, source_before)
    n696_exact, n703_exact = _rejection_contracts(args)
    with np.load(args.train_dataset) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if len(arrays["vector_step"]) != args.expected_train_events:
        raise RuntimeError("train event count does not match preregistration")
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    phase_after = np.rint(arrays["phase_after_event"].astype(np.float64) * 6.0)
    phase_before = np.rint(
        arrays["tail"][:, :-1, legal_tail_size + 32].astype(np.float64) * 6.0
    )
    if not np.all(phase_before == 0) or not np.all(phase_after == 1):
        raise RuntimeError("train successful-prefix phase contract mismatch")
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    original_event, crop_start = _crop_index(len(phase_after), crop_starts)
    exposure = _crop_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
    )
    pair_exposure = _crop_pair_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
    )
    target_all = _target_batch(
        arrays, original_event, crop_start, sequence_length=args.sequence_length
    )
    target_mean, target_std = _target_normalization(
        target_all, crop_start, exposure, burn_in=args.burn_in
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grouped = _component_parameters(model)
    for name in COMPONENTS:
        for parameter in grouped[name]:
            parameter.requires_grad_(True)
    event_ids = np.arange(len(phase_after), dtype=np.int64)
    reference = _fixed_parent_reference(
        model,
        arrays,
        event_ids,
        batch_size=args.fixed_reference_batch_size,
        device=device,
    )
    boundary = {
        "deterministic_max_abs_error": float(
            np.abs(
                reference["deterministic"][:, -1]
                - arrays["preevent_deterministic"]
            ).max()
        ),
        "logits_max_abs_error": float(
            np.abs(
                reference["logits"][:, -1] - arrays["preevent_logits"]
            ).max()
        ),
        "stochastic_index_mismatch_count": int(
            (
                reference["logits"][:, -1].argmax(-1)
                != arrays["preevent_stochastic_index"]
            ).sum()
        ),
    }
    chosen = np.random.default_rng(args.audit_batch_seed).choice(
        len(original_event), args.batch_size, replace=False
    )
    event = original_event[chosen]
    starts = crop_start[chosen]
    legal, action = _legal_action_batch(
        arrays,
        event,
        starts,
        sequence_length=args.sequence_length,
        device=device,
    )
    state = _observe_legal_sequence(
        model,
        legal,
        action,
        _crop_initial_state(
            arrays, reference, event, event, starts, device=device
        ),
    )
    feature = _soft_posterior_features(state)[:, args.burn_in :]
    target_np = _target_batch(
        arrays, event, starts, sequence_length=args.sequence_length
    )[:, args.burn_in :]
    target_m = torch.from_numpy(target_np).to(device)
    normalized_target = (target_m - target_mean) / target_std
    level_weight, pair_weight = _loss_weights(
        target_m,
        starts,
        exposure,
        pair_exposure,
        burn_in=args.burn_in,
        near_plane_m=args.near_plane_m,
        near_plane_multiplier=args.near_plane_multiplier,
    )
    loss, metrics = _progress_geometry_loss(
        feature,
        normalized_target,
        level_weight,
        pair_weight,
        level_weight_coefficient=args.level_weight,
        delta_weight_coefficient=args.delta_weight,
    )
    rotated_loss, rotated_metrics = _progress_geometry_loss(
        _householder_rotate(feature),
        normalized_target,
        level_weight,
        pair_weight,
        level_weight_coefficient=args.level_weight,
        delta_weight_coefficient=args.delta_weight,
    )
    flipped_loss, flipped_metrics = _progress_geometry_loss(
        feature,
        -normalized_target,
        level_weight,
        pair_weight,
        level_weight_coefficient=args.level_weight,
        delta_weight_coefficient=args.delta_weight,
    )
    flat_parameters = [
        parameter for name in COMPONENTS for parameter in grouped[name]
    ]
    gradients = torch.autograd.grad(loss, flat_parameters, allow_unused=False)
    gradient_norms: dict[str, float] = {}
    gradient_maximum: dict[str, float] = {}
    cursor = 0
    gradients_finite = True
    for name in COMPONENTS:
        count = len(grouped[name])
        selected = gradients[cursor : cursor + count]
        cursor += count
        gradients_finite = gradients_finite and all(
            bool(torch.isfinite(gradient).all()) for gradient in selected
        )
        gradient_norms[name] = math.sqrt(
            sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in selected
            )
        )
        gradient_maximum[name] = max(
            float(gradient.detach().abs().max().cpu()) for gradient in selected
        )
    changed, _forbidden = _changed_model_keys(model_before, model.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
        "pair_support_report": _sha256(args.pair_support_report),
        "reference_batch_audit_report": _sha256(
            args.reference_batch_audit_report
        ),
        "n696_report": _sha256(args.n696_report),
        "n703_report": _sha256(args.n703_report),
    }
    invariance = {
        "orthogonal_rotation_loss_abs_error": abs(
            float(rotated_loss.detach().cpu()) - float(loss.detach().cpu())
        ),
        "target_sign_flip_loss_abs_error": abs(
            float(flipped_loss.detach().cpu()) - float(loss.detach().cpu())
        ),
        "rotation_level_score_abs_error": abs(
            rotated_metrics["level"]["score"] - metrics["level"]["score"]
        ),
        "rotation_delta_score_abs_error": abs(
            rotated_metrics["delta"]["score"] - metrics["delta"]["score"]
        ),
        "sign_level_score_abs_error": abs(
            flipped_metrics["level"]["score"] - metrics["level"]["score"]
        ),
        "sign_delta_score_abs_error": abs(
            flipped_metrics["delta"]["score"] - metrics["delta"]["score"]
        ),
    }
    mechanism_gates = {
        "loss_finite": math.isfinite(float(loss.detach().cpu())),
        "level_score_finite_nonzero": (
            math.isfinite(metrics["level"]["score"])
            and metrics["level"]["score"] > args.minimum_score
        ),
        "delta_score_finite_nonzero": (
            math.isfinite(metrics["delta"]["score"])
            and metrics["delta"]["score"] > args.minimum_score
        ),
        "level_feature_variance_noncollapsed": (
            metrics["level"]["feature_trace_variance"]
            >= args.minimum_feature_variance
        ),
        "delta_feature_variance_noncollapsed": (
            metrics["delta"]["feature_trace_variance"]
            >= args.minimum_feature_variance
        ),
        "level_target_variance_noncollapsed": (
            metrics["level"]["target_variance"]
            >= args.minimum_target_variance
        ),
        "delta_target_variance_noncollapsed": (
            metrics["delta"]["target_variance"]
            >= args.minimum_delta_target_variance
        ),
        "orthogonal_rotation_invariant": max(
            invariance["orthogonal_rotation_loss_abs_error"],
            invariance["rotation_level_score_abs_error"],
            invariance["rotation_delta_score_abs_error"],
        )
        <= args.maximum_invariance_error,
        "target_sign_invariant": max(
            invariance["target_sign_flip_loss_abs_error"],
            invariance["sign_level_score_abs_error"],
            invariance["sign_delta_score_abs_error"],
        )
        <= args.maximum_invariance_error,
        "component_gradients_finite": gradients_finite,
        "component_gradients_nonzero": all(
            value > 0.0 for value in gradient_norms.values()
        ),
    }
    process_gates = {
        "sources_unchanged": source_before == source_after,
        "source_contracts_exact": contracts_exact,
        "n696_rejection_exact": n696_exact,
        "n703_rejection_exact": n703_exact,
        "train_event_count_exact": len(phase_after) == args.expected_train_events,
        "train_crop_count_exact": len(original_event) == args.expected_train_crops,
        "fixed_reference_boundary_passes": (
            boundary["deterministic_max_abs_error"] <= 0.002
            and boundary["logits_max_abs_error"] <= 0.002
            and boundary["stochastic_index_mismatch_count"] == 0
        ),
        "model_tensors_exact": changed == [],
        "no_validation_or_test_path": True,
    }
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    process_passed = all(process_gates.values()) and resource_gate
    mechanism_admitted = process_passed and all(mechanism_gates.values())
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "mechanism_gates": mechanism_gates,
        "mechanism_admitted": mechanism_admitted,
        "objective_metrics": metrics,
        "invariance": invariance,
        "gradient_norms": gradient_norms,
        "gradient_maximum_abs": gradient_maximum,
        "audit_batch": chosen.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "boundary_replay": boundary,
        "train_events": int(len(phase_after)),
        "train_crops": int(len(original_event)),
        "model_tensor_changes": changed,
        "optimizer_steps": 0,
        "checkpoint_written": 0,
        "validation_dataset_path_received": 0,
        "validation_dataset_opened": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
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
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--pair-support-report", type=Path, required=True)
    parser.add_argument("--reference-batch-audit-report", type=Path, required=True)
    parser.add_argument("--n696-report", type=Path, required=True)
    parser.add_argument("--n703-report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--audit-batch-seed", type=int, default=704)
    parser.add_argument("--expected-train-events", type=int, default=272)
    parser.add_argument("--expected-train-crops", type=int, default=1904)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--level-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--minimum-score", type=float, default=1e-8)
    parser.add_argument("--minimum-feature-variance", type=float, default=1e-6)
    parser.add_argument("--minimum-target-variance", type=float, default=1e-4)
    parser.add_argument("--minimum-delta-target-variance", type=float, default=1e-8)
    parser.add_argument("--maximum-invariance-error", type=float, default=1e-6)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("rotation-invariant audit refuses to overwrite report")
    for name in (
        "sequence_length",
        "burn_in",
        "batch_size",
        "fixed_reference_batch_size",
        "expected_train_events",
        "expected_train_crops",
        "maximum_cuda_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    if args.fixed_reference_batch_size != 416:
        parser.error("fixed reference batch size must remain 416")
    if args.near_plane_m <= 0.0 or args.near_plane_multiplier <= 0.0:
        parser.error("near-plane settings must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run_audit(parse_args()), indent=2, sort_keys=True))
