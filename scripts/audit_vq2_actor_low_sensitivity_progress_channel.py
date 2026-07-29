#!/usr/bin/env python3
"""Audit a deterministic progress channel in the actor's least-sensitive direction."""

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
    _cache_frozen_features,
    _component_parameters,
    _sha256,
    _source_contracts,
    _target_normalization,
)
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _fixed_parent_reference,
    _weighted_delta_loss,
    _weighted_plane_loss,
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


AUDIT_SCHEMA = "vq2_actor_low_sensitivity_progress_channel_audit_v1"


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item) for item in value.split(",") if item.strip())
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("candidate scales must be positive")
    if tuple(sorted(set(scales))) != scales:
        raise ValueError("candidate scales must be unique and increasing")
    return scales


def _canonical_low_sensitivity_direction(
    actor_first_weight: torch.Tensor, deterministic_size: int
) -> tuple[torch.Tensor, dict[str, float | int]]:
    weight = actor_first_weight.detach().cpu().double()
    if weight.ndim != 2 or deterministic_size <= 0 or deterministic_size > weight.shape[1]:
        raise ValueError("actor first-layer geometry is invalid")
    _left, singular, right_h = torch.linalg.svd(
        weight[:, :deterministic_size], full_matrices=False
    )
    direction_det = right_h[-1]
    canonical_index = int(direction_det.abs().argmax())
    if direction_det[canonical_index] < 0.0:
        direction_det = -direction_det
    direction = torch.zeros(weight.shape[1], dtype=torch.float64)
    direction[:deterministic_size] = direction_det
    residual = weight @ direction
    stochastic = direction[deterministic_size:]
    return direction.float(), {
        "deterministic_size": deterministic_size,
        "feature_size": int(weight.shape[1]),
        "canonical_positive_index": canonical_index,
        "minimum_singular_value": float(singular[-1]),
        "second_minimum_singular_value": float(singular[-2]),
        "minimum_singular_gap_ratio": float(singular[-2] / singular[-1]),
        "actor_first_layer_residual_norm": float(torch.linalg.vector_norm(residual)),
        "actor_first_layer_residual_max_abs": float(residual.abs().max()),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "stochastic_direction_max_abs": (
            0.0 if stochastic.numel() == 0 else float(stochastic.abs().max())
        ),
    }


def _canonical_actor_tangent_null_direction(
    actor_first_weight: torch.Tensor,
    deterministic_size: int,
    stochastic_groups: int,
    stochastic_classes: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    weight = actor_first_weight.detach().cpu().double()
    stochastic_size = stochastic_groups * stochastic_classes
    if (
        weight.ndim != 2
        or deterministic_size <= 0
        or stochastic_groups <= 0
        or stochastic_classes <= 1
        or deterministic_size + stochastic_size != weight.shape[1]
    ):
        raise ValueError("actor/simplex tangent geometry is invalid")
    simplex = torch.zeros(
        (stochastic_groups, weight.shape[1]), dtype=torch.float64
    )
    for group in range(stochastic_groups):
        start = deterministic_size + group * stochastic_classes
        simplex[group, start : start + stochastic_classes] = 1.0
    constraints = torch.cat((weight, simplex), 0)
    gram = constraints @ constraints.T
    solved = torch.linalg.solve(gram, constraints[:, :deterministic_size])
    projected_norm_square = 1.0 - (
        constraints[:, :deterministic_size] * solved
    ).sum(0)
    canonical_index = int(projected_norm_square.argmax())
    basis = torch.zeros(weight.shape[1], dtype=torch.float64)
    basis[canonical_index] = 1.0
    direction = basis - constraints.T @ solved[:, canonical_index]
    direction = direction / torch.linalg.vector_norm(direction)
    if direction[canonical_index] < 0.0:
        direction = -direction
    actor_residual = weight @ direction
    simplex_residual = simplex @ direction
    stochastic = direction[deterministic_size:]
    return direction.float(), {
        "deterministic_size": deterministic_size,
        "feature_size": int(weight.shape[1]),
        "stochastic_groups": stochastic_groups,
        "stochastic_classes": stochastic_classes,
        "canonical_positive_index": canonical_index,
        "projected_basis_norm": float(projected_norm_square[canonical_index].sqrt()),
        "actor_first_layer_residual_norm": float(torch.linalg.vector_norm(actor_residual)),
        "actor_first_layer_residual_max_abs": float(actor_residual.abs().max()),
        "simplex_tangent_residual_max_abs": float(simplex_residual.abs().max()),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "deterministic_direction_norm": float(
            torch.linalg.vector_norm(direction[:deterministic_size])
        ),
        "stochastic_direction_norm": float(torch.linalg.vector_norm(stochastic)),
        "stochastic_direction_max_abs": float(stochastic.abs().max()),
    }


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weight: np.ndarray
) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    weight = np.asarray(weight, dtype=np.float64).ravel()
    if left.shape != right.shape or left.shape != weight.shape or np.any(weight <= 0.0):
        raise ValueError("weighted correlation arrays must align with positive weight")
    weight = weight / weight.sum()
    left_centered = left - np.dot(weight, left)
    right_centered = right - np.dot(weight, right)
    covariance = np.dot(weight, left_centered * right_centered)
    variance = math.sqrt(
        np.dot(weight, np.square(left_centered))
        * np.dot(weight, np.square(right_centered))
    )
    return float(covariance / variance)


def _drift_accumulator() -> dict[str, float | int]:
    return {"sum_square": 0.0, "maximum": 0.0, "count": 0}


def _add_drift(accumulator: dict[str, float | int], difference: torch.Tensor) -> None:
    value = difference.detach().float()
    accumulator["sum_square"] += float(value.square().sum().cpu())
    accumulator["maximum"] = max(
        float(accumulator["maximum"]), float(value.abs().max().cpu())
    )
    accumulator["count"] += value.numel()


def _finish_drift(accumulator: dict[str, float | int]) -> dict[str, float]:
    return {
        "rmse": math.sqrt(
            float(accumulator["sum_square"]) / int(accumulator["count"])
        ),
        "max_abs": float(accumulator["maximum"]),
    }


@torch.no_grad()
def _counterfactual_scale_metrics(
    model,
    feature_cache: torch.Tensor,
    normalized_target: np.ndarray,
    direction: torch.Tensor,
    weight: np.ndarray,
    *,
    scale: float,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    flat_feature = feature_cache.reshape(-1, feature_cache.shape[-1])
    flat_target = np.asarray(normalized_target, dtype=np.float32).reshape(-1)
    direction_device = direction.to(device)
    raw_drift = _drift_accumulator()
    action_drift = _drift_accumulator()
    bound_violations = 0
    values = 0
    negative_stochastic = 0
    stochastic_values = 0
    simplex_sum_max_abs = 0.0
    model.eval()
    for offset in range(0, len(flat_feature), microbatch_size):
        feature = flat_feature[offset : offset + microbatch_size].to(device)
        target = torch.from_numpy(
            flat_target[offset : offset + microbatch_size]
        ).to(device)
        adjusted = feature + scale * target[:, None] * direction_device
        raw_parent = model.actor(feature)
        raw_adjusted = model.actor(adjusted)
        _add_drift(raw_drift, raw_adjusted - raw_parent)
        parent_action = model.deterministic_actor_action(
            model.actor_distribution(feature)
        )
        adjusted_action = model.deterministic_actor_action(
            model.actor_distribution(adjusted)
        )
        _add_drift(action_drift, adjusted_action - parent_action)
        deterministic = adjusted[:, : model.rssm.deterministic_size]
        bound_violations += int((deterministic.abs() > 1.0 + 1e-6).sum().cpu())
        values += deterministic.numel()
        stochastic = adjusted[:, model.rssm.deterministic_size :].reshape(
            -1, model.rssm.stochastic_groups, model.rssm.stochastic_classes
        )
        negative_stochastic += int((stochastic < -1e-7).sum().cpu())
        stochastic_values += stochastic.numel()
        simplex_sum_max_abs = max(
            simplex_sum_max_abs,
            float((stochastic.sum(-1) - 1.0).abs().max().cpu()),
        )
    parent_projection = torch.matmul(feature_cache, direction.cpu()).numpy()
    adjusted_projection = parent_projection + scale * normalized_target
    reconstructed = (adjusted_projection - parent_projection) / scale
    residual_error = reconstructed - normalized_target
    parent_correlation = _weighted_correlation(
        parent_projection, normalized_target, weight
    )
    adjusted_correlation = _weighted_correlation(
        adjusted_projection, normalized_target, weight
    )
    return {
        "scale": scale,
        "raw_actor_drift": _finish_drift(raw_drift),
        "deterministic_action_drift": _finish_drift(action_drift),
        "deterministic_bound_violation_fraction": bound_violations / values,
        "negative_stochastic_probability_fraction": (
            negative_stochastic / stochastic_values
        ),
        "simplex_sum_max_abs_error": simplex_sum_max_abs,
        "parent_projection_correlation": parent_correlation,
        "adjusted_projection_correlation": adjusted_correlation,
        "absolute_correlation_improvement": adjusted_correlation
        - parent_correlation,
        "residual_readout_rmse": float(np.sqrt(np.mean(np.square(residual_error)))),
        "residual_readout_max_abs": float(np.abs(residual_error).max()),
        "finite": bool(
            math.isfinite(adjusted_correlation)
            and np.isfinite(residual_error).all()
        ),
    }


def _direct_gradient(
    model,
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    feature_cache: torch.Tensor,
    original_event: np.ndarray,
    crop_start: np.ndarray,
    chosen: np.ndarray,
    direction: torch.Tensor,
    exposure: np.ndarray,
    pair_exposure: np.ndarray,
    *,
    scale: float,
    sequence_length: int,
    burn_in: int,
    target_mean: float,
    target_std: float,
    near_plane_m: float,
    near_plane_multiplier: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    event = original_event[chosen]
    starts = crop_start[chosen]
    model.train()
    legal, action = _legal_action_batch(
        arrays,
        event,
        starts,
        sequence_length=sequence_length,
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
    child = _soft_posterior_features(state)[:, burn_in:]
    parent = feature_cache[chosen].to(device)
    prediction = ((child - parent) @ direction.to(device)) / scale
    target_np = _target_batch(
        arrays, event, starts, sequence_length=sequence_length
    )[:, burn_in:]
    target_m = torch.from_numpy(target_np).to(device)
    normalized = (target_m - target_mean) / target_std
    plane, _plane_weight = _weighted_plane_loss(
        prediction,
        normalized,
        target_m,
        starts,
        exposure,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    delta, _delta_weight = _weighted_delta_loss(
        prediction,
        normalized,
        target_m,
        starts,
        pair_exposure,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    loss = plane + delta
    grouped = _component_parameters(model)
    parameters = [parameter for name in COMPONENTS for parameter in grouped[name]]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    cursor = 0
    norms: dict[str, float] = {}
    maximum: dict[str, float] = {}
    for name in COMPONENTS:
        count = len(grouped[name])
        selected = gradients[cursor : cursor + count]
        norms[name] = math.sqrt(
            sum(float(gradient.detach().float().square().sum().cpu()) for gradient in selected)
        )
        maximum[name] = max(
            float(gradient.detach().abs().max().cpu()) for gradient in selected
        )
        cursor += count
    return norms, {
        "total": float(loss.detach().cpu()),
        "plane": float(plane.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        **{f"{name}_maximum_abs": value for name, value in maximum.items()},
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    scales = _parse_scales(args.candidate_scales)
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
        "reference_batch_audit_report": _sha256(args.reference_batch_audit_report),
        "n696_report": _sha256(args.n696_report),
        "n698_report": _sha256(args.n698_report),
    }
    contracts_exact = _source_contracts(args, source_before)
    n696 = json.loads(args.n696_report.read_text(encoding="utf-8"))
    n696_contract = bool(
        n696.get("contract") == "vq2_probe_representation_gradient_alignment_v1"
        and n696.get("process_passed")
        and not n696.get("alignment_admitted")
        and n696.get("model_tensor_changes") == []
        and n696.get("test_dataset_opened") == 0
    )
    n698 = json.loads(args.n698_report.read_text(encoding="utf-8"))
    n698_contract = bool(
        n698.get("contract") == AUDIT_SCHEMA
        and n698.get("process_passed")
        and not n698.get("channel_admitted")
        and n698.get("direction_mode") == "deterministic_low_sensitivity"
        and n698.get("direct_gradient_norms", {}).get("encoder") == 0.0
        and n698.get("direct_gradient_norms", {}).get("posterior") == 0.0
        and n698.get("model_tensor_changes") == []
        and n698.get("test_dataset_opened") == 0
    )
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
    normalized_target = (target_all[:, args.burn_in :] - target_mean) / target_std
    time_index = crop_start[:, None] + np.arange(
        args.burn_in, args.sequence_length, dtype=np.int64
    )[None, :]
    plane_weight = (1.0 / exposure[time_index]) * np.where(
        np.abs(target_all[:, args.burn_in :]) <= args.near_plane_m,
        args.near_plane_multiplier,
        1.0,
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    model_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.direction_mode == "deterministic_low_sensitivity":
        direction, geometry = _canonical_low_sensitivity_direction(
            model.actor[0].weight, model.rssm.deterministic_size
        )
    elif args.direction_mode == "actor_tangent_null":
        direction, geometry = _canonical_actor_tangent_null_direction(
            model.actor[0].weight,
            model.rssm.deterministic_size,
            model.rssm.stochastic_groups,
            model.rssm.stochastic_classes,
        )
    else:
        raise ValueError(f"unsupported direction mode {args.direction_mode}")
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
            np.abs(reference["deterministic"][:, -1] - arrays["preevent_deterministic"]).max()
        ),
        "logits_max_abs_error": float(
            np.abs(reference["logits"][:, -1] - arrays["preevent_logits"]).max()
        ),
        "stochastic_index_mismatch_count": int(
            (
                reference["logits"][:, -1].argmax(-1)
                != arrays["preevent_stochastic_index"]
            ).sum()
        ),
    }
    feature_cache = _cache_frozen_features(
        model,
        arrays,
        reference,
        original_event,
        crop_start,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        batch_size=args.batch_size,
        device=device,
    )
    scale_results = [
        _counterfactual_scale_metrics(
            model,
            feature_cache,
            normalized_target,
            direction,
            plane_weight,
            scale=scale,
            microbatch_size=args.actor_microbatch_size,
            device=device,
        )
        for scale in scales
    ]
    passing_scales = [
        row
        for row in scale_results
        if row["finite"]
        and row["raw_actor_drift"]["rmse"] <= args.maximum_raw_actor_rmse
        and row["raw_actor_drift"]["max_abs"] <= args.maximum_raw_actor_max_abs
        and row["deterministic_action_drift"]["rmse"]
        <= args.maximum_action_rmse
        and row["deterministic_action_drift"]["max_abs"]
        <= args.maximum_action_max_abs
        and row["residual_readout_rmse"] <= args.maximum_residual_readout_rmse
        and row["absolute_correlation_improvement"]
        >= args.minimum_absolute_correlation_improvement
        and row["negative_stochastic_probability_fraction"]
        <= args.maximum_negative_stochastic_fraction
        and row["simplex_sum_max_abs_error"] <= args.maximum_simplex_sum_error
    ]
    selected = passing_scales[-1] if passing_scales else None
    gradient_norms: dict[str, float] = {}
    gradient_loss: dict[str, float] = {}
    if selected is not None:
        grouped = _component_parameters(model)
        for name in COMPONENTS:
            for parameter in grouped[name]:
                parameter.requires_grad_(True)
        audit_batch = np.random.default_rng(args.audit_batch_seed).choice(
            len(original_event), args.batch_size, replace=False
        )
        gradient_norms, gradient_loss = _direct_gradient(
            model,
            arrays,
            reference,
            feature_cache,
            original_event,
            crop_start,
            audit_batch,
            direction,
            exposure,
            pair_exposure,
            scale=float(selected["scale"]),
            sequence_length=args.sequence_length,
            burn_in=args.burn_in,
            target_mean=target_mean,
            target_std=target_std,
            near_plane_m=args.near_plane_m,
            near_plane_multiplier=args.near_plane_multiplier,
            device=device,
        )
    else:
        audit_batch = np.asarray([], dtype=np.int64)
    changed, _forbidden = _changed_model_keys(model_before, model.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
        "pair_support_report": _sha256(args.pair_support_report),
        "reference_batch_audit_report": _sha256(args.reference_batch_audit_report),
        "n696_report": _sha256(args.n696_report),
        "n698_report": _sha256(args.n698_report),
    }
    if args.direction_mode == "deterministic_low_sensitivity":
        geometry_gates = {
            "minimum_singular_value_at_most_maximum": (
                geometry["minimum_singular_value"]
                <= args.maximum_minimum_singular_value
            ),
            "minimum_singular_gap_at_least_minimum": (
                geometry["minimum_singular_gap_ratio"]
                >= args.minimum_singular_gap_ratio
            ),
        }
    else:
        geometry_gates = {
            "actor_null_residual_at_most_maximum": (
                geometry["actor_first_layer_residual_max_abs"]
                <= args.maximum_actor_null_residual
            ),
            "simplex_tangent_residual_at_most_maximum": (
                geometry["simplex_tangent_residual_max_abs"]
                <= args.maximum_simplex_tangent_residual
            ),
            "stochastic_direction_nonzero": geometry["stochastic_direction_norm"] > 0.0,
        }
    channel_gates = {
        **geometry_gates,
        "selected_scale_exists": selected is not None,
        "selected_scale_at_least_minimum": (
            selected is not None and selected["scale"] >= args.minimum_selected_scale
        ),
        "direct_gradients_finite_nonzero": bool(
            gradient_norms
            and all(math.isfinite(value) and value > 0.0 for value in gradient_norms.values())
        ),
    }
    process_gates = {
        "sources_unchanged": source_before == source_after,
        "source_contracts_exact": contracts_exact,
        "n696_rejection_exact": n696_contract,
        "n698_rejection_exact": n698_contract,
        "train_geometry_exact": (
            len(phase_after) == args.expected_train_events
            and len(original_event) == args.expected_train_crops
        ),
        "fixed_reference_boundary_passes": (
            boundary["deterministic_max_abs_error"] <= 0.002
            and boundary["logits_max_abs_error"] <= 0.002
            and boundary["stochastic_index_mismatch_count"] == 0
        ),
        "feature_cache_finite_float32": (
            feature_cache.dtype == torch.float32 and bool(torch.isfinite(feature_cache).all())
        ),
        "direction_mode_contract": (
            geometry["stochastic_direction_max_abs"] == 0.0
            if args.direction_mode == "deterministic_low_sensitivity"
            else geometry["simplex_tangent_residual_max_abs"]
            <= args.maximum_simplex_tangent_residual
        ),
        "direction_is_unit_norm": abs(geometry["direction_norm"] - 1.0) <= 1e-6,
        "model_tensors_exact": not changed,
        "all_scale_metrics_finite": all(row["finite"] for row in scale_results),
        "no_validation_or_test_path": True,
    }
    cuda_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    process_passed = all(process_gates.values()) and resource_gate
    channel_admitted = process_passed and all(channel_gates.values())
    return {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_before == source_after,
        "candidate_scales": list(scales),
        "direction_mode": args.direction_mode,
        "selected_scale": None if selected is None else selected["scale"],
        "geometry": geometry,
        "scale_results": scale_results,
        "direct_gradient_audit_batch": audit_batch.tolist(),
        "direct_gradient_norms": gradient_norms,
        "direct_gradient_loss": gradient_loss,
        "target_mean": target_mean,
        "target_std": target_std,
        "boundary_replay": boundary,
        "feature_cache_shape": list(feature_cache.shape),
        "feature_cache_bytes": feature_cache.numel() * feature_cache.element_size(),
        "train_events": len(phase_after),
        "train_crops": len(original_event),
        "model_tensor_changes": changed,
        "model_optimizer_steps": 0,
        "representation_updates": 0,
        "actor_updates": 0,
        "checkpoint_written": 0,
        "validation_dataset_path_received": 0,
        "validation_dataset_opened": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "flightsim_packets": 0,
        "process_gates": process_gates,
        "process_passed": process_passed,
        "channel_gates": channel_gates,
        "channel_admitted": channel_admitted,
        "resource_gate": resource_gate,
        "device": str(device),
        "cuda_peak_memory_allocated_bytes": cuda_peak,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--pair-support-report", type=Path, required=True)
    parser.add_argument("--reference-batch-audit-report", type=Path, required=True)
    parser.add_argument("--n696-report", type=Path, required=True)
    parser.add_argument("--n698-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-scales", default="0.05,0.1,0.2,0.4")
    parser.add_argument(
        "--direction-mode",
        choices=("deterministic_low_sensitivity", "actor_tangent_null"),
        default="deterministic_low_sensitivity",
    )
    parser.add_argument("--audit-batch-seed", type=int, default=697)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--actor-microbatch-size", type=int, default=4096)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--expected-train-events", type=int, default=272)
    parser.add_argument("--expected-train-crops", type=int, default=1904)
    parser.add_argument("--maximum-minimum-singular-value", type=float, default=0.005)
    parser.add_argument("--minimum-singular-gap-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-raw-actor-rmse", type=float, default=0.001)
    parser.add_argument("--maximum-raw-actor-max-abs", type=float, default=0.01)
    parser.add_argument("--maximum-action-rmse", type=float, default=0.0001)
    parser.add_argument("--maximum-action-max-abs", type=float, default=0.001)
    parser.add_argument("--maximum-residual-readout-rmse", type=float, default=1e-5)
    parser.add_argument(
        "--minimum-absolute-correlation-improvement", type=float, default=0.10
    )
    parser.add_argument("--minimum-selected-scale", type=float, default=0.10)
    parser.add_argument(
        "--maximum-negative-stochastic-fraction", type=float, default=1.0
    )
    parser.add_argument("--maximum-simplex-sum-error", type=float, default=1e-5)
    parser.add_argument("--maximum-actor-null-residual", type=float, default=1e-8)
    parser.add_argument(
        "--maximum-simplex-tangent-residual", type=float, default=1e-8
    )
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("audit refuses to overwrite its report")
    report = run_audit(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["channel_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
