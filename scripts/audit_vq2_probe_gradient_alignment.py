#!/usr/bin/env python3
"""Measure random-probe representation-gradient coupling on N681 train only."""

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

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_partition_support import (
    AUDIT_SCHEMA as SAMPLE_SUPPORT_SCHEMA,
    _crop_exposure_counts,
)
from scripts.audit_vq2_event_pair_support import (
    PAIR_AUDIT_SCHEMA,
    _crop_pair_exposure_counts,
)
from scripts.audit_vq2_success_prefix_sequence_capacity import _crop_index
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _fixed_parent_reference,
    _weighted_delta_loss,
    _weighted_plane_loss,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    SoftPlaneProbe,
    _changed_model_keys,
    _crop_initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _soft_posterior_features,
    _target_batch,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import _finite_clip_grad_norm


AUDIT_SCHEMA = "vq2_probe_representation_gradient_alignment_v1"
COMPONENTS = ("encoder", "sequence", "posterior")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_probe_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("probe seeds must contain distinct nonnegative integers")
    return seeds


def _component_parameters(model) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "encoder": list(model.rssm.encoder.parameters()),
        "sequence": list(model.rssm.sequence.parameters()),
        "posterior": list(model.rssm.posterior.parameters()),
    }


def _target_normalization(
    target_m: np.ndarray,
    crop_start: np.ndarray,
    exposure_by_history_index: np.ndarray,
    *,
    burn_in: int,
) -> tuple[float, float]:
    trained = np.asarray(target_m, dtype=np.float64)[:, burn_in:]
    time_index = np.asarray(crop_start, dtype=np.int64)[:, None] + np.arange(
        burn_in, target_m.shape[1], dtype=np.int64
    )[None, :]
    weight = 1.0 / np.asarray(exposure_by_history_index, dtype=np.float64)[
        time_index
    ]
    total = float(weight.sum())
    mean = float((weight * trained).sum() / total)
    std = float(np.sqrt((weight * np.square(trained - mean)).sum() / total))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
        raise RuntimeError("train-only target normalization is invalid")
    return mean, std


@torch.no_grad()
def _cache_frozen_features(
    model,
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    original_event: np.ndarray,
    crop_start: np.ndarray,
    *,
    sequence_length: int,
    burn_in: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    feature_size = model.rssm.feature_size
    cached = torch.empty(
        (len(original_event), sequence_length - burn_in, feature_size),
        dtype=torch.float32,
        device="cpu",
    )
    model.eval()
    for offset in range(0, len(original_event), batch_size):
        chosen = np.arange(offset, min(offset + batch_size, len(original_event)))
        event = original_event[chosen]
        starts = crop_start[chosen]
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
                arrays,
                reference,
                event,
                event,
                starts,
                device=device,
            ),
        )
        cached[chosen] = _soft_posterior_features(state)[:, burn_in:].cpu()
    return cached


def _probe_loss(
    probe: SoftPlaneProbe,
    feature: torch.Tensor,
    target_m: torch.Tensor,
    crop_start: np.ndarray,
    exposure_by_history_index: np.ndarray,
    pair_exposure_by_endpoint: np.ndarray,
    *,
    burn_in: int,
    target_mean: float,
    target_std: float,
    near_plane_m: float,
    near_plane_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = probe(feature)
    normalized = (target_m - target_mean) / target_std
    plane, plane_weight = _weighted_plane_loss(
        prediction,
        normalized,
        target_m,
        crop_start,
        exposure_by_history_index,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    delta, delta_weight = _weighted_delta_loss(
        prediction,
        normalized,
        target_m,
        crop_start,
        pair_exposure_by_endpoint,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    return plane + delta, {
        "plane": float(plane.detach().cpu()),
        "delta": float(delta.detach().cpu()),
        "plane_weight_minimum": float(plane_weight.min().detach().cpu()),
        "plane_weight_maximum": float(plane_weight.max().detach().cpu()),
        "delta_weight_minimum": float(delta_weight.min().detach().cpu()),
        "delta_weight_maximum": float(delta_weight.max().detach().cpu()),
    }


def _representation_gradient(
    model,
    probe: SoftPlaneProbe,
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    original_event: np.ndarray,
    crop_start: np.ndarray,
    chosen: np.ndarray,
    exposure_by_history_index: np.ndarray,
    pair_exposure_by_endpoint: np.ndarray,
    *,
    sequence_length: int,
    burn_in: int,
    target_mean: float,
    target_std: float,
    near_plane_m: float,
    near_plane_multiplier: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    event = original_event[chosen]
    starts = crop_start[chosen]
    model.train()
    probe.train()
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
            arrays,
            reference,
            event,
            event,
            starts,
            device=device,
        ),
    )
    target_np = _target_batch(
        arrays, event, starts, sequence_length=sequence_length
    )[:, burn_in:]
    target = torch.from_numpy(target_np).to(device)
    loss, loss_metrics = _probe_loss(
        probe,
        _soft_posterior_features(state)[:, burn_in:],
        target,
        starts,
        exposure_by_history_index,
        pair_exposure_by_endpoint,
        burn_in=burn_in,
        target_mean=target_mean,
        target_std=target_std,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    grouped = _component_parameters(model)
    flat_parameters = [parameter for name in COMPONENTS for parameter in grouped[name]]
    gradients = torch.autograd.grad(loss, flat_parameters, allow_unused=False)
    result: dict[str, torch.Tensor] = {}
    cursor = 0
    for name in COMPONENTS:
        count = len(grouped[name])
        result[name] = torch.cat(
            [gradient.detach().float().flatten().cpu() for gradient in gradients[cursor : cursor + count]]
        )
        cursor += count
    loss_metrics["total"] = float(loss.detach().cpu())
    return result, loss_metrics


def _vector_group_metrics(
    rows: list[dict[str, torch.Tensor]],
    components: tuple[str, ...],
    *,
    top_coordinates: int,
) -> dict[str, object]:
    norms = [
        math.sqrt(
            sum(float(torch.dot(row[name], row[name])) for name in components)
        )
        for row in rows
    ]
    pairwise_cosine: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            dot = sum(
                float(torch.dot(rows[left][name], rows[right][name]))
                for name in components
            )
            pairwise_cosine.append(dot / (norms[left] * norms[right]))
    mean_gradient = {
        name: sum((row[name] for row in rows), torch.zeros_like(rows[0][name]))
        / len(rows)
        for name in components
    }
    ensemble_norm = math.sqrt(
        sum(float(torch.dot(value, value)) for value in mean_gradient.values())
    )
    cosine_to_ensemble = [
        sum(float(torch.dot(row[name], mean_gradient[name])) for name in components)
        / (norm * ensemble_norm)
        for row, norm in zip(rows, norms, strict=True)
    ]
    mean_absolute = torch.cat(
        [
            sum((row[name].abs() for row in rows), torch.zeros_like(rows[0][name]))
            / len(rows)
            for name in components
        ]
    )
    selected_count = min(top_coordinates, len(mean_absolute))
    top_index = torch.topk(mean_absolute, selected_count, sorted=False).indices
    selected_signs: list[torch.Tensor] = []
    for row in rows:
        joined = torch.cat([row[name] for name in components])
        selected_signs.append(torch.sign(joined[top_index]))
    pairwise_sign: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            pairwise_sign.append(
                float((selected_signs[left] == selected_signs[right]).float().mean())
            )
    norm_array = np.asarray(norms, dtype=np.float64)
    return {
        "parameters": int(sum(rows[0][name].numel() for name in components)),
        "norms": norms,
        "norm_mean": float(norm_array.mean()),
        "norm_minimum": float(norm_array.min()),
        "norm_maximum": float(norm_array.max()),
        "norm_coefficient_of_variation": float(norm_array.std() / norm_array.mean()),
        "pairwise_cosine_minimum": float(min(pairwise_cosine)),
        "pairwise_cosine_mean": float(np.mean(pairwise_cosine)),
        "pairwise_cosine_median": float(np.median(pairwise_cosine)),
        "pairwise_sign_agreement_mean": float(np.mean(pairwise_sign)),
        "pairwise_sign_agreement_median": float(np.median(pairwise_sign)),
        "top_coordinates": selected_count,
        "cosine_to_ensemble": cosine_to_ensemble,
        "positive_cosine_to_ensemble": int(
            sum(value > 0.0 for value in cosine_to_ensemble)
        ),
        "all_finite": bool(
            np.isfinite(norm_array).all()
            and np.isfinite(pairwise_cosine).all()
            and np.isfinite(pairwise_sign).all()
            and np.isfinite(cosine_to_ensemble).all()
        ),
    }


def _gradient_alignment_summary(
    rows: list[dict[str, torch.Tensor]], *, top_coordinates: int
) -> dict[str, dict[str, object]]:
    if len(rows) < 2 or any(set(row) != set(COMPONENTS) for row in rows):
        raise ValueError("gradient rows must provide every representation component")
    result = {
        name: _vector_group_metrics(rows, (name,), top_coordinates=top_coordinates)
        for name in COMPONENTS
    }
    result["overall"] = _vector_group_metrics(
        rows, COMPONENTS, top_coordinates=top_coordinates
    )
    return result


def _source_contracts(args: argparse.Namespace, hashes: dict[str, str]) -> bool:
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    support = json.loads(args.support_report.read_text(encoding="utf-8"))
    pair = json.loads(args.pair_support_report.read_text(encoding="utf-8"))
    reference = json.loads(
        args.reference_batch_audit_report.read_text(encoding="utf-8")
    )
    return bool(
        materialization.get("contract") == MATERIALIZATION_SCHEMA
        and materialization.get("process_passed")
        and materialization.get("output_sha256", {}).get("train")
        == hashes["train_dataset"]
        and materialization.get("test_rows_scored") == 0
        and support.get("contract") == SAMPLE_SUPPORT_SCHEMA
        and support.get("process_passed")
        and support.get("source_hashes", {}).get("train")
        == hashes["train_dataset"]
        and support.get("source_hashes", {}).get("materialization_report")
        == hashes["materialization_report"]
        and pair.get("contract") == PAIR_AUDIT_SCHEMA
        and pair.get("process_passed")
        and pair.get("source_hashes", {}).get("train")
        == hashes["train_dataset"]
        and pair.get("source_hashes", {}).get("materialization_report")
        == hashes["materialization_report"]
        and pair.get("source_hashes", {}).get("sample_support_report")
        == hashes["support_report"]
        and reference.get("contract")
        == "vq2_partition_reference_batch_sensitivity_v1"
        and reference.get("process_passed")
        and reference.get("source_hashes", {}).get("checkpoint")
        == hashes["checkpoint"]
        and reference.get("source_hashes", {}).get("train")
        == hashes["train_dataset"]
        and reference.get("source_hashes", {}).get("support_report")
        == hashes["support_report"]
        and reference.get("test_dataset_opened") == 0
    )


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    seeds = _parse_probe_seeds(args.probe_seeds)
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
    }
    contracts_exact = _source_contracts(args, source_before)
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
    if len(original_event) % args.batch_size:
        raise RuntimeError("train crop count is not a whole number of batches")
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
        arrays,
        original_event,
        crop_start,
        sequence_length=args.sequence_length,
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
            np.abs(reference["logits"][:, -1] - arrays["preevent_logits"]).max()
        ),
        "stochastic_index_mismatch_count": int(
            (
                reference["logits"][:, -1].argmax(-1)
                != arrays["preevent_stochastic_index"]
            ).sum()
        ),
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    for name in COMPONENTS:
        for parameter in grouped[name]:
            parameter.requires_grad_(True)
    order = np.random.default_rng(args.order_seed).permutation(len(original_event))
    audit_batch = order[: args.batch_size]
    epoch_batches = [
        order[offset : offset + args.batch_size]
        for offset in range(0, len(order), args.batch_size)
    ]
    probes: list[SoftPlaneProbe] = []
    initial_rows: list[dict[str, torch.Tensor]] = []
    initial_losses: list[dict[str, float]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        probe = SoftPlaneProbe(model.rssm.feature_size, args.probe_hidden_size).to(
            device
        )
        probes.append(probe)
        gradients, losses = _representation_gradient(
            model,
            probe,
            arrays,
            reference,
            original_event,
            crop_start,
            audit_batch,
            exposure,
            pair_exposure,
            sequence_length=args.sequence_length,
            burn_in=args.burn_in,
            target_mean=target_mean,
            target_std=target_std,
            near_plane_m=args.near_plane_m,
            near_plane_multiplier=args.near_plane_multiplier,
            device=device,
        )
        initial_rows.append(gradients)
        initial_losses.append(losses)
    initial_alignment = _gradient_alignment_summary(
        initial_rows, top_coordinates=args.top_coordinates
    )
    del initial_rows
    warmup_rows: list[dict[str, torch.Tensor]] = []
    warmup_losses: list[dict[str, float]] = []
    warmup_max_gradient: list[float] = []
    warmup_exposure_minimum: list[int] = []
    warmup_exposure_maximum: list[int] = []
    target_postburn = target_all[:, args.burn_in :]
    for probe in probes:
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=args.probe_learning_rate,
            weight_decay=args.probe_weight_decay,
        )
        counts = np.zeros(len(original_event), dtype=np.int64)
        maximum_gradient = 0.0
        final_loss: dict[str, float] = {}
        probe.train()
        for chosen in epoch_batches:
            np.add.at(counts, chosen, 1)
            feature = feature_cache[chosen].to(device)
            target = torch.from_numpy(target_postburn[chosen]).to(device)
            loss, final_loss = _probe_loss(
                probe,
                feature,
                target,
                crop_start[chosen],
                exposure,
                pair_exposure,
                burn_in=args.burn_in,
                target_mean=target_mean,
                target_std=target_std,
                near_plane_m=args.near_plane_m,
                near_plane_multiplier=args.near_plane_multiplier,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = _finite_clip_grad_norm(
                list(probe.parameters()), args.probe_gradient_clip, label="probe warmup"
            )
            maximum_gradient = max(maximum_gradient, float(gradient_norm.cpu()))
            optimizer.step()
        warmup_exposure_minimum.append(int(counts.min()))
        warmup_exposure_maximum.append(int(counts.max()))
        warmup_max_gradient.append(maximum_gradient)
        final_loss["total"] = float(loss.detach().cpu())
        warmup_losses.append(final_loss)
        gradients, _post_loss = _representation_gradient(
            model,
            probe,
            arrays,
            reference,
            original_event,
            crop_start,
            audit_batch,
            exposure,
            pair_exposure,
            sequence_length=args.sequence_length,
            burn_in=args.burn_in,
            target_mean=target_mean,
            target_std=target_std,
            near_plane_m=args.near_plane_m,
            near_plane_multiplier=args.near_plane_multiplier,
            device=device,
        )
        warmup_rows.append(gradients)
    warmup_alignment = _gradient_alignment_summary(
        warmup_rows, top_coordinates=args.top_coordinates
    )
    del warmup_rows
    changed, _forbidden = _changed_model_keys(model_before, model.state_dict())
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "train_dataset": _sha256(args.train_dataset),
        "materialization_report": _sha256(args.materialization_report),
        "support_report": _sha256(args.support_report),
        "pair_support_report": _sha256(args.pair_support_report),
        "reference_batch_audit_report": _sha256(args.reference_batch_audit_report),
    }
    initial_overall = initial_alignment["overall"]
    warmup_overall = warmup_alignment["overall"]
    component_cosine_improved = {
        name: (
            warmup_alignment[name]["pairwise_cosine_median"]
            > initial_alignment[name]["pairwise_cosine_median"]
        )
        for name in COMPONENTS
    }
    alignment_gates = {
        "post_median_cosine_at_least_minimum": (
            warmup_overall["pairwise_cosine_median"]
            >= args.minimum_post_median_cosine
        ),
        "median_cosine_improvement_at_least_minimum": (
            warmup_overall["pairwise_cosine_median"]
            - initial_overall["pairwise_cosine_median"]
            >= args.minimum_median_cosine_improvement
        ),
        "post_sign_agreement_at_least_minimum": (
            warmup_overall["pairwise_sign_agreement_median"]
            >= args.minimum_post_sign_agreement
        ),
        "sign_agreement_improvement_at_least_minimum": (
            warmup_overall["pairwise_sign_agreement_median"]
            - initial_overall["pairwise_sign_agreement_median"]
            >= args.minimum_sign_agreement_improvement
        ),
        "every_component_cosine_improves": all(component_cosine_improved.values()),
        "post_positive_ensemble_count_at_least_minimum": (
            warmup_overall["positive_cosine_to_ensemble"]
            >= args.minimum_positive_ensemble_count
        ),
        "post_norm_cv_at_most_maximum": (
            warmup_overall["norm_coefficient_of_variation"]
            <= args.maximum_post_norm_cv
        ),
    }
    process_gates = {
        "sources_unchanged": source_before == source_after,
        "source_contracts_exact": contracts_exact,
        "train_event_count_exact": len(phase_after) == args.expected_train_events,
        "train_crop_count_exact": len(original_event) == args.expected_train_crops,
        "fixed_reference_boundary_passes": (
            boundary["deterministic_max_abs_error"] <= 0.002
            and boundary["logits_max_abs_error"] <= 0.002
            and boundary["stochastic_index_mismatch_count"] == 0
        ),
        "feature_cache_finite": bool(torch.isfinite(feature_cache).all()),
        "feature_cache_float32": feature_cache.dtype == torch.float32,
        "model_tensors_exact": not changed,
        "initial_gradients_finite": all(
            initial_alignment[name]["all_finite"]
            and initial_alignment[name]["norm_minimum"] > 0.0
            for name in (*COMPONENTS, "overall")
        ),
        "warmup_gradients_finite": all(
            warmup_alignment[name]["all_finite"]
            and warmup_alignment[name]["norm_minimum"] > 0.0
            for name in (*COMPONENTS, "overall")
        ),
        "one_exact_probe_epoch_each": (
            warmup_exposure_minimum == [1] * len(seeds)
            and warmup_exposure_maximum == [1] * len(seeds)
        ),
        "warmup_gradients_nonzero_finite": all(
            math.isfinite(value) and value > 0.0 for value in warmup_max_gradient
        ),
        "no_validation_or_test_path": True,
    }
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    process_passed = all(process_gates.values()) and resource_gate
    alignment_admitted = process_passed and all(alignment_gates.values())
    return {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_before == source_after,
        "probe_seeds": list(seeds),
        "order_seed": args.order_seed,
        "train_events": len(phase_after),
        "train_crops": len(original_event),
        "crop_starts": list(crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "batch_size": args.batch_size,
        "warmup_batches_per_probe": len(epoch_batches),
        "probe_optimizer_steps": len(epoch_batches) * len(seeds),
        "model_optimizer_steps": 0,
        "representation_updates": 0,
        "actor_updates": 0,
        "reward_updates": 0,
        "critic_updates": 0,
        "target_mean": target_mean,
        "target_std": target_std,
        "boundary_replay": boundary,
        "feature_cache_shape": list(feature_cache.shape),
        "feature_cache_bytes": feature_cache.numel() * feature_cache.element_size(),
        "audit_batch_crop_indices": audit_batch.tolist(),
        "initial_losses": initial_losses,
        "warmup_final_losses": warmup_losses,
        "warmup_maximum_preclip_gradient_norms": warmup_max_gradient,
        "warmup_crop_count_minimum": warmup_exposure_minimum,
        "warmup_crop_count_maximum": warmup_exposure_maximum,
        "initial_alignment": initial_alignment,
        "warmup_alignment": warmup_alignment,
        "component_cosine_improved": component_cosine_improved,
        "alignment_gates": alignment_gates,
        "alignment_admitted": alignment_admitted,
        "model_tensor_changes": changed,
        "discarded_probe_count": len(probes),
        "checkpoint_written": 0,
        "validation_dataset_path_received": 0,
        "validation_dataset_opened": 0,
        "test_dataset_path_received": 0,
        "test_dataset_opened": 0,
        "native_environment_created": 0,
        "flightsim_packets": 0,
        "process_gates": process_gates,
        "process_passed": process_passed,
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-seeds", default="6960,6961,6962,6963,6964,6965,6966,6967")
    parser.add_argument("--order-seed", type=int, default=696)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=416)
    parser.add_argument("--probe-hidden-size", type=int, default=128)
    parser.add_argument("--probe-learning-rate", type=float, default=0.0003)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0001)
    parser.add_argument("--probe-gradient-clip", type=float, default=10.0)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--near-plane-multiplier", type=float, default=2.0)
    parser.add_argument("--top-coordinates", type=int, default=10000)
    parser.add_argument("--expected-train-events", type=int, default=272)
    parser.add_argument("--expected-train-crops", type=int, default=1904)
    parser.add_argument("--minimum-post-median-cosine", type=float, default=0.20)
    parser.add_argument(
        "--minimum-median-cosine-improvement", type=float, default=0.15
    )
    parser.add_argument("--minimum-post-sign-agreement", type=float, default=0.55)
    parser.add_argument(
        "--minimum-sign-agreement-improvement", type=float, default=0.03
    )
    parser.add_argument("--minimum-positive-ensemble-count", type=int, default=7)
    parser.add_argument("--maximum-post-norm-cv", type=float, default=1.0)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("audit refuses to overwrite its report")
    for name in (
        "sequence_length",
        "burn_in",
        "batch_size",
        "fixed_reference_batch_size",
        "probe_hidden_size",
        "top_coordinates",
        "expected_train_events",
        "expected_train_crops",
        "minimum_positive_ensemble_count",
        "maximum_cuda_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    if args.near_plane_multiplier <= 0.0:
        parser.error("near-plane multiplier must be positive")
    report = run_audit(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["alignment_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
