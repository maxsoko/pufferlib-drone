#!/usr/bin/env python3
"""Joint successful-prefix probe/RSSM fitting with a frozen N587 actor."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_success_prefix_sequence_capacity import (
    _crop_index,
    _metrics,
)
from scripts.audit_vq2_event_partition_support import (
    AUDIT_SCHEMA as SUPPORT_AUDIT_SCHEMA,
    _crop_exposure_counts,
)
from scripts.audit_vq2_event_pair_support import (
    PAIR_AUDIT_SCHEMA,
    _crop_pair_exposure_counts,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    CONTINUATION_SCHEMA,
    SoftPlaneProbe,
    _changed_model_keys,
    _crop_full,
    _crop_initial_state,
    _drift,
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _parent_reference,
    _replay_geometry,
    _soft_posterior_features,
    _target_batch,
)
from scripts.materialize_vq2_event_window_partitions import MATERIALIZATION_SCHEMA
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


JOINT_SCHEMA = "vq2_success_prefix_joint_representation_v1"
TANGENT_SOURCE_CONTRACT_SCHEMA = "vq2_tangent_channel_source_contract_audit_v1"
N700_REPORT_SHA256 = (
    "8cd15de4eb04ccfcd1cc7e09a5e4bccbecd5af5811c3ed561265cbf878b4a8b8"
)
INVARIANT_SOURCE_CONTRACT_SCHEMA = "vq2_rotation_invariant_progress_geometry_audit_v1"
N704_REPORT_SHA256 = (
    "73e701069af3d25c9bc6578602871fda787f9523be734f09fe46386251aed733"
)
FROZEN_READOUT_CONTRACT_SCHEMA = "vq2_frozen_train_ridge_readout_audit_v1"
N706_REPORT_SHA256 = (
    "b5f9a26dda344486595fdd85f0f4a80242e887f36d362d6c92a9bc52a303e3de"
)


class _EpochBatchSampler:
    """Shuffle once per epoch while exposing every item before reuse."""

    def __init__(
        self,
        values: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        self.values = np.asarray(values, dtype=np.int64)
        if (
            self.values.ndim != 1
            or len(self.values) == 0
            or batch_size <= 0
            or len(self.values) % batch_size
        ):
            raise ValueError("balanced epoch sampler requires whole minibatches")
        self.batch_size = batch_size
        self.rng = rng
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0

    def draw(self) -> np.ndarray:
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(self.values)
            self.cursor = 0
        result = self.order[self.cursor : self.cursor + self.batch_size]
        if len(result) != self.batch_size:
            raise RuntimeError("balanced epoch sampler crossed an epoch boundary")
        self.cursor += self.batch_size
        return result


def _load_partitioned_arrays(
    train_dataset: Path, validation_dataset: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(train_dataset) as loaded:
        train = {name: loaded[name].copy() for name in loaded.files}
    with np.load(validation_dataset) as loaded:
        validation = {name: loaded[name].copy() for name in loaded.files}
    if set(train) != set(validation) or not train:
        raise RuntimeError("train and validation corpus arrays differ")
    for name in train:
        if (
            train[name].ndim == 0
            or validation[name].ndim == 0
            or train[name].shape[1:] != validation[name].shape[1:]
            or train[name].dtype != validation[name].dtype
        ):
            raise RuntimeError(f"train/validation array {name} is incompatible")
    arrays = {
        name: np.concatenate((train[name], validation[name]), axis=0)
        for name in train
    }
    event_split = np.concatenate(
        (
            np.zeros(len(next(iter(train.values()))), dtype=np.uint8),
            np.ones(len(next(iter(validation.values()))), dtype=np.uint8),
        )
    )
    return arrays, event_split


def _fixed_reference_event_chunks(
    event_ids: np.ndarray, batch_size: int
) -> list[tuple[np.ndarray, int]]:
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if event_ids.ndim != 1 or len(event_ids) == 0 or batch_size <= 0:
        raise ValueError("event ids and fixed reference batch are invalid")
    chunks: list[tuple[np.ndarray, int]] = []
    for offset in range(0, len(event_ids), batch_size):
        chosen = event_ids[offset : offset + batch_size]
        valid = len(chosen)
        if valid < batch_size:
            chosen = np.concatenate(
                (chosen, np.resize(event_ids, batch_size - valid))
            )
        chunks.append((chosen, valid))
    return chunks


def _fixed_parent_reference(
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
    for chosen, valid in _fixed_reference_event_chunks(event_ids, batch_size):
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


def _weighted_plane_loss(
    prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    target_m: torch.Tensor,
    crop_starts: np.ndarray,
    exposure_by_history_index: np.ndarray,
    *,
    burn_in: int,
    near_plane_m: float,
    near_plane_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != normalized_target.shape or prediction.shape != target_m.shape:
        raise ValueError("plane prediction and targets must align")
    time_index = crop_starts[:, None] + np.arange(
        burn_in, burn_in + prediction.shape[1], dtype=np.int64
    )[None, :]
    exposure = np.asarray(exposure_by_history_index, dtype=np.float64)[time_index]
    if np.any(exposure <= 0):
        raise RuntimeError("a trained timestep has zero crop exposure")
    weight = torch.from_numpy((1.0 / exposure).astype(np.float32)).to(
        prediction.device
    )
    if near_plane_multiplier != 1.0:
        weight = weight * torch.where(
            target_m.abs() <= near_plane_m,
            torch.as_tensor(near_plane_multiplier, device=prediction.device),
            torch.ones((), device=prediction.device),
        )
    loss = (weight * torch.square(prediction - normalized_target)).sum() / weight.sum()
    return loss, weight


def _weighted_delta_loss(
    prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    target_m: torch.Tensor,
    crop_starts: np.ndarray,
    pair_exposure_by_endpoint: np.ndarray,
    *,
    burn_in: int,
    near_plane_m: float,
    near_plane_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != normalized_target.shape or prediction.shape != target_m.shape:
        raise ValueError("delta prediction and targets must align")
    if prediction.shape[1] < 2:
        raise ValueError("delta loss requires at least two prediction steps")
    endpoint_index = crop_starts[:, None] + np.arange(
        burn_in + 1, burn_in + prediction.shape[1], dtype=np.int64
    )[None, :]
    exposure = np.asarray(pair_exposure_by_endpoint, dtype=np.float64)[
        endpoint_index
    ]
    if np.any(exposure <= 0):
        raise RuntimeError("a trained temporal pair has zero crop exposure")
    weight = torch.from_numpy((1.0 / exposure).astype(np.float32)).to(
        prediction.device
    )
    if near_plane_multiplier != 1.0:
        near_pair = (target_m[:, :-1].abs() <= near_plane_m) | (
            target_m[:, 1:].abs() <= near_plane_m
        )
        weight = weight * torch.where(
            near_pair,
            torch.as_tensor(near_plane_multiplier, device=prediction.device),
            torch.ones((), device=prediction.device),
        )
    prediction_delta = torch.diff(prediction, dim=1)
    target_delta = torch.diff(normalized_target, dim=1)
    loss = (weight * torch.square(prediction_delta - target_delta)).sum() / weight.sum()
    return loss, weight


def _canonical_actor_tangent_null_direction(
    actor_first_weight: torch.Tensor,
    deterministic_size: int,
    stochastic_groups: int,
    stochastic_classes: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Return the deterministic canonical actor-null/simplex-tangent direction."""
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
        "projected_basis_norm": float(
            projected_norm_square[canonical_index].sqrt()
        ),
        "actor_first_layer_residual_norm": float(
            torch.linalg.vector_norm(actor_residual)
        ),
        "actor_first_layer_residual_max_abs": float(actor_residual.abs().max()),
        "simplex_tangent_residual_max_abs": float(simplex_residual.abs().max()),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "deterministic_direction_norm": float(
            torch.linalg.vector_norm(direction[:deterministic_size])
        ),
        "stochastic_direction_norm": float(
            torch.linalg.vector_norm(stochastic)
        ),
        "stochastic_direction_max_abs": float(stochastic.abs().max()),
    }


def _tangent_residual_losses(
    child_feature: torch.Tensor,
    parent_deterministic: torch.Tensor,
    parent_logits: torch.Tensor,
    direction: torch.Tensor,
    normalized_target: torch.Tensor,
    target_m: torch.Tensor,
    crop_starts: np.ndarray,
    exposure_by_history_index: np.ndarray,
    pair_exposure_by_endpoint: np.ndarray,
    *,
    scale: float,
    burn_in: int,
    near_plane_m: float,
    near_plane_multiplier: float,
    delta_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale <= 0.0 or delta_weight < 0.0:
        raise ValueError("tangent scale and delta weight are invalid")
    parent_probability = parent_logits.float().softmax(-1)
    parent_feature = torch.cat(
        (parent_deterministic.float(), parent_probability.flatten(-2)), -1
    )
    if child_feature.shape != parent_feature.shape:
        raise ValueError("child and frozen-parent tangent features must align")
    prediction = (
        (child_feature.float() - parent_feature)
        @ direction.to(child_feature.device)
    )[:, burn_in:] / scale
    plane, _plane_weight = _weighted_plane_loss(
        prediction,
        normalized_target,
        target_m,
        crop_starts,
        exposure_by_history_index,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    delta, _delta_weight = _weighted_delta_loss(
        prediction,
        normalized_target,
        target_m,
        crop_starts,
        pair_exposure_by_endpoint,
        burn_in=burn_in,
        near_plane_m=near_plane_m,
        near_plane_multiplier=near_plane_multiplier,
    )
    return plane + delta_weight * delta, plane, delta


def _tangent_source_contract_exact(path: Path | None) -> bool:
    if path is None or _sha256(path) != N700_REPORT_SHA256:
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        report.get("contract") == TANGENT_SOURCE_CONTRACT_SCHEMA
        and report.get("corrected_process_passed") is True
        and report.get("corrected_channel_admitted") is True
        and report.get("implementation_smoke_authorized") is True
        and report.get("correction_scope") == "legacy_n698_report_field_only"
        and all(report.get("audit_gates", {}).values())
        and report.get("model_loaded") == 0
        and report.get("model_tensor_changes") == []
        and report.get("optimizer_steps") == 0
        and report.get("checkpoint_written") == 0
        and report.get("test_dataset_path_received") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


def _weighted_covariance_score(
    feature: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    if feature.shape[:-1] != target.shape or target.shape != weight.shape:
        raise ValueError("covariance feature, target, and weight shapes must align")
    if feature.shape[-1] <= 0 or bool((weight <= 0.0).any()):
        raise ValueError("covariance features and weights are invalid")
    flat_feature = feature.float().reshape(-1, feature.shape[-1])
    flat_target = target.float().reshape(-1)
    flat_weight = weight.float().reshape(-1)
    normalized_weight = flat_weight / flat_weight.sum()
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
    score = cross.square().sum() / (
        feature_variance * target_variance + epsilon
    )
    return score, {
        "score": float(score.detach().cpu()),
        "feature_trace_variance": float(feature_variance.detach().cpu()),
        "target_variance": float(target_variance.detach().cpu()),
        "cross_covariance_norm": float(
            torch.linalg.vector_norm(cross).detach().cpu()
        ),
    }


def _invariant_progress_loss(
    feature: torch.Tensor,
    normalized_target: torch.Tensor,
    level_weight: torch.Tensor,
    pair_weight: torch.Tensor,
    *,
    level_coefficient: float,
    delta_coefficient: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    level_score, level_metrics = _weighted_covariance_score(
        feature, normalized_target, level_weight
    )
    delta_score, delta_metrics = _weighted_covariance_score(
        torch.diff(feature, dim=1),
        torch.diff(normalized_target, dim=1),
        pair_weight,
    )
    loss = (
        level_coefficient * (1.0 - level_score)
        + delta_coefficient * (1.0 - delta_score)
    )
    return loss, {"level": level_metrics, "delta": delta_metrics}


def _invariant_source_contract_exact(path: Path | None) -> bool:
    if path is None or _sha256(path) != N704_REPORT_SHA256:
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        report.get("contract") == INVARIANT_SOURCE_CONTRACT_SCHEMA
        and report.get("process_passed") is True
        and report.get("mechanism_admitted") is True
        and all(report.get("mechanism_gates", {}).values())
        and report.get("model_tensor_changes") == []
        and report.get("optimizer_steps") == 0
        and report.get("checkpoint_written") == 0
        and report.get("validation_dataset_path_received") == 0
        and report.get("test_dataset_path_received") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


def _frozen_readout_contract_exact(path: Path | None) -> bool:
    if path is None or _sha256(path) != N706_REPORT_SHA256:
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        report.get("contract") == FROZEN_READOUT_CONTRACT_SCHEMA
        and report.get("process_passed") is True
        and report.get("readout_smoke_passed") is True
        and report.get("ridge_candidates") == [0.001]
        and report.get("validation_used_for_fit_or_selection") == 0
        and report.get("model_tensor_changes") == []
        and report.get("optimizer_steps") == 0
        and report.get("checkpoint_written") == 0
        and report.get("test_dataset_path_received") == 0
        and report.get("test_dataset_opened") == 0
        and report.get("native_environment_created") == 0
        and report.get("flightsim_packets") == 0
    )


def _soft_actor_action(model, state) -> torch.Tensor:
    return model.deterministic_actor_action(
        model.actor_distribution(_soft_posterior_features(state))
    )


def _parent_soft_action(model, reference: dict[str, np.ndarray]) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(reference["deterministic"]), 4):
            deterministic = torch.from_numpy(
                reference["deterministic"][offset : offset + 4]
            ).to(next(model.parameters()).device)
            logits = torch.from_numpy(reference["logits"][offset : offset + 4]).to(
                deterministic.device
            )
            probability = logits.float().softmax(-1)
            feature = torch.cat(
                (deterministic.float(), probability.flatten(-2)), -1
            )
            action = model.deterministic_actor_action(
                model.actor_distribution(feature)
            )
            rows.append(action.float().cpu().numpy())
    return np.concatenate(rows)


def _sample_dense_pool(
    model,
    replay: QuantizedSequenceReplay,
    *,
    pool_size: int,
    sequence_length: int,
    seed: int,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observation, action, _reward, _continuation = replay.sample(
        pool_size,
        sequence_length,
        device=torch.device("cpu"),
        rng=random.Random(seed),
    )
    legal = observation[..., :LEGAL_OBS_SIZE].contiguous()
    parent_det: list[torch.Tensor] = []
    parent_logits: list[torch.Tensor] = []
    parent_hard_action: list[torch.Tensor] = []
    parent_soft_action: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, pool_size, microbatch_size):
            chosen_legal = legal[offset : offset + microbatch_size].to(device)
            chosen_action = action[offset : offset + microbatch_size].to(device)
            state = _observe_legal_sequence(
                model,
                chosen_legal,
                chosen_action,
                model.rssm.initial(len(chosen_legal), device=device),
            )
            parent_det.append(state.deterministic.float().cpu())
            parent_logits.append(state.logits.float().cpu())
            parent_hard_action.append(
                model.deterministic_actor_action(
                    model.actor_distribution(state.features)
                ).float().cpu()
            )
            parent_soft_action.append(_soft_actor_action(model, state).float().cpu())
    return {
        "legal": legal,
        "input_action": action.float().contiguous(),
        "deterministic": torch.cat(parent_det),
        "logits": torch.cat(parent_logits),
        "hard_action": torch.cat(parent_hard_action),
        "soft_action": torch.cat(parent_soft_action),
    }


def _dense_outputs(
    model,
    pool: dict[str, torch.Tensor],
    *,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, list[torch.Tensor]] = {
        "deterministic": [],
        "logits": [],
        "hard_action": [],
        "soft_action": [],
    }
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(pool["legal"]), microbatch_size):
            legal = pool["legal"][offset : offset + microbatch_size].to(device)
            action = pool["input_action"][offset : offset + microbatch_size].to(
                device
            )
            state = _observe_legal_sequence(
                model,
                legal,
                action,
                model.rssm.initial(len(legal), device=device),
            )
            result["deterministic"].append(state.deterministic.float().cpu())
            result["logits"].append(state.logits.float().cpu())
            result["hard_action"].append(
                model.deterministic_actor_action(
                    model.actor_distribution(state.features)
                ).float().cpu()
            )
            result["soft_action"].append(_soft_actor_action(model, state).float().cpu())
    return {name: torch.cat(rows) for name, rows in result.items()}


def _continuous_validation(
    model,
    probe: SoftPlaneProbe,
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    validation_events: np.ndarray,
    event_to_reference: np.ndarray,
    crop_starts: tuple[int, ...],
    evaluation_indices: np.ndarray,
    *,
    sequence_length: int,
    near_plane_m: float,
    minimum_pair_delta_m: float,
    target_mean: float,
    target_std: float,
    microbatch_size: int,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, float]]:
    prediction_rows: list[np.ndarray] = []
    action_rows: list[torch.Tensor] = []
    model.eval()
    probe.eval()
    full_length = arrays["mask"].shape[1] - 1
    with torch.no_grad():
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
                model,
                legal,
                input_action,
                _initial_state(arrays, chosen, device=device),
            )
            prediction_rows.append(
                (
                    probe(_soft_posterior_features(state)).float() * target_std
                    + target_mean
                ).cpu().numpy()
            )
            action_rows.append(
                model.deterministic_actor_action(
                    model.actor_distribution(state.features)
                ).float().cpu()
            )
    full_prediction = np.concatenate(prediction_rows)
    child_action = torch.cat(action_rows)
    local_event, starts = _crop_index(len(validation_events), crop_starts)
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = starts[:, None] + offsets[None, :]
    crop_prediction = full_prediction[local_event[:, None], time_index]
    original_event = validation_events[local_event]
    target = _target_batch(
        arrays,
        original_event,
        starts,
        sequence_length=sequence_length,
    )
    metrics = _metrics(
        target,
        crop_prediction,
        evaluation_indices=evaluation_indices,
        near_plane_m=near_plane_m,
        minimum_pair_delta_m=minimum_pair_delta_m,
    )
    parent_action = torch.from_numpy(
        reference["action"][event_to_reference[validation_events]]
    )
    return metrics, _drift(child_action, parent_action)


def _preservation(
    validation_action: dict[str, float],
    dense: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, bool]:
    return {
        "validation_action_rmse_bounded": (
            validation_action["rmse"] <= args.maximum_action_rmse
        ),
        "validation_action_max_bounded": (
            validation_action["max_abs"] <= args.maximum_action_max_abs
        ),
        "dense_deterministic_rmse_bounded": (
            dense["deterministic"]["rmse"] <= args.maximum_dense_feature_rmse
        ),
        "dense_logits_rmse_bounded": (
            dense["logits"]["rmse"] <= args.maximum_dense_feature_rmse
        ),
        "dense_action_rmse_bounded": (
            dense["hard_action"]["rmse"] <= args.maximum_action_rmse
        ),
        "dense_action_max_bounded": (
            dense["hard_action"]["max_abs"] <= args.maximum_action_max_abs
        ),
    }


def _candidate_key(
    candidate: dict[str, object], args: argparse.Namespace | None = None
) -> tuple:
    metrics = candidate["validation_progress"]
    preservation = candidate["preservation_gates"]
    assert isinstance(metrics, dict) and isinstance(preservation, dict)
    progress_pass = True
    if args is not None:
        progress_pass = bool(
            float(metrics["correlation"])
            >= args.minimum_validation_correlation
            and float(metrics["mae_m"]) <= args.maximum_validation_mae_m
            and float(metrics["near_plane_mae_m"])
            <= args.maximum_validation_near_plane_mae_m
            and float(metrics["direction_accuracy"])
            >= args.minimum_validation_direction_accuracy
        )
    return (
        all(preservation.values()),
        progress_pass,
        float(metrics["correlation"]),
        -float(metrics["near_plane_mae_m"]),
        float(metrics["direction_accuracy"]),
        -float(metrics["mae_m"]),
        -int(candidate["step"]),
    )


def _candidate_should_replace(
    best: dict[str, object] | None,
    candidate: dict[str, object],
    args: argparse.Namespace,
) -> bool:
    if best is None:
        return True
    if args.fixed_final_selection:
        return int(candidate["step"]) > int(best["step"])
    return _candidate_key(candidate, args) > _candidate_key(best, args)


def _source_contracts(
    args: argparse.Namespace, source_hashes: dict[str, object]
) -> tuple[dict, dict, dict]:
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    n669_report = json.loads(args.n669_report.read_text(encoding="utf-8"))
    n670_report = json.loads(args.n670_report.read_text(encoding="utf-8"))
    if (
        source_report.get("contract")
        != "vq2_actor_frozen_gate_event_windows_replay_filtered_v1"
        or not source_report.get("process_passed")
        or source_report.get("output_dataset_sha256") != source_hashes["dataset"]
    ):
        raise RuntimeError("N668 source contract mismatch")
    if (
        n669_report.get("contract")
        != "vq2_training_only_success_prefix_recurrent_capacity_v1"
        or not n669_report.get("process_passed")
        or not n669_report.get("selected_passes_capacity_gate")
        or n669_report.get("source_hashes", {}).get("dataset")
        != source_hashes["dataset"]
        or n669_report.get("source_hashes", {}).get("source_report")
        != source_hashes["source_report"]
    ):
        raise RuntimeError("N669 source contract mismatch")
    if (
        n670_report.get("contract") != CONTINUATION_SCHEMA
        or not n670_report.get("process_passed")
        or n670_report.get("validation_admitted") is not False
        or n670_report.get("output_checkpoint_sha256")
        != source_hashes["n670_checkpoint"]
    ):
        raise RuntimeError("N670 rejected-source contract mismatch")
    return source_report, n669_report, n670_report


def _partition_source_contracts(
    args: argparse.Namespace, source_hashes: dict[str, object]
) -> tuple[dict, dict]:
    materialization = json.loads(
        args.materialization_report.read_text(encoding="utf-8")
    )
    support = json.loads(args.support_report.read_text(encoding="utf-8"))
    batch_audit = json.loads(
        args.reference_batch_audit_report.read_text(encoding="utf-8")
    )
    pair_support = json.loads(
        args.pair_support_report.read_text(encoding="utf-8")
    )
    if (
        materialization.get("contract") != MATERIALIZATION_SCHEMA
        or not materialization.get("process_passed")
        or materialization.get("output_sha256", {}).get("train")
        != source_hashes["dataset"]
        or materialization.get("output_sha256", {}).get("validation")
        != source_hashes["validation_dataset"]
        or materialization.get("test_rows_scored") != 0
        or materialization.get("test_values_used_for_selection") != 0
    ):
        raise RuntimeError("partition materialization contract mismatch")
    if (
        support.get("contract") != SUPPORT_AUDIT_SCHEMA
        or not support.get("process_passed")
        or support.get("source_hashes")
        != {
            "train": source_hashes["dataset"],
            "validation": source_hashes["validation_dataset"],
            "materialization_report": source_hashes["materialization_report"],
        }
        or support.get("test_dataset_path_received") != 0
        or support.get("test_dataset_opened") != 0
        or support.get("test_rows_scored") != 0
    ):
        raise RuntimeError("train/validation support audit contract mismatch")
    matching_batches = [
        row
        for row in batch_audit.get("batch_results", ())
        if row.get("microbatch_size") == args.fixed_reference_batch_size
    ]
    if (
        batch_audit.get("contract")
        != "vq2_partition_reference_batch_sensitivity_v1"
        or not batch_audit.get("process_passed")
        or batch_audit.get("source_hashes")
        != {
            "checkpoint": source_hashes["checkpoint"],
            "train": source_hashes["dataset"],
            "validation": source_hashes["validation_dataset"],
            "materialization_report": source_hashes["materialization_report"],
            "support_report": source_hashes["support_report"],
        }
        or len(matching_batches) != 1
        or matching_batches[0].get("deterministic_max_abs_error", math.inf) > 0.002
        or matching_batches[0].get("logits_max_abs_error", math.inf) > 0.002
        or matching_batches[0].get("stochastic_index_mismatch_count") != 0
        or batch_audit.get("test_dataset_opened") != 0
    ):
        raise RuntimeError("fixed reference batch audit contract mismatch")
    if (
        pair_support.get("contract") != PAIR_AUDIT_SCHEMA
        or not pair_support.get("process_passed")
        or pair_support.get("source_hashes")
        != {
            "train": source_hashes["dataset"],
            "validation": source_hashes["validation_dataset"],
            "materialization_report": source_hashes["materialization_report"],
            "sample_support_report": source_hashes["support_report"],
        }
        or tuple(pair_support.get("crop_starts", ()))
        != tuple(int(value) for value in args.crop_starts.split(","))
        or pair_support.get("sequence_length") != args.sequence_length
        or pair_support.get("burn_in") != args.burn_in
        or pair_support.get("near_plane_m") != args.near_plane_m
        or pair_support.get("minimum_pair_delta_m") != args.minimum_pair_delta_m
        or pair_support.get("near_plane_multiplier")
        != args.near_plane_loss_multiplier
        or pair_support.get("test_dataset_opened") != 0
    ):
        raise RuntimeError("temporal-pair support audit contract mismatch")
    return materialization, support


def continue_joint(args: argparse.Namespace) -> dict[str, object]:
    if args.output_checkpoint.resolve() == args.checkpoint.resolve():
        raise ValueError("joint integration cannot overwrite its parent")
    if args.output_checkpoint.exists() or args.report.exists():
        raise ValueError("joint integration refuses to overwrite outputs")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    partitioned_source_mode = args.validation_dataset is not None
    tangent_enabled = args.tangent_residual_weight > 0.0
    invariant_enabled = args.invariant_progress_weight > 0.0
    tangent_source_contract_exact = (
        not tangent_enabled
        or _tangent_source_contract_exact(args.tangent_source_contract_report)
    )
    if tangent_enabled and not tangent_source_contract_exact:
        raise RuntimeError("N700 tangent source contract mismatch")
    invariant_source_contract_exact = (
        not invariant_enabled
        or _invariant_source_contract_exact(args.invariant_source_contract_report)
    )
    if invariant_enabled and not invariant_source_contract_exact:
        raise RuntimeError("N704 invariant source contract mismatch")
    frozen_readout_contract_exact = (
        not args.fixed_final_selection
        or _frozen_readout_contract_exact(args.frozen_readout_smoke_report)
    )
    if args.fixed_final_selection and not frozen_readout_contract_exact:
        raise RuntimeError("N706 frozen-readout source contract mismatch")
    if partitioned_source_mode:
        source_before: dict[str, object] = {
            "executable": _sha256(Path(__file__)),
            "checkpoint": _sha256(args.checkpoint),
            "dataset": _sha256(args.dataset),
            "validation_dataset": _sha256(args.validation_dataset),
            "materialization_report": _sha256(args.materialization_report),
            "support_report": _sha256(args.support_report),
            "pair_support_report": _sha256(args.pair_support_report),
            "reference_batch_audit_report": _sha256(
                args.reference_batch_audit_report
            ),
            "dense_replay": _replay_hashes(args.dense_replay_dir),
        }
        if tangent_enabled:
            source_before["tangent_source_contract_report"] = _sha256(
                args.tangent_source_contract_report
            )
        if invariant_enabled:
            source_before["invariant_source_contract_report"] = _sha256(
                args.invariant_source_contract_report
            )
        if args.fixed_final_selection:
            source_before["frozen_readout_smoke_report"] = _sha256(
                args.frozen_readout_smoke_report
            )
        _materialization, support_report = _partition_source_contracts(
            args, source_before
        )
        n669_report = None
    else:
        source_before = {
            "executable": _sha256(Path(__file__)),
            "checkpoint": _sha256(args.checkpoint),
            "dataset": _sha256(args.dataset),
            "source_report": _sha256(args.source_report),
            "n669_report": _sha256(args.n669_report),
            "n670_checkpoint": _sha256(args.n670_checkpoint),
            "n670_report": _sha256(args.n670_report),
            "dense_replay": _replay_hashes(args.dense_replay_dir),
        }
        _source_report, n669_report, _n670_report = _source_contracts(
            args, source_before
        )
        support_report = None
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    tangent_direction: torch.Tensor | None = None
    tangent_geometry: dict[str, float | int] = {}
    if tangent_enabled:
        tangent_direction, tangent_geometry = (
            _canonical_actor_tangent_null_direction(
                model.actor[0].weight,
                model.rssm.deterministic_size,
                model.rssm.stochastic_groups,
                model.rssm.stochastic_classes,
            )
        )
        tangent_direction = tangent_direction.to(device)
    probe = SoftPlaneProbe(model.rssm.feature_size, args.probe_hidden_size).to(device)
    target_mean: float | None = None
    target_std: float | None = None
    if partitioned_source_mode:
        arrays, event_split = _load_partitioned_arrays(
            args.dataset, args.validation_dataset
        )
    else:
        n670_payload = torch.load(
            args.n670_checkpoint, map_location="cpu", weights_only=False
        )
        auxiliary = n670_payload.get("success_prefix_representation_aux")
        if (
            not isinstance(auxiliary, dict)
            or auxiliary.get("schema") != CONTINUATION_SCHEMA
        ):
            raise RuntimeError("N670 checkpoint lacks its training-only probe")
        probe.load_state_dict(auxiliary["probe_state"])
        target_mean = float(auxiliary["target_mean"])
        target_std = float(auxiliary["target_std"])
        with np.load(args.dataset) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    phase_after = np.rint(arrays["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    phase_before = np.rint(
        arrays["tail"][:, :-1, legal_tail_size + 32].astype(np.float64) * 6.0
    ).astype(np.int64)
    if not np.all(phase_before == 0) or not np.all(phase_after == 1):
        raise RuntimeError("successful-prefix phase contract mismatch")
    if not partitioned_source_mode:
        event_split = _stratified_group_split(
            arrays["vector_step"].astype(np.int64),
            phase_after,
            seed=args.split_seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    crop_contract = support_report if partitioned_source_mode else n669_report
    assert crop_contract is not None
    if (
        tuple(crop_contract.get("crop_starts", ())) != crop_starts
        or crop_contract.get("sequence_length") != args.sequence_length
        or crop_contract.get("burn_in") != args.burn_in
    ):
        raise RuntimeError("successful-prefix crop contract mismatch")
    original_event, crop_start = _crop_index(len(event_split), crop_starts)
    crop_split = event_split[original_event]
    train_crops = np.flatnonzero(crop_split == 0)
    train_events = np.flatnonzero(event_split == 0)
    validation_events = np.flatnonzero(event_split == 1)
    allowed_events = np.concatenate((train_events, validation_events))
    allowed_events.sort()
    event_to_reference = np.full(len(event_split), -1, dtype=np.int64)
    event_to_reference[allowed_events] = np.arange(len(allowed_events))
    reference_event = event_to_reference[original_event]
    evaluation_indices = np.arange(
        args.burn_in - 1,
        args.sequence_length,
        args.evaluation_stride,
        dtype=np.int64,
    )
    exposure_by_history_index = _crop_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
    )
    pair_exposure_by_endpoint = _crop_pair_exposure_counts(
        arrays["mask"].shape[1],
        crop_starts,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
    )
    if partitioned_source_mode:
        train_target = _target_batch(
            arrays,
            original_event[train_crops],
            crop_start[train_crops],
            sequence_length=args.sequence_length,
        )[:, args.burn_in :]
        if args.inverse_crop_exposure_weighting:
            normalization_index = crop_start[train_crops, None] + np.arange(
                args.burn_in, args.sequence_length, dtype=np.int64
            )[None, :]
            normalization_weight = 1.0 / exposure_by_history_index[
                normalization_index
            ]
            normalization_weight_sum = float(normalization_weight.sum())
            target_mean = float(
                (train_target * normalization_weight).sum()
                / normalization_weight_sum
            )
            target_std = float(
                np.sqrt(
                    (
                        np.square(train_target - target_mean)
                        * normalization_weight
                    ).sum()
                    / normalization_weight_sum
                )
            )
        else:
            target_mean = float(train_target.mean())
            target_std = float(train_target.std())
        if target_std <= 0.0:
            raise RuntimeError("partitioned training plane target has zero variance")
    assert target_mean is not None and target_std is not None
    if args.fixed_reference_batch_size:
        reference = _fixed_parent_reference(
            model,
            arrays,
            allowed_events,
            batch_size=args.fixed_reference_batch_size,
            device=device,
        )
    else:
        reference = _parent_reference(
            model,
            arrays,
            allowed_events,
            microbatch_size=args.reference_microbatch_size,
            device=device,
        )
    reference["soft_action"] = _parent_soft_action(model, reference)
    boundary_replay = {
        "deterministic_max_abs_error": float(
            np.abs(
                reference["deterministic"][:, -1]
                - arrays["preevent_deterministic"][allowed_events]
            ).max()
        ),
        "logits_max_abs_error": float(
            np.abs(
                reference["logits"][:, -1]
                - arrays["preevent_logits"][allowed_events]
            ).max()
        ),
        "stochastic_index_mismatch_count": int(
            (
                reference["logits"][:, -1].argmax(-1)
                != arrays["preevent_stochastic_index"][allowed_events]
            ).sum()
        ),
    }

    replay_capacity, replay_agents = _replay_geometry(args.dense_replay_dir)
    replay = QuantizedSequenceReplay(
        replay_capacity,
        replay_agents,
        storage_dir=args.dense_replay_dir,
        resume=True,
    )
    dense_pool = _sample_dense_pool(
        model,
        replay,
        pool_size=args.dense_pool_size,
        sequence_length=args.dense_sequence_length,
        seed=args.dense_seed,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    representation_parameters = [
        *model.rssm.encoder.parameters(),
        *model.rssm.sequence.parameters(),
        *model.rssm.posterior.parameters(),
    ]
    for parameter in representation_parameters:
        parameter.requires_grad_(True)
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    representation_optimizer = torch.optim.AdamW(
        representation_parameters,
        lr=args.representation_learning_rate,
        weight_decay=args.representation_weight_decay,
    )
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_learning_rate,
        weight_decay=args.probe_weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    train_sampler = None
    dense_sampler = None
    if args.balanced_epoch_sampling:
        train_sampler = _EpochBatchSampler(train_crops, args.batch_size, rng)
        dense_sampler = _EpochBatchSampler(
            np.arange(args.dense_pool_size, dtype=np.int64),
            args.dense_batch_size,
            rng,
        )
    train_crop_position = np.full(len(original_event), -1, dtype=np.int64)
    train_crop_position[train_crops] = np.arange(len(train_crops), dtype=np.int64)
    train_sampling_counts = np.zeros(len(train_crops), dtype=np.int64)
    dense_sampling_counts = np.zeros(args.dense_pool_size, dtype=np.int64)
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_probe_state: dict[str, torch.Tensor] | None = None
    max_gradient = {"representation": 0.0, "probe": 0.0}
    component_gradient_max = {"encoder": 0.0, "sequence": 0.0, "posterior": 0.0}
    tangent_component_gradient_max = {
        "encoder": 0.0,
        "sequence": 0.0,
        "posterior": 0.0,
    }
    invariant_component_gradient_max = {
        "encoder": 0.0,
        "sequence": 0.0,
        "posterior": 0.0,
    }
    max_preprojection_delta = 0.0
    projected_values = 0
    last_losses: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        model.train()
        probe.train()
        if train_sampler is None:
            chosen_crop = rng.choice(
                train_crops,
                args.batch_size,
                replace=len(train_crops) < args.batch_size,
            )
        else:
            chosen_crop = train_sampler.draw()
        np.add.at(train_sampling_counts, train_crop_position[chosen_crop], 1)
        original = original_event[chosen_crop]
        ref = reference_event[chosen_crop]
        starts = crop_start[chosen_crop]
        legal, input_action = _legal_action_batch(
            arrays,
            original,
            starts,
            sequence_length=args.sequence_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model,
            legal,
            input_action,
            _crop_initial_state(
                arrays, reference, ref, original, starts, device=device
            ),
        )
        soft_feature = _soft_posterior_features(state)
        probe_feature = (
            soft_feature.detach()
            if args.detach_probe_representation
            else soft_feature
        )
        prediction = probe(probe_feature)[:, args.burn_in :]
        target_np = _target_batch(
            arrays, original, starts, sequence_length=args.sequence_length
        )
        target_m = torch.from_numpy(target_np).to(device)[:, args.burn_in :]
        normalized_target = (target_m - target_mean) / target_std
        if (
            args.inverse_crop_exposure_weighting
            or args.near_plane_loss_multiplier != 1.0
        ):
            plane_loss, plane_weight = _weighted_plane_loss(
                prediction,
                normalized_target,
                target_m,
                starts,
                exposure_by_history_index,
                burn_in=args.burn_in,
                near_plane_m=args.near_plane_m,
                near_plane_multiplier=args.near_plane_loss_multiplier,
            )
        else:
            plane_loss = F.mse_loss(prediction, normalized_target)
            plane_weight = torch.ones_like(prediction)
        if args.inverse_crop_pair_exposure_weighting:
            delta_loss, delta_weight = _weighted_delta_loss(
                prediction,
                normalized_target,
                target_m,
                starts,
                pair_exposure_by_endpoint,
                burn_in=args.burn_in,
                near_plane_m=args.near_plane_m,
                near_plane_multiplier=args.near_plane_loss_multiplier,
            )
        else:
            delta_loss = F.mse_loss(
                torch.diff(prediction, dim=1),
                torch.diff(normalized_target, dim=1),
            )
            delta_weight = torch.ones_like(prediction[:, 1:])
        if invariant_enabled:
            invariant_progress_loss, invariant_progress_metrics = (
                _invariant_progress_loss(
                    soft_feature[:, args.burn_in :],
                    normalized_target,
                    plane_weight,
                    delta_weight,
                    level_coefficient=args.invariant_level_coefficient,
                    delta_coefficient=args.invariant_delta_coefficient,
                )
            )
        else:
            invariant_progress_loss = plane_loss.new_zeros(())
            invariant_progress_metrics = {}
        parent_det = torch.from_numpy(
            _crop_full(
                reference["deterministic"], ref, starts,
                sequence_length=args.sequence_length,
            )
        ).to(device)
        parent_logits = torch.from_numpy(
            _crop_full(
                reference["logits"], ref, starts,
                sequence_length=args.sequence_length,
            )
        ).to(device)
        parent_hard_action = torch.from_numpy(
            _crop_full(
                reference["action"], ref, starts,
                sequence_length=args.sequence_length,
            )
        ).to(device)
        parent_soft_action = torch.from_numpy(
            _crop_full(
                reference["soft_action"], ref, starts,
                sequence_length=args.sequence_length,
            )
        ).to(device)
        child_hard_action = model.deterministic_actor_action(
            model.actor_distribution(state.features)
        )
        child_soft_action = _soft_actor_action(model, state)
        success_feature_loss = F.mse_loss(
            state.deterministic.float(), parent_det.float()
        ) + F.mse_loss(state.logits.float(), parent_logits.float())
        success_hard_action_loss = F.mse_loss(
            child_hard_action.float(), parent_hard_action.float()
        )
        success_soft_action_loss = F.mse_loss(
            child_soft_action.float(), parent_soft_action.float()
        )
        if tangent_enabled:
            assert tangent_direction is not None
            (
                tangent_residual_loss,
                tangent_plane_loss,
                tangent_delta_loss,
            ) = _tangent_residual_losses(
                _soft_posterior_features(state),
                parent_det,
                parent_logits,
                tangent_direction,
                normalized_target,
                target_m,
                starts,
                exposure_by_history_index,
                pair_exposure_by_endpoint,
                scale=args.tangent_residual_scale,
                burn_in=args.burn_in,
                near_plane_m=args.near_plane_m,
                near_plane_multiplier=args.near_plane_loss_multiplier,
                delta_weight=args.delta_weight,
            )
        else:
            tangent_residual_loss = plane_loss.new_zeros(())
            tangent_plane_loss = plane_loss.new_zeros(())
            tangent_delta_loss = plane_loss.new_zeros(())

        if dense_sampler is None:
            dense_index = rng.choice(
                args.dense_pool_size,
                args.dense_batch_size,
                replace=args.dense_pool_size < args.dense_batch_size,
            )
        else:
            dense_index = dense_sampler.draw()
        np.add.at(dense_sampling_counts, dense_index, 1)
        dense_legal = dense_pool["legal"][dense_index].to(device)
        dense_input_action = dense_pool["input_action"][dense_index].to(device)
        dense_state = _observe_legal_sequence(
            model,
            dense_legal,
            dense_input_action,
            model.rssm.initial(len(dense_index), device=device),
        )
        dense_feature_loss = F.mse_loss(
            dense_state.deterministic.float(),
            dense_pool["deterministic"][dense_index].to(device).float(),
        ) + F.mse_loss(
            dense_state.logits.float(),
            dense_pool["logits"][dense_index].to(device).float(),
        )
        dense_hard_action_loss = F.mse_loss(
            model.deterministic_actor_action(
                model.actor_distribution(dense_state.features)
            ).float(),
            dense_pool["hard_action"][dense_index].to(device).float(),
        )
        dense_soft_action_loss = F.mse_loss(
            _soft_actor_action(model, dense_state).float(),
            dense_pool["soft_action"][dense_index].to(device).float(),
        )
        loss = (
            plane_loss
            + args.delta_weight * delta_loss
            + args.success_feature_weight * success_feature_loss
            + args.success_hard_action_weight * success_hard_action_loss
            + args.success_soft_action_weight * success_soft_action_loss
            + args.dense_feature_weight * dense_feature_loss
            + args.dense_hard_action_weight * dense_hard_action_loss
            + args.dense_soft_action_weight * dense_soft_action_loss
            + args.tangent_residual_weight * tangent_residual_loss
            + args.invariant_progress_weight * invariant_progress_loss
        )
        representation_optimizer.zero_grad(set_to_none=True)
        probe_optimizer.zero_grad(set_to_none=True)
        if tangent_enabled:
            tangent_gradients = torch.autograd.grad(
                args.tangent_residual_weight * tangent_residual_loss,
                representation_parameters,
                retain_graph=True,
                allow_unused=False,
            )
            gradient_cursor = 0
            for component_name, module in (
                ("encoder", model.rssm.encoder),
                ("sequence", model.rssm.sequence),
                ("posterior", model.rssm.posterior),
            ):
                parameter_count = len(list(module.parameters()))
                component_gradients = tangent_gradients[
                    gradient_cursor : gradient_cursor + parameter_count
                ]
                gradient_cursor += parameter_count
                tangent_component_gradient_max[component_name] = max(
                    tangent_component_gradient_max[component_name],
                    max(
                        float(gradient.detach().abs().max().cpu())
                        for gradient in component_gradients
                    ),
                )
        if invariant_enabled:
            invariant_gradients = torch.autograd.grad(
                args.invariant_progress_weight * invariant_progress_loss,
                representation_parameters,
                retain_graph=True,
                allow_unused=False,
            )
            gradient_cursor = 0
            for component_name, module in (
                ("encoder", model.rssm.encoder),
                ("sequence", model.rssm.sequence),
                ("posterior", model.rssm.posterior),
            ):
                parameter_count = len(list(module.parameters()))
                component_gradients = invariant_gradients[
                    gradient_cursor : gradient_cursor + parameter_count
                ]
                gradient_cursor += parameter_count
                invariant_component_gradient_max[component_name] = max(
                    invariant_component_gradient_max[component_name],
                    max(
                        float(gradient.detach().abs().max().cpu())
                        for gradient in component_gradients
                    ),
                )
        loss.backward()
        for component_name, module in (
            ("encoder", model.rssm.encoder),
            ("sequence", model.rssm.sequence),
            ("posterior", model.rssm.posterior),
        ):
            component_gradients = [
                float(parameter.grad.detach().abs().max().cpu())
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            if component_gradients:
                component_gradient_max[component_name] = max(
                    component_gradient_max[component_name],
                    max(component_gradients),
                )
        representation_norm = _finite_clip_grad_norm(
            representation_parameters,
            args.representation_gradient_clip,
            label="joint representation",
        )
        probe_norm = _finite_clip_grad_norm(
            list(probe.parameters()), args.probe_gradient_clip, label="joint probe"
        )
        max_gradient["representation"] = max(
            max_gradient["representation"], float(representation_norm.cpu())
        )
        max_gradient["probe"] = max(
            max_gradient["probe"], float(probe_norm.cpu())
        )
        representation_optimizer.step()
        probe_optimizer.step()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                delta = parameter - parent_model_state[name]
                max_preprojection_delta = max(
                    max_preprojection_delta, float(delta.abs().max().cpu())
                )
                clipped = delta.clamp(
                    -args.maximum_parameter_delta, args.maximum_parameter_delta
                )
                projected_values += int((clipped != delta).sum().cpu())
                parameter.copy_(parent_model_state[name] + clipped)
        last_losses = {
            "total": float(loss.detach().cpu()),
            "plane": float(plane_loss.detach().cpu()),
            "delta": float(delta_loss.detach().cpu()),
            "success_feature": float(success_feature_loss.detach().cpu()),
            "success_hard_action": float(success_hard_action_loss.detach().cpu()),
            "success_soft_action": float(success_soft_action_loss.detach().cpu()),
            "dense_feature": float(dense_feature_loss.detach().cpu()),
            "dense_hard_action": float(dense_hard_action_loss.detach().cpu()),
            "dense_soft_action": float(dense_soft_action_loss.detach().cpu()),
            "tangent_residual": float(tangent_residual_loss.detach().cpu()),
            "tangent_plane": float(tangent_plane_loss.detach().cpu()),
            "tangent_delta": float(tangent_delta_loss.detach().cpu()),
            "invariant_progress": float(
                invariant_progress_loss.detach().cpu()
            ),
            "invariant_level_score": (
                invariant_progress_metrics.get("level", {}).get("score", 0.0)
            ),
            "invariant_delta_score": (
                invariant_progress_metrics.get("delta", {}).get("score", 0.0)
            ),
            "plane_weight_min": float(plane_weight.min().detach().cpu()),
            "plane_weight_mean": float(plane_weight.mean().detach().cpu()),
            "plane_weight_max": float(plane_weight.max().detach().cpu()),
            "delta_weight_min": float(delta_weight.min().detach().cpu()),
            "delta_weight_mean": float(delta_weight.mean().detach().cpu()),
            "delta_weight_max": float(delta_weight.max().detach().cpu()),
        }
        if step % args.validation_interval and step != args.steps:
            continue
        validation_progress, validation_action = _continuous_validation(
            model,
            probe,
            arrays,
            reference,
            validation_events,
            event_to_reference,
            crop_starts,
            evaluation_indices,
            sequence_length=args.sequence_length,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
            target_mean=target_mean,
            target_std=target_std,
            microbatch_size=args.reference_microbatch_size,
            device=device,
        )
        dense_child = _dense_outputs(
            model,
            dense_pool,
            microbatch_size=args.reference_microbatch_size,
            device=device,
        )
        dense_drift = {
            name: _drift(dense_child[name], dense_pool[name])
            for name in ("deterministic", "logits", "hard_action", "soft_action")
        }
        candidate: dict[str, object] = {
            "step": step,
            "validation_progress": validation_progress,
            "validation_action_drift": validation_action,
            "dense_drift": dense_drift,
            "preservation_gates": _preservation(
                validation_action, dense_drift, args
            ),
        }
        history.append(candidate)
        if _candidate_should_replace(best, candidate, args):
            best = copy.deepcopy(candidate)
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_probe_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }

    assert best is not None and best_model_state is not None and best_probe_state is not None
    model.load_state_dict(best_model_state)
    probe.load_state_dict(best_probe_state)
    model.eval()
    probe.eval()
    changed, forbidden = _changed_model_keys(parent_model_state, model.state_dict())
    final_max_parameter_delta = max(
        (
            float((value - parent_model_state[name]).abs().max().cpu())
            for name, value in model.state_dict().items()
            if name in changed
        ),
        default=0.0,
    )
    selected_progress = best["validation_progress"]
    selected_preservation = best["preservation_gates"]
    assert isinstance(selected_progress, dict) and isinstance(selected_preservation, dict)
    progress_gates = {
        "correlation_at_least_minimum": (
            selected_progress["correlation"] >= args.minimum_validation_correlation
        ),
        "mae_at_most_maximum": (
            selected_progress["mae_m"] <= args.maximum_validation_mae_m
        ),
        "near_plane_mae_at_most_maximum": (
            selected_progress["near_plane_mae_m"]
            <= args.maximum_validation_near_plane_mae_m
        ),
        "direction_at_least_minimum": (
            selected_progress["direction_accuracy"]
            >= args.minimum_validation_direction_accuracy
        ),
    }
    if partitioned_source_mode:
        source_after: dict[str, object] = {
            "executable": _sha256(Path(__file__)),
            "checkpoint": _sha256(args.checkpoint),
            "dataset": _sha256(args.dataset),
            "validation_dataset": _sha256(args.validation_dataset),
            "materialization_report": _sha256(args.materialization_report),
            "support_report": _sha256(args.support_report),
            "pair_support_report": _sha256(args.pair_support_report),
            "reference_batch_audit_report": _sha256(
                args.reference_batch_audit_report
            ),
            "dense_replay": _replay_hashes(args.dense_replay_dir),
        }
        if tangent_enabled:
            source_after["tangent_source_contract_report"] = _sha256(
                args.tangent_source_contract_report
            )
        if invariant_enabled:
            source_after["invariant_source_contract_report"] = _sha256(
                args.invariant_source_contract_report
            )
        if args.fixed_final_selection:
            source_after["frozen_readout_smoke_report"] = _sha256(
                args.frozen_readout_smoke_report
            )
    else:
        source_after = {
            "executable": _sha256(Path(__file__)),
            "checkpoint": _sha256(args.checkpoint),
            "dataset": _sha256(args.dataset),
            "source_report": _sha256(args.source_report),
            "n669_report": _sha256(args.n669_report),
            "n670_checkpoint": _sha256(args.n670_checkpoint),
            "n670_report": _sha256(args.n670_report),
            "dense_replay": _replay_hashes(args.dense_replay_dir),
        }
    train_event_draws = np.bincount(
        original_event[train_crops],
        weights=train_sampling_counts,
        minlength=len(event_split),
    )[train_events]
    sampling_summary = {
        "train_draws": int(train_sampling_counts.sum()),
        "train_count_minimum": int(train_sampling_counts.min()),
        "train_count_maximum": int(train_sampling_counts.max()),
        "train_count_std": float(train_sampling_counts.std()),
        "train_unseen_crops": int((train_sampling_counts == 0).sum()),
        "train_event_draw_minimum": int(train_event_draws.min()),
        "train_event_draw_maximum": int(train_event_draws.max()),
        "train_event_draw_std": float(train_event_draws.std()),
        "dense_draws": int(dense_sampling_counts.sum()),
        "dense_count_minimum": int(dense_sampling_counts.min()),
        "dense_count_maximum": int(dense_sampling_counts.max()),
        "dense_count_std": float(dense_sampling_counts.std()),
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "boundary_deterministic_at_most_0p002": (
            boundary_replay["deterministic_max_abs_error"] <= 0.002
        ),
        "boundary_logits_at_most_0p002": (
            boundary_replay["logits_max_abs_error"] <= 0.002
        ),
        "boundary_stochastic_exact": (
            boundary_replay["stochastic_index_mismatch_count"] == 0
        ),
        "model_changed": bool(changed),
        "only_allowed_model_tensors_changed": not forbidden,
        "actor_weights_exact": not any(name.startswith("actor.") for name in changed),
        "gradient_reaches_encoder": component_gradient_max["encoder"] > 0.0,
        "gradient_reaches_sequence": component_gradient_max["sequence"] > 0.0,
        "gradient_reaches_posterior": component_gradient_max["posterior"] > 0.0,
        "test_labels_not_evaluated": True,
        "sealed_test_path_not_received": True,
        "balanced_sampler_count_spread_at_most_one": (
            not args.balanced_epoch_sampling
            or (
                sampling_summary["train_count_maximum"]
                - sampling_summary["train_count_minimum"]
                <= 1
                and sampling_summary["dense_count_maximum"]
                - sampling_summary["dense_count_minimum"]
                <= 1
            )
        ),
        "finite_validation_history": all(
            math.isfinite(float(row["validation_progress"][key]))
            for row in history
                for key in ("correlation", "mae_m", "near_plane_mae_m", "direction_accuracy")
        ),
        "tangent_source_contract_exact": tangent_source_contract_exact,
        "tangent_geometry_exact": (
            not tangent_enabled
            or (
                tangent_geometry["actor_first_layer_residual_max_abs"] <= 1e-8
                and tangent_geometry["simplex_tangent_residual_max_abs"] <= 1e-8
                and tangent_geometry["stochastic_direction_norm"] > 0.0
                and abs(float(tangent_geometry["direction_norm"]) - 1.0) <= 1e-6
            )
        ),
        "tangent_loss_finite": (
            not tangent_enabled
            or all(
                math.isfinite(last_losses[name])
                for name in ("tangent_residual", "tangent_plane", "tangent_delta")
            )
        ),
        "tangent_gradient_reaches_encoder": (
            not tangent_enabled or tangent_component_gradient_max["encoder"] > 0.0
        ),
        "tangent_gradient_reaches_sequence": (
            not tangent_enabled or tangent_component_gradient_max["sequence"] > 0.0
        ),
        "tangent_gradient_reaches_posterior": (
            not tangent_enabled or tangent_component_gradient_max["posterior"] > 0.0
        ),
        "invariant_source_contract_exact": invariant_source_contract_exact,
        "invariant_probe_representation_detached": (
            not invariant_enabled or args.detach_probe_representation
        ),
        "invariant_loss_finite": (
            not invariant_enabled
            or math.isfinite(last_losses["invariant_progress"])
        ),
        "invariant_gradient_reaches_encoder": (
            not invariant_enabled
            or invariant_component_gradient_max["encoder"] > 0.0
        ),
        "invariant_gradient_reaches_sequence": (
            not invariant_enabled
            or invariant_component_gradient_max["sequence"] > 0.0
        ),
        "invariant_gradient_reaches_posterior": (
            not invariant_enabled
            or invariant_component_gradient_max["posterior"] > 0.0
        ),
        "frozen_readout_contract_exact": frozen_readout_contract_exact,
        "fixed_final_step_selected": (
            not args.fixed_final_selection or int(best["step"]) == args.steps
        ),
    }
    cuda_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    validation_admitted = (
        all(process_gates.values())
        and all(selected_preservation.values())
        and all(progress_gates.values())
        and final_max_parameter_delta
        <= args.maximum_parameter_delta + args.parameter_delta_tolerance
        and resource_gate
    )
    representation_training_passed = (
        all(process_gates.values())
        and all(selected_preservation.values())
        and final_max_parameter_delta
        <= args.maximum_parameter_delta + args.parameter_delta_tolerance
        and resource_gate
    )
    output_payload = dict(payload)
    output_payload["model"] = model.state_dict()
    output_payload["success_prefix_joint_representation_aux"] = {
        "schema": JOINT_SCHEMA,
        "probe_state": {
            name: value.detach().cpu() for name, value in probe.state_dict().items()
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "selected_step": int(best["step"]),
        "representation_updates": int(best["step"]),
        "crop_starts": crop_starts,
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "split_seed": args.split_seed,
        "partitioned_source_mode": partitioned_source_mode,
        "inverse_crop_exposure_weighting": args.inverse_crop_exposure_weighting,
        "inverse_crop_pair_exposure_weighting": (
            args.inverse_crop_pair_exposure_weighting
        ),
        "near_plane_loss_multiplier": args.near_plane_loss_multiplier,
        "fixed_reference_batch_size": args.fixed_reference_batch_size,
        "balanced_epoch_sampling": args.balanced_epoch_sampling,
        "selection_prioritizes_complete_progress_pass": True,
        "tangent_residual_weight": args.tangent_residual_weight,
        "tangent_residual_scale": args.tangent_residual_scale,
        "tangent_geometry": tangent_geometry,
        "tangent_source_contract_sha256": (
            _sha256(args.tangent_source_contract_report)
            if tangent_enabled
            else None
        ),
        "invariant_progress_weight": args.invariant_progress_weight,
        "invariant_level_coefficient": args.invariant_level_coefficient,
        "invariant_delta_coefficient": args.invariant_delta_coefficient,
        "detach_probe_representation": args.detach_probe_representation,
        "invariant_source_contract_sha256": (
            _sha256(args.invariant_source_contract_report)
            if invariant_enabled
            else None
        ),
        "fixed_final_selection": args.fixed_final_selection,
        "frozen_readout_smoke_sha256": (
            _sha256(args.frozen_readout_smoke_report)
            if args.fixed_final_selection
            else None
        ),
    }
    _save_checkpoint(args.output_checkpoint, output_payload)
    report: dict[str, object] = {
        "contract": JOINT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "events": int(len(event_split)),
        "train_events": int(len(train_events)),
        "validation_events": int(len(validation_events)),
        "test_events": int((event_split == 2).sum()),
        "train_crops": int(len(train_crops)),
        "test_evaluations": 0,
        "test_observations_evaluated": 0,
        "test_privileged_labels_evaluated": 0,
        "crop_starts": list(crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "partitioned_source_mode": partitioned_source_mode,
        "inverse_crop_exposure_weighting": args.inverse_crop_exposure_weighting,
        "inverse_crop_pair_exposure_weighting": (
            args.inverse_crop_pair_exposure_weighting
        ),
        "near_plane_loss_multiplier": args.near_plane_loss_multiplier,
        "fixed_reference_batch_size": args.fixed_reference_batch_size,
        "balanced_epoch_sampling": args.balanced_epoch_sampling,
        "sampling_summary": sampling_summary,
        "selection_prioritizes_complete_progress_pass": True,
        "tangent_residual_enabled": tangent_enabled,
        "tangent_residual_weight": args.tangent_residual_weight,
        "tangent_residual_scale": args.tangent_residual_scale,
        "tangent_geometry": tangent_geometry,
        "tangent_source_contract_sha256": (
            _sha256(args.tangent_source_contract_report)
            if tangent_enabled
            else None
        ),
        "invariant_progress_enabled": invariant_enabled,
        "invariant_progress_weight": args.invariant_progress_weight,
        "invariant_level_coefficient": args.invariant_level_coefficient,
        "invariant_delta_coefficient": args.invariant_delta_coefficient,
        "detach_probe_representation": args.detach_probe_representation,
        "invariant_source_contract_sha256": (
            _sha256(args.invariant_source_contract_report)
            if invariant_enabled
            else None
        ),
        "fixed_final_selection": args.fixed_final_selection,
        "selection_mode": (
            "fixed_final_step"
            if args.fixed_final_selection
            else "validation_progress"
        ),
        "frozen_readout_smoke_sha256": (
            _sha256(args.frozen_readout_smoke_report)
            if args.fixed_final_selection
            else None
        ),
        "target_mean": target_mean,
        "target_std": target_std,
        "sealed_test_dataset_path_received": 0,
        "sealed_test_dataset_opened": 0,
        "evaluation_indices": evaluation_indices.tolist(),
        "boundary_replay": boundary_replay,
        "optimizer_steps": args.steps,
        "selected_step": int(best["step"]),
        "validation_history": history,
        "selected_validation_progress": selected_progress,
        "selected_validation_action_drift": best["validation_action_drift"],
        "selected_dense_drift": best["dense_drift"],
        "selected_preservation_gates": selected_preservation,
        "progress_gates": progress_gates,
        "last_training_losses": last_losses,
        "maximum_preclip_gradient_norm": max_gradient,
        "maximum_gradient_abs_by_component": component_gradient_max,
        "maximum_tangent_gradient_abs_by_component": (
            tangent_component_gradient_max
        ),
        "maximum_invariant_gradient_abs_by_component": (
            invariant_component_gradient_max
        ),
        "maximum_preprojection_parameter_delta": max_preprojection_delta,
        "final_maximum_parameter_delta": final_max_parameter_delta,
        "projected_parameter_values": projected_values,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "resource_gate": resource_gate,
        "validation_admitted": validation_admitted,
        "representation_training_passed": representation_training_passed,
        "probe_updates": args.steps,
        "representation_updates": args.steps,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "semantic_precision": "float32",
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
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--materialization-report", type=Path)
    parser.add_argument("--support-report", type=Path)
    parser.add_argument("--pair-support-report", type=Path)
    parser.add_argument("--reference-batch-audit-report", type=Path)
    parser.add_argument("--fixed-reference-batch-size", type=int, default=0)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--n669-report", type=Path)
    parser.add_argument("--n670-checkpoint", type=Path)
    parser.add_argument("--n670-report", type=Path)
    parser.add_argument("--dense-replay-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=668)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--probe-hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--balanced-epoch-sampling", action="store_true")
    parser.add_argument("--probe-learning-rate", type=float, default=0.0003)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0001)
    parser.add_argument("--probe-gradient-clip", type=float, default=10.0)
    parser.add_argument("--representation-learning-rate", type=float, default=0.00004)
    parser.add_argument("--representation-weight-decay", type=float, default=0.0)
    parser.add_argument("--representation-gradient-clip", type=float, default=10.0)
    parser.add_argument("--maximum-parameter-delta", type=float, default=0.001)
    parser.add_argument("--parameter-delta-tolerance", type=float, default=0.000001)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument(
        "--inverse-crop-exposure-weighting", action="store_true"
    )
    parser.add_argument(
        "--inverse-crop-pair-exposure-weighting", action="store_true"
    )
    parser.add_argument("--near-plane-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--tangent-residual-weight", type=float, default=0.0)
    parser.add_argument("--tangent-residual-scale", type=float, default=0.05)
    parser.add_argument("--tangent-source-contract-report", type=Path)
    parser.add_argument("--invariant-progress-weight", type=float, default=0.0)
    parser.add_argument("--invariant-level-coefficient", type=float, default=1.0)
    parser.add_argument("--invariant-delta-coefficient", type=float, default=1.0)
    parser.add_argument("--invariant-source-contract-report", type=Path)
    parser.add_argument("--detach-probe-representation", action="store_true")
    parser.add_argument("--fixed-final-selection", action="store_true")
    parser.add_argument("--frozen-readout-smoke-report", type=Path)
    parser.add_argument("--success-feature-weight", type=float, default=0.01)
    parser.add_argument("--success-hard-action-weight", type=float, default=1000.0)
    parser.add_argument("--success-soft-action-weight", type=float, default=1000.0)
    parser.add_argument("--dense-feature-weight", type=float, default=0.01)
    parser.add_argument("--dense-hard-action-weight", type=float, default=1000.0)
    parser.add_argument("--dense-soft-action-weight", type=float, default=1000.0)
    parser.add_argument("--dense-pool-size", type=int, default=128)
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--dense-sequence-length", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=671001)
    parser.add_argument("--minimum-validation-correlation", type=float, default=0.8)
    parser.add_argument("--maximum-validation-mae-m", type=float, default=0.5)
    parser.add_argument("--maximum-validation-near-plane-mae-m", type=float, default=0.5)
    parser.add_argument("--minimum-validation-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--maximum-dense-feature-rmse", type=float, default=0.02)
    parser.add_argument("--maximum-action-rmse", type=float, default=0.001)
    parser.add_argument("--maximum-action-max-abs", type=float, default=0.01)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=671)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "probe_hidden_size",
        "steps", "validation_interval", "batch_size", "dense_pool_size",
        "dense_batch_size", "dense_sequence_length", "reference_microbatch_size",
        "maximum_cuda_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    if args.maximum_parameter_delta <= 0.0:
        parser.error("maximum parameter delta must be positive")
    if args.tangent_residual_weight < 0.0:
        parser.error("tangent residual weight cannot be negative")
    if args.tangent_residual_scale <= 0.0:
        parser.error("tangent residual scale must be positive")
    if args.invariant_progress_weight < 0.0:
        parser.error("invariant progress weight cannot be negative")
    if (
        args.invariant_level_coefficient <= 0.0
        or args.invariant_delta_coefficient <= 0.0
    ):
        parser.error("invariant progress coefficients must be positive")
    partitioned = args.validation_dataset is not None
    partition_sources = (
        args.materialization_report,
        args.support_report,
        args.pair_support_report,
        args.reference_batch_audit_report,
    )
    legacy_sources = (
        args.source_report,
        args.n669_report,
        args.n670_checkpoint,
        args.n670_report,
    )
    if partitioned and any(path is None for path in partition_sources):
        parser.error(
            "partitioned mode requires materialization and support reports"
        )
    if not partitioned and any(path is None for path in legacy_sources):
        parser.error("legacy mode requires source, N669, and N670 artifacts")
    if args.near_plane_loss_multiplier <= 0.0:
        parser.error("near-plane loss multiplier must be positive")
    if args.fixed_reference_batch_size < 0:
        parser.error("fixed reference batch size cannot be negative")
    if partitioned and args.fixed_reference_batch_size <= 0:
        parser.error("partitioned mode requires a fixed reference batch size")
    if not partitioned and args.fixed_reference_batch_size:
        parser.error("fixed reference batching is only supported in partitioned mode")
    if args.inverse_crop_pair_exposure_weighting and not partitioned:
        parser.error("pair exposure weighting requires partitioned mode")
    if args.tangent_residual_weight > 0.0:
        if not partitioned:
            parser.error("tangent residual objective requires partitioned mode")
        if args.tangent_source_contract_report is None:
            parser.error("tangent residual objective requires the N700 report")
        if not (
            args.inverse_crop_exposure_weighting
            and args.inverse_crop_pair_exposure_weighting
        ):
            parser.error(
                "tangent residual objective requires inverse sample and pair weighting"
            )
    if args.invariant_progress_weight > 0.0:
        if not partitioned:
            parser.error("invariant progress objective requires partitioned mode")
        if args.invariant_source_contract_report is None:
            parser.error("invariant progress objective requires the N704 report")
        if not args.detach_probe_representation:
            parser.error(
                "invariant progress objective requires detached probe representation"
            )
        if not (
            args.inverse_crop_exposure_weighting
            and args.inverse_crop_pair_exposure_weighting
        ):
            parser.error(
                "invariant progress objective requires inverse sample and pair weighting"
            )
    if args.fixed_final_selection:
        if args.invariant_progress_weight <= 0.0:
            parser.error("fixed final selection requires invariant progress")
        if args.frozen_readout_smoke_report is None:
            parser.error("fixed final selection requires the N706 report")
    if (
        args.near_plane_loss_multiplier != 1.0
        and not args.inverse_crop_exposure_weighting
    ):
        parser.error("near-plane weighting requires inverse crop exposure weighting")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_joint(parse_args()), indent=2, sort_keys=True))
