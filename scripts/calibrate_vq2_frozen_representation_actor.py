#!/usr/bin/env python3
"""Calibrate an N674 probe and actor with the representation fully frozen."""

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
    _selection_key,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.continue_vq2_success_prefix_representation_offline import (
    SoftPlaneProbe,
    _crop_full,
    _drift,
    _initial_state,
    _legal_action_batch,
    _model_from_payload,
    _observe_legal_sequence,
    _replay_geometry,
    _soft_posterior_features,
    _target_batch,
)
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    JOINT_SCHEMA,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


CALIBRATION_SCHEMA = "vq2_frozen_representation_probe_actor_calibration_v1"


@torch.no_grad()
def _event_feature_corpus(
    model,
    arrays: dict[str, np.ndarray],
    event_ids: np.ndarray,
    *,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    hard_rows: list[np.ndarray] = []
    soft_rows: list[np.ndarray] = []
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
        hard_rows.append(state.features.float().cpu().numpy())
        soft_rows.append(_soft_posterior_features(state).cpu().numpy())
    return {
        "hard": np.concatenate(hard_rows).astype(np.float32),
        "soft": np.concatenate(soft_rows).astype(np.float32),
    }


@torch.no_grad()
def _dense_feature_corpus(
    donor,
    parent,
    replay: QuantizedSequenceReplay,
    *,
    count: int,
    sequence_length: int,
    seed: int,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    candidates = replay._candidate_starts(sequence_length)
    if count > len(candidates):
        raise RuntimeError("dense replay lacks enough unique sequences")
    selected = random.Random(seed).sample(range(len(candidates)), count)
    legal_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    sequence_ids: list[tuple[int, int]] = []
    for candidate_index in selected:
        start, agent = candidates[candidate_index]
        logical = np.arange(start, start + sequence_length)
        physical = replay._physical(logical)
        mask = replay.mask[physical, agent].astype(np.float32) / 255.0
        tail = replay.tail[physical, agent].astype(np.float32)
        legal_rows.append(np.concatenate((mask, tail), -1)[..., :LEGAL_OBS_SIZE])
        action_rows.append(replay.action[physical, agent].astype(np.float32))
        sequence_ids.append((int(start), int(agent)))
    legal = torch.from_numpy(np.stack(legal_rows))
    input_action = torch.from_numpy(np.stack(action_rows))
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    donor.eval()
    parent.eval()
    for offset in range(0, count, microbatch_size):
        chosen_legal = legal[offset : offset + microbatch_size].to(device)
        chosen_action = input_action[offset : offset + microbatch_size].to(device)
        donor_state = _observe_legal_sequence(
            donor,
            chosen_legal,
            chosen_action,
            donor.rssm.initial(len(chosen_legal), device=device),
        )
        parent_state = _observe_legal_sequence(
            parent,
            chosen_legal,
            chosen_action,
            parent.rssm.initial(len(chosen_legal), device=device),
        )
        parent_action = parent.deterministic_actor_action(
            parent.actor_distribution(parent_state.features)
        )
        feature_rows.append(donor_state.features.float().cpu())
        target_rows.append(parent_action.float().cpu())
    return {
        "feature": torch.cat(feature_rows),
        "target_action": torch.cat(target_rows),
        "sequence_id": torch.tensor(sequence_ids, dtype=torch.int64),
    }


def _progress_gates(metrics: dict[str, float | int], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "correlation_at_least_minimum": (
            metrics["correlation"] >= args.minimum_validation_correlation
        ),
        "mae_at_most_maximum": metrics["mae_m"] <= args.maximum_validation_mae_m,
        "near_plane_mae_at_most_maximum": (
            metrics["near_plane_mae_m"] <= args.maximum_validation_near_plane_mae_m
        ),
        "direction_at_least_minimum": (
            metrics["direction_accuracy"] >= args.minimum_validation_direction_accuracy
        ),
    }


def _probe_validation(
    probe: SoftPlaneProbe,
    feature: np.ndarray,
    arrays: dict[str, np.ndarray],
    allowed_index: np.ndarray,
    original_event: np.ndarray,
    crop_start: np.ndarray,
    validation_crops: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    sequence_length: int,
    target_mean: float,
    target_std: float,
    near_plane_m: float,
    minimum_pair_delta_m: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    predictions: list[np.ndarray] = []
    probe.eval()
    with torch.no_grad():
        for offset in range(0, len(validation_crops), batch_size):
            chosen = validation_crops[offset : offset + batch_size]
            batch = _crop_full(
                feature,
                allowed_index[original_event[chosen]],
                crop_start[chosen],
                sequence_length=sequence_length,
            )
            prediction = probe(torch.from_numpy(batch).to(device))
            predictions.append((prediction.cpu().numpy() * target_std + target_mean))
    target = _target_batch(
        arrays,
        original_event[validation_crops],
        crop_start[validation_crops],
        sequence_length=sequence_length,
    )
    return _metrics(
        target,
        np.concatenate(predictions),
        evaluation_indices=evaluation_indices,
        near_plane_m=near_plane_m,
        minimum_pair_delta_m=minimum_pair_delta_m,
    )


def _probe_candidate_key(candidate: dict[str, object], seed: int) -> tuple:
    metrics = candidate["metrics"]
    gates = candidate["gates"]
    assert isinstance(metrics, dict) and isinstance(gates, dict)
    return (
        all(gates.values()),
        *_selection_key(metrics, seed, "frozen_representation"),
        -int(candidate["step"]),
    )


@torch.no_grad()
def _actor_drift_from_numpy(
    model,
    feature: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    model.eval()
    flat_feature = feature.reshape(-1, feature.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    for offset in range(0, len(flat_feature), batch_size):
        prediction = model.deterministic_actor_action(
            model.actor_distribution(
                torch.from_numpy(flat_feature[offset : offset + batch_size]).to(device)
            )
        )
        predictions.append(prediction.float().cpu())
    return _drift(torch.cat(predictions), torch.from_numpy(flat_target))


@torch.no_grad()
def _actor_drift_from_tensor(
    model,
    feature: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    model.eval()
    flat_feature = feature.reshape(-1, feature.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    for offset in range(0, len(flat_feature), batch_size):
        prediction = model.deterministic_actor_action(
            model.actor_distribution(flat_feature[offset : offset + batch_size].to(device))
        )
        predictions.append(prediction.float().cpu())
    return _drift(torch.cat(predictions), flat_target)


def _action_gates(
    successful: dict[str, float], dense: dict[str, float], args: argparse.Namespace
) -> dict[str, bool]:
    return {
        "successful_rmse_bounded": successful["rmse"] <= args.maximum_action_rmse,
        "successful_max_bounded": successful["max_abs"] <= args.maximum_action_max_abs,
        "dense_rmse_bounded": dense["rmse"] <= args.maximum_action_rmse,
        "dense_max_bounded": dense["max_abs"] <= args.maximum_action_max_abs,
    }


def _actor_candidate_key(candidate: dict[str, object], args: argparse.Namespace) -> tuple:
    successful = candidate["successful_drift"]
    dense = candidate["dense_drift"]
    gates = candidate["gates"]
    assert isinstance(successful, dict) and isinstance(dense, dict) and isinstance(gates, dict)
    normalized = (
        successful["rmse"] / args.maximum_action_rmse,
        successful["max_abs"] / args.maximum_action_max_abs,
        dense["rmse"] / args.maximum_action_rmse,
        dense["max_abs"] / args.maximum_action_max_abs,
    )
    return (all(gates.values()), -max(normalized), -sum(normalized), -int(candidate["step"]))


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    if args.output_checkpoint.resolve() == args.donor_checkpoint.resolve():
        raise ValueError("calibration cannot overwrite its donor")
    if args.output_checkpoint.exists() or args.report.exists():
        raise ValueError("calibration refuses to overwrite outputs")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "donor_checkpoint": _sha256(args.donor_checkpoint),
        "donor_report": _sha256(args.donor_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    donor_report = json.loads(args.donor_report.read_text(encoding="utf-8"))
    if (
        donor_report.get("contract") != JOINT_SCHEMA
        or not donor_report.get("process_passed")
        or not donor_report.get("validation_admitted")
        or donor_report.get("selected_step") != 350
        or donor_report.get("output_checkpoint_sha256") != source_before["donor_checkpoint"]
    ):
        raise RuntimeError("N674 donor contract mismatch")
    donor_payload = torch.load(args.donor_checkpoint, map_location=device, weights_only=False)
    donor = _model_from_payload(donor_payload, device)
    parent_payload = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    parent = _model_from_payload(parent_payload, device)
    donor_auxiliary = donor_payload.get("success_prefix_joint_representation_aux")
    if (
        not isinstance(donor_auxiliary, dict)
        or donor_auxiliary.get("schema") != JOINT_SCHEMA
        or donor_auxiliary.get("selected_step") != 350
    ):
        raise RuntimeError("N674 donor auxiliary mismatch")
    donor_model_before = {
        name: value.detach().clone() for name, value in donor.state_dict().items()
    }
    probe = SoftPlaneProbe(donor.rssm.feature_size, args.probe_hidden_size).to(device)
    probe.load_state_dict(donor_auxiliary["probe_state"])
    target_mean = float(donor_auxiliary["target_mean"])
    target_std = float(donor_auxiliary["target_std"])

    with np.load(args.dataset) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    phase_after = np.rint(arrays["phase_after_event"].astype(np.float64) * 6.0).astype(np.int64)
    phase_before = np.rint(
        arrays["tail"][:, :-1, legal_tail_size + 32].astype(np.float64) * 6.0
    ).astype(np.int64)
    if not np.all(phase_before == 0) or not np.all(phase_after == 1):
        raise RuntimeError("N668 phase contract mismatch")
    event_split = _stratified_group_split(
        arrays["vector_step"].astype(np.int64),
        phase_after,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    train_events = np.flatnonzero(event_split == 0)
    validation_events = np.flatnonzero(event_split == 1)
    allowed_events = np.concatenate((train_events, validation_events))
    allowed_events.sort()
    allowed_index = np.full(len(event_split), -1, dtype=np.int64)
    allowed_index[allowed_events] = np.arange(len(allowed_events))
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    original_event, crop_start = _crop_index(len(event_split), crop_starts)
    crop_split = event_split[original_event]
    train_crops = np.flatnonzero(crop_split == 0)
    validation_crops = np.flatnonzero(crop_split == 1)
    evaluation_indices = np.arange(
        args.burn_in - 1, args.sequence_length, args.evaluation_stride, dtype=np.int64
    )
    event_features = _event_feature_corpus(
        donor,
        arrays,
        allowed_events,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    donor_probe_parity = _probe_validation(
        probe,
        event_features["soft"],
        arrays,
        allowed_index,
        original_event,
        crop_start,
        validation_crops,
        evaluation_indices,
        sequence_length=args.sequence_length,
        target_mean=target_mean,
        target_std=target_std,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    donor_probe_parity_error = max(
        abs(float(donor_probe_parity[key]) - float(donor_report["selected_validation_progress"][key]))
        for key in donor_probe_parity
        if isinstance(donor_probe_parity[key], (int, float))
    )

    replay_capacity, replay_agents = _replay_geometry(args.dense_replay_dir)
    replay = QuantizedSequenceReplay(
        replay_capacity, replay_agents, storage_dir=args.dense_replay_dir, resume=True
    )
    dense = _dense_feature_corpus(
        donor,
        parent,
        replay,
        count=args.dense_train_count + args.dense_validation_count,
        sequence_length=args.dense_sequence_length,
        seed=args.dense_seed,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    dense_train = slice(0, args.dense_train_count)
    dense_validation = slice(args.dense_train_count, None)
    dense_train_ids = {
        tuple(row) for row in dense["sequence_id"][dense_train].tolist()
    }
    dense_validation_ids = {
        tuple(row) for row in dense["sequence_id"][dense_validation].tolist()
    }
    dense_overlap = dense_train_ids & dense_validation_ids

    # Probe calibration: representation and model remain detached/frozen.
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay
    )
    rng = np.random.default_rng(args.seed)
    initial_probe_candidate: dict[str, object] = {
        "step": 0,
        "metrics": donor_probe_parity,
        "gates": _progress_gates(donor_probe_parity, args),
    }
    probe_history: list[dict[str, object]] = [initial_probe_candidate]
    best_probe = copy.deepcopy(initial_probe_candidate)
    best_probe_state = {
        name: value.detach().cpu().clone() for name, value in probe.state_dict().items()
    }
    max_probe_gradient = 0.0
    for step in range(1, args.probe_steps + 1):
        chosen = rng.choice(train_crops, args.probe_batch_size, replace=True)
        feature = _crop_full(
            event_features["soft"],
            allowed_index[original_event[chosen]],
            crop_start[chosen],
            sequence_length=args.sequence_length,
        )
        target = _target_batch(
            arrays,
            original_event[chosen],
            crop_start[chosen],
            sequence_length=args.sequence_length,
        )
        feature_tensor = torch.from_numpy(feature).to(device)
        target_tensor = torch.from_numpy((target - target_mean) / target_std).to(device)
        prediction = probe(feature_tensor)[:, args.burn_in :]
        supervised = target_tensor[:, args.burn_in :]
        loss = F.mse_loss(prediction, supervised) + args.delta_weight * F.mse_loss(
            torch.diff(prediction, dim=1), torch.diff(supervised, dim=1)
        )
        probe_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = _finite_clip_grad_norm(
            list(probe.parameters()), args.probe_gradient_clip, label="frozen probe calibration"
        )
        max_probe_gradient = max(max_probe_gradient, float(norm.cpu()))
        probe_optimizer.step()
        if step % args.validation_interval and step != args.probe_steps:
            continue
        metrics = _probe_validation(
            probe,
            event_features["soft"],
            arrays,
            allowed_index,
            original_event,
            crop_start,
            validation_crops,
            evaluation_indices,
            sequence_length=args.sequence_length,
            target_mean=target_mean,
            target_std=target_std,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        candidate: dict[str, object] = {
            "step": step,
            "metrics": metrics,
            "gates": _progress_gates(metrics, args),
        }
        probe_history.append(candidate)
        if _probe_candidate_key(candidate, args.seed) > _probe_candidate_key(best_probe, args.seed):
            best_probe = copy.deepcopy(candidate)
            best_probe_state = {
                name: value.detach().cpu().clone() for name, value in probe.state_dict().items()
            }
    probe.load_state_dict(best_probe_state)

    # Actor calibration: only actor tensors require gradients.
    for parameter in donor.parameters():
        parameter.requires_grad_(False)
    for parameter in donor.actor.parameters():
        parameter.requires_grad_(True)
    actor_optimizer = torch.optim.AdamW(
        donor.actor.parameters(), lr=args.actor_learning_rate, weight_decay=args.actor_weight_decay
    )
    success_train_feature = event_features["hard"][allowed_index[train_events]].reshape(
        -1, event_features["hard"].shape[-1]
    )
    success_train_target = arrays["action"][train_events, :-1].reshape(-1, 4).astype(np.float32)
    success_validation_feature = event_features["hard"][allowed_index[validation_events]]
    success_validation_target = arrays["action"][validation_events, :-1].astype(np.float32)
    initial_success_drift = _actor_drift_from_numpy(
        donor,
        success_validation_feature,
        success_validation_target,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    initial_dense_drift = _actor_drift_from_tensor(
        donor,
        dense["feature"][dense_validation],
        dense["target_action"][dense_validation],
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    initial_actor_candidate: dict[str, object] = {
        "step": 0,
        "successful_drift": initial_success_drift,
        "dense_drift": initial_dense_drift,
        "gates": _action_gates(initial_success_drift, initial_dense_drift, args),
    }
    actor_history: list[dict[str, object]] = [initial_actor_candidate]
    best_actor = copy.deepcopy(initial_actor_candidate)
    best_actor_state = {
        name: value.detach().cpu().clone() for name, value in donor.actor.state_dict().items()
    }
    max_actor_gradient = 0.0
    dense_train_feature = dense["feature"][dense_train].reshape(-1, dense["feature"].shape[-1])
    dense_train_target = dense["target_action"][dense_train].reshape(-1, 4)
    for step in range(1, args.actor_steps + 1):
        successful_index = rng.integers(
            0, len(success_train_feature), size=args.actor_success_batch_size
        )
        dense_index = rng.integers(0, len(dense_train_feature), size=args.actor_dense_batch_size)
        successful_feature = torch.from_numpy(success_train_feature[successful_index]).to(device)
        successful_target = torch.from_numpy(success_train_target[successful_index]).to(device)
        selected_dense_feature = dense_train_feature[dense_index].to(device)
        selected_dense_target = dense_train_target[dense_index].to(device)
        successful_prediction = donor.deterministic_actor_action(
            donor.actor_distribution(successful_feature)
        )
        dense_prediction = donor.deterministic_actor_action(
            donor.actor_distribution(selected_dense_feature)
        )
        actor_loss = F.mse_loss(successful_prediction, successful_target) + F.mse_loss(
            dense_prediction, selected_dense_target
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        norm = _finite_clip_grad_norm(
            list(donor.actor.parameters()), args.actor_gradient_clip, label="frozen actor calibration"
        )
        max_actor_gradient = max(max_actor_gradient, float(norm.cpu()))
        actor_optimizer.step()
        if step % args.validation_interval and step != args.actor_steps:
            continue
        successful_drift = _actor_drift_from_numpy(
            donor,
            success_validation_feature,
            success_validation_target,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        dense_drift = _actor_drift_from_tensor(
            donor,
            dense["feature"][dense_validation],
            dense["target_action"][dense_validation],
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        candidate = {
            "step": step,
            "successful_drift": successful_drift,
            "dense_drift": dense_drift,
            "gates": _action_gates(successful_drift, dense_drift, args),
        }
        actor_history.append(candidate)
        if _actor_candidate_key(candidate, args) > _actor_candidate_key(best_actor, args):
            best_actor = copy.deepcopy(candidate)
            best_actor_state = {
                name: value.detach().cpu().clone() for name, value in donor.actor.state_dict().items()
            }
    donor.actor.load_state_dict(best_actor_state)
    donor.eval()
    probe.eval()

    changed = [
        name for name, value in donor.state_dict().items()
        if not torch.equal(value, donor_model_before[name])
    ]
    forbidden = [name for name in changed if not name.startswith("actor.")]
    representation_exact = all(
        torch.equal(value, donor_model_before[name])
        for name, value in donor.state_dict().items()
        if name.startswith("rssm.")
    )
    progress_gates = best_probe["gates"]
    action_gates = best_actor["gates"]
    assert isinstance(progress_gates, dict) and isinstance(action_gates, dict)
    source_after: dict[str, object] = {
        "executable": _sha256(Path(__file__)),
        "donor_checkpoint": _sha256(args.donor_checkpoint),
        "donor_report": _sha256(args.donor_report),
        "parent_checkpoint": _sha256(args.parent_checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "donor_probe_parity": donor_probe_parity_error <= args.donor_parity_tolerance,
        "model_changed": bool(changed),
        "only_actor_model_tensors_changed": not forbidden,
        "representation_exact": representation_exact,
        "dense_train_validation_disjoint": not dense_overlap,
        "probe_finite": all(
            math.isfinite(float(row["metrics"][key]))
            for row in probe_history
            for key in ("correlation", "mae_m", "near_plane_mae_m", "direction_accuracy")
        ),
        "test_labels_not_evaluated": True,
    }
    cuda_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    resource_gate = device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    validation_admitted = (
        all(process_gates.values())
        and all(progress_gates.values())
        and all(action_gates.values())
        and resource_gate
    )
    output_payload = dict(donor_payload)
    output_payload["model"] = donor.state_dict()
    output_payload["frozen_representation_calibration_aux"] = {
        "schema": CALIBRATION_SCHEMA,
        "probe_state": {
            name: value.detach().cpu() for name, value in probe.state_dict().items()
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "selected_probe_step": int(best_probe["step"]),
        "selected_actor_step": int(best_actor["step"]),
        "representation_updates": 0,
        "split_seed": args.split_seed,
    }
    _save_checkpoint(args.output_checkpoint, output_payload)
    report: dict[str, object] = {
        "contract": CALIBRATION_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "events": int(len(event_split)),
        "train_events": int(len(train_events)),
        "validation_events": int(len(validation_events)),
        "test_events": int((event_split == 2).sum()),
        "test_evaluations": 0,
        "test_observations_evaluated": 0,
        "test_privileged_labels_evaluated": 0,
        "dense_train_sequences": args.dense_train_count,
        "dense_validation_sequences": args.dense_validation_count,
        "dense_train_validation_overlap": len(dense_overlap),
        "donor_probe_parity": donor_probe_parity,
        "donor_probe_parity_max_abs_error": donor_probe_parity_error,
        "probe_history": probe_history,
        "selected_probe_step": int(best_probe["step"]),
        "selected_validation_progress": best_probe["metrics"],
        "progress_gates": progress_gates,
        "actor_history": actor_history,
        "selected_actor_step": int(best_actor["step"]),
        "selected_validation_action_drift": best_actor["successful_drift"],
        "selected_dense_action_drift": best_actor["dense_drift"],
        "action_gates": action_gates,
        "maximum_probe_gradient_norm": max_probe_gradient,
        "maximum_actor_gradient_norm": max_actor_gradient,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "validation_admitted": validation_admitted,
        "resource_gate": resource_gate,
        "probe_updates": args.probe_steps,
        "actor_updates": args.actor_steps,
        "representation_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
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
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-checkpoint", type=Path, required=True)
    parser.add_argument("--donor-report", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
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
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-learning-rate", type=float, default=0.0003)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0001)
    parser.add_argument("--probe-gradient-clip", type=float, default=10.0)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--actor-steps", type=int, default=1000)
    parser.add_argument("--actor-success-batch-size", type=int, default=512)
    parser.add_argument("--actor-dense-batch-size", type=int, default=512)
    parser.add_argument("--actor-learning-rate", type=float, default=0.0001)
    parser.add_argument("--actor-weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-gradient-clip", type=float, default=10.0)
    parser.add_argument("--dense-train-count", type=int, default=256)
    parser.add_argument("--dense-validation-count", type=int, default=128)
    parser.add_argument("--dense-sequence-length", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=675001)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--evaluation-batch-size", type=int, default=4096)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--minimum-validation-correlation", type=float, default=0.8)
    parser.add_argument("--maximum-validation-mae-m", type=float, default=0.5)
    parser.add_argument("--maximum-validation-near-plane-mae-m", type=float, default=0.5)
    parser.add_argument("--minimum-validation-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--maximum-action-rmse", type=float, default=0.001)
    parser.add_argument("--maximum-action-max-abs", type=float, default=0.01)
    parser.add_argument("--donor-parity-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=675)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "probe_hidden_size",
        "probe_steps", "probe_batch_size", "actor_steps", "actor_success_batch_size",
        "actor_dense_batch_size", "dense_train_count", "dense_validation_count",
        "dense_sequence_length", "validation_interval", "evaluation_batch_size",
        "reference_microbatch_size", "maximum_cuda_bytes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    return args


if __name__ == "__main__":
    print(json.dumps(calibrate(parse_args()), indent=2, sort_keys=True))
