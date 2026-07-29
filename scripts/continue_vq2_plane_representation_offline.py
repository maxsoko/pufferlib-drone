#!/usr/bin/env python3
"""Train one legal posterior plane-progress auxiliary step offline.

Only the visual encoder, recurrent sequence cell, posterior network, and a new
training-only linear plane probe can change.  The prior, deployed actor,
privileged decoder, task reward/continuation, critic, and target critic remain
bit exact.  N617 validation/test groups are never optimized.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _autocast_context,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


CONTINUATION_SCHEMA = "vq2_dense_anchored_plane_representation_v1"
ALLOWED_MODEL_PREFIXES = (
    "rssm.encoder.",
    "rssm.sequence.",
    "rssm.posterior.",
)


def _rolling_samples(
    observation: torch.Tensor,
    action: torch.Tensor,
    event_indices: np.ndarray,
    *,
    targets: tuple[int, ...],
    context: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = torch.as_tensor(event_indices, device=observation.device, dtype=torch.long)
    observation_rows = torch.stack(
        [observation[selected, target - context : target] for target in targets], 1
    )
    action_rows = torch.stack(
        [action[selected, target - context : target] for target in targets], 1
    )
    target_rows = torch.stack(
        [observation[selected, target - 1, LEGAL_OBS_SIZE + 3] for target in targets],
        1,
    )
    return (
        observation_rows.flatten(0, 1),
        action_rows.flatten(0, 1),
        target_rows.flatten(0, 1),
    )


def _last_state(output):
    return (
        output.posterior.deterministic[:, -1],
        output.posterior.stochastic[:, -1],
        output.posterior.logits[:, -1],
        output.posterior.features[:, -1],
    )


@torch.no_grad()
def _reference(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    microbatch_size: int,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    values = {name: [] for name in ("deterministic", "logits", "action")}
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        with _autocast_context(observation.device, amp_dtype):
            output = model.observe_sequence(
                observation[start:stop],
                action[start:stop],
                deterministic_latent=True,
            )
            deterministic, _stochastic, logits, features = _last_state(output)
            actor_action = model.deterministic_actor_action(
                model.actor_distribution(features)
            )
        values["deterministic"].append(deterministic.float())
        values["logits"].append(logits.float())
        values["action"].append(actor_action.float())
    return {name: torch.cat(rows, 0) for name, rows in values.items()}


@torch.no_grad()
def _posterior_features(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    microbatch_size: int,
    amp_dtype: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        with _autocast_context(observation.device, amp_dtype):
            output = model.observe_sequence(
                observation[start:stop],
                action[start:stop],
                deterministic_latent=True,
            )
        rows.append(output.posterior.features[:, -1].float())
    return torch.cat(rows, 0)


def _initialize_probe_ridge(
    probe: torch.nn.Linear,
    features: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, float | int]:
    if ridge <= 0.0:
        raise ValueError("probe ridge must be positive")
    feature = features.detach().double().cpu().numpy()
    target = normalized_target.detach().double().cpu().numpy()
    mean = feature.mean(0)
    std = feature.std(0)
    active = std > 1e-6
    standardized = (feature[:, active] - mean[active]) / std[active]
    design = np.concatenate((standardized, np.ones((feature.shape[0], 1))), 1)
    gram = design.T @ design / feature.shape[0]
    rhs = design.T @ target / feature.shape[0]
    regularizer = np.eye(gram.shape[0]) * ridge
    regularizer[-1, -1] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, rhs)
    raw_weight = np.zeros(feature.shape[1], dtype=np.float64)
    raw_weight[active] = coefficient[:-1] / std[active]
    raw_bias = coefficient[-1] - np.sum(coefficient[:-1] * mean[active] / std[active])
    with torch.no_grad():
        probe.weight.copy_(
            torch.from_numpy(raw_weight).to(probe.weight).reshape_as(probe.weight)
        )
        probe.bias.fill_(float(raw_bias))
    return {
        "ridge": ridge,
        "active_features": int(active.sum()),
        "weight_abs_max": float(np.abs(raw_weight).max()),
        "bias": float(raw_bias),
    }


def _anchor_loss(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    reference: dict[str, torch.Tensor],
    *,
    start: int,
    stop: int,
    amp_dtype: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with _autocast_context(observation.device, amp_dtype):
        output = model.observe_sequence(
            observation[start:stop],
            action[start:stop],
            deterministic_latent=True,
        )
        deterministic, _stochastic, logits, features = _last_state(output)
        actor_action = model.deterministic_actor_action(
            model.actor_distribution(features)
        )
        deterministic_loss = F.mse_loss(
            deterministic.float(), reference["deterministic"][start:stop]
        )
        logits_loss = F.mse_loss(logits.float(), reference["logits"][start:stop])
        action_loss = F.mse_loss(
            actor_action.float(), reference["action"][start:stop]
        )
    return deterministic_loss + logits_loss, {
        "deterministic": deterministic_loss,
        "logits": logits_loss,
        "action": action_loss,
    }


@torch.no_grad()
def _probe_predictions(
    model: VQ2InformedDreamer,
    probe: torch.nn.Linear,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    microbatch_size: int,
    amp_dtype: str,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    rows: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        with _autocast_context(observation.device, amp_dtype):
            output = model.observe_sequence(
                observation[start:stop],
                action[start:stop],
                deterministic_latent=True,
            )
            prediction = probe(output.posterior.features[:, -1]).squeeze(-1)
        rows.append((prediction.float() * target_std + target_mean).cpu())
    return torch.cat(rows).numpy()


def _progress_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    step = float(np.abs(np.diff(target, axis=1)).mean())
    return {
        "correlation": _correlation(target, prediction),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_true_step": step,
        "mae_in_true_step_units": float(np.abs(error).mean()) / step,
        "strictly_decreasing_window_fraction": float(
            (np.diff(prediction, axis=1) < 0.0).all(1).mean()
        ),
        "final_is_minimum_window_fraction": float(
            (prediction[:, -1] < prediction[:, :-1].min(1)).mean()
        ),
    }


def _changed_model_keys(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    changed = [name for name, value in after.items() if not torch.equal(value, before[name])]
    forbidden = [
        name for name in changed if not name.startswith(ALLOWED_MODEL_PREFIXES)
    ]
    return changed, forbidden


def continue_representation(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("representation continuation cannot overwrite its parent")
    if args.updates != 1:
        raise ValueError("integration executable is locked to one optimizer update")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha_before = _sha256(args.checkpoint)
    event_sha_before = _sha256(args.event_dataset)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    replay_state = payload.get("replay")
    if not isinstance(training_args, dict) or not isinstance(replay_state, dict):
        raise RuntimeError("checkpoint lacks training/replay state")
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(
            training_args.get("actor_distribution_mode", "legacy_tanh_normal")
        ),
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.train()
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_parameters = [
        *model.rssm.encoder.parameters(),
        *model.rssm.sequence.parameters(),
        *model.rssm.posterior.parameters(),
    ]
    for parameter in train_parameters:
        parameter.requires_grad_(True)
    probe = torch.nn.Linear(model.rssm.feature_size, 1).to(device)
    torch.nn.init.zeros_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    if args.resume_probe_from_checkpoint and args.initialize_probe_ridge:
        raise ValueError("probe resume and ridge initialization are mutually exclusive")
    resumed_probe_updates = 0
    if args.resume_probe_from_checkpoint:
        auxiliary = payload.get("plane_representation_aux")
        if not isinstance(auxiliary, dict) or auxiliary.get("schema") != CONTINUATION_SCHEMA:
            raise RuntimeError("parent lacks a compatible plane-probe auxiliary")
        probe_state = auxiliary.get("probe_state")
        if not isinstance(probe_state, dict):
            raise RuntimeError("parent plane-probe state is missing")
        probe.load_state_dict(probe_state)
        resumed_probe_updates = int(auxiliary.get("updates", 0))
        if resumed_probe_updates <= 0:
            raise RuntimeError("parent plane-probe update count is invalid")
    optimizer_parameters = [*train_parameters, *probe.parameters()]
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate, weight_decay=0.0
    )

    dataset_context = 16
    dataset = _load_event_dataset(
        args.event_dataset,
        context_length=dataset_context,
        event_threshold=args.event_threshold,
    )
    full_observation, full_action, _reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    event_split = _stratified_group_split(
        groups,
        phases,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    negative_offsets = tuple(
        sorted((int(value) for value in args.negative_offsets.split(",")), reverse=True)
    )
    targets = tuple(dataset_context - value for value in negative_offsets) + (
        dataset_context,
    )
    train_events = np.flatnonzero(event_split == 0)
    validation_events = np.flatnonzero(event_split == 1)
    event_observation, event_action, event_target = _rolling_samples(
        full_observation,
        full_action,
        train_events,
        targets=targets,
        context=args.feature_context,
    )
    validation_observation, validation_action, validation_target = _rolling_samples(
        full_observation,
        full_action,
        validation_events,
        targets=targets,
        context=args.feature_context,
    )
    target_mean = float(event_target.mean().cpu())
    target_std = float(event_target.std(unbiased=False).cpu())
    if target_std <= 0.0:
        raise RuntimeError("plane target standard deviation is zero")
    if args.resume_probe_from_checkpoint:
        auxiliary = payload["plane_representation_aux"]
        if abs(float(auxiliary["target_mean"]) - target_mean) > 1e-12:
            raise RuntimeError("resumed plane-probe target mean mismatch")
        if abs(float(auxiliary["target_std"]) - target_std) > 1e-12:
            raise RuntimeError("resumed plane-probe target std mismatch")
    ridge_initialization = None
    if args.initialize_probe_ridge:
        model.eval()
        ridge_features = _posterior_features(
            model,
            event_observation,
            event_action,
            microbatch_size=args.microbatch_size,
            amp_dtype=args.amp_dtype,
        )
        ridge_initialization = _initialize_probe_ridge(
            probe,
            ridge_features,
            (event_target - target_mean) / target_std,
            ridge=args.probe_ridge,
        )
    initial_probe_weight_abs_max = float(probe.weight.detach().abs().max().cpu())
    initial_probe_bias = float(probe.bias.detach().cpu())

    pre_validation_prediction = _probe_predictions(
        model,
        probe,
        validation_observation,
        validation_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
        target_mean=target_mean,
        target_std=target_std,
    ).reshape(len(validation_events), len(targets))
    validation_target_np = validation_target.float().cpu().numpy().reshape(
        len(validation_events), len(targets)
    )
    pre_validation_metrics = _progress_metrics(
        validation_target_np, pre_validation_prediction
    )

    replay = QuantizedSequenceReplay(
        int(replay_state["capacity"]),
        int(replay_state["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    observed_replay = replay.state_dict()
    for key in ("schema", "capacity", "agents", "position", "size"):
        if observed_replay[key] != replay_state[key]:
            raise RuntimeError(f"source-locked replay {key} mismatch")
    dense_observation, dense_action, _dense_reward, _dense_continuation = replay.sample(
        args.dense_batch_size,
        args.feature_context,
        device=device,
        rng=random.Random(args.dense_seed),
    )
    model.eval()
    event_reference = _reference(
        model,
        event_observation,
        event_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    dense_reference = _reference(
        model,
        dense_observation,
        dense_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    aggregate = {
        "plane": 0.0,
        "event_action_anchor": 0.0,
        "dense_feature_anchor": 0.0,
        "dense_action_anchor": 0.0,
    }
    microbatches = 0
    for start in range(0, event_observation.shape[0], args.microbatch_size):
        stop = min(start + args.microbatch_size, event_observation.shape[0])
        weight = (stop - start) / event_observation.shape[0]
        with _autocast_context(device, args.amp_dtype):
            stochastic_output = model.observe_sequence(
                event_observation[start:stop],
                event_action[start:stop],
                deterministic_latent=False,
            )
            normalized_prediction = probe(
                stochastic_output.posterior.features[:, -1]
            ).squeeze(-1)
            normalized_target = (
                event_target[start:stop] - target_mean
            ) / target_std
            plane_loss = F.mse_loss(
                normalized_prediction.float(), normalized_target.float()
            )
        _feature_anchor, event_anchor = _anchor_loss(
            model,
            event_observation,
            event_action,
            event_reference,
            start=start,
            stop=stop,
            amp_dtype=args.amp_dtype,
        )
        loss = plane_loss + args.event_action_anchor_weight * event_anchor["action"]
        (loss * weight).backward()
        aggregate["plane"] += float(plane_loss.detach().cpu()) * weight
        aggregate["event_action_anchor"] += (
            float(event_anchor["action"].detach().cpu()) * weight
        )
        microbatches += 1
    for start in range(0, dense_observation.shape[0], args.microbatch_size):
        stop = min(start + args.microbatch_size, dense_observation.shape[0])
        weight = (stop - start) / dense_observation.shape[0]
        feature_anchor, dense_anchor = _anchor_loss(
            model,
            dense_observation,
            dense_action,
            dense_reference,
            start=start,
            stop=stop,
            amp_dtype=args.amp_dtype,
        )
        loss = args.dense_feature_anchor_weight * feature_anchor + (
            args.dense_action_anchor_weight * dense_anchor["action"]
        )
        (loss * weight).backward()
        aggregate["dense_feature_anchor"] += (
            float(feature_anchor.detach().cpu()) * weight
        )
        aggregate["dense_action_anchor"] += (
            float(dense_anchor["action"].detach().cpu()) * weight
        )
        microbatches += 1
    gradient_norm = _finite_clip_grad_norm(
        optimizer_parameters, args.max_gradient_norm, label="plane-representation"
    )
    optimizer.step()
    preprojection_max_parameter_delta = 0.0
    projected_parameter_values = 0
    if args.max_parameter_delta > 0.0:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                parent = parent_state[name]
                delta = parameter - parent
                preprojection_max_parameter_delta = max(
                    preprojection_max_parameter_delta,
                    float(delta.abs().max().cpu()),
                )
                clipped = delta.clamp(
                    -args.max_parameter_delta, args.max_parameter_delta
                )
                projected_parameter_values += int((clipped != delta).sum().cpu())
                parameter.copy_(parent + clipped)
    model.eval()

    validation_prediction = _probe_predictions(
        model,
        probe,
        validation_observation,
        validation_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
        target_mean=target_mean,
        target_std=target_std,
    ).reshape(len(validation_events), len(targets))
    validation_metrics = _progress_metrics(
        validation_target_np, validation_prediction
    )
    post_dense = _reference(
        model,
        dense_observation,
        dense_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    dense_drift = {
        name: {
            "rmse": float(
                (post_dense[name] - dense_reference[name]).square().mean().sqrt().cpu()
            ),
            "max_abs": float(
                (post_dense[name] - dense_reference[name]).abs().max().cpu()
            ),
        }
        for name in dense_reference
    }
    changed, forbidden = _changed_model_keys(parent_state, model.state_dict())
    output_payload = dict(payload)
    output_payload["model"] = model.state_dict()
    output_payload["plane_representation_aux"] = {
        "schema": CONTINUATION_SCHEMA,
        "probe_state": probe.state_dict(),
        "target_mean": target_mean,
        "target_std": target_std,
        "targets": targets,
        "feature_context": args.feature_context,
        "split_seed": args.split_seed,
        "updates": resumed_probe_updates + args.updates,
    }
    _save_checkpoint(args.output_checkpoint, output_payload)
    checkpoint_sha_after = _sha256(args.checkpoint)
    event_sha_after = _sha256(args.event_dataset)
    replay_hashes_after = _replay_hashes(args.replay_dir)
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": checkpoint_sha_after,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "event_dataset_sha256": event_sha_after,
        "replay_hashes": replay_hashes_after,
        "sources_unchanged": (
            checkpoint_sha_after == checkpoint_sha_before
            and event_sha_after == event_sha_before
            and replay_hashes_after == replay_hashes_before
        ),
        "train_events": int(len(train_events)),
        "validation_events": int(len(validation_events)),
        "test_events": int((event_split == 2).sum()),
        "test_evaluations": 0,
        "targets": list(targets),
        "feature_context": args.feature_context,
        "target_mean": target_mean,
        "target_std": target_std,
        "updates": args.updates,
        "resumed_probe_updates": resumed_probe_updates,
        "cumulative_probe_updates": resumed_probe_updates + args.updates,
        "resume_probe_from_checkpoint": int(args.resume_probe_from_checkpoint),
        "initialize_probe_ridge": int(args.initialize_probe_ridge),
        "probe_ridge": args.probe_ridge,
        "ridge_initialization": ridge_initialization,
        "max_parameter_delta": args.max_parameter_delta,
        "preprojection_max_parameter_delta": preprojection_max_parameter_delta,
        "projected_parameter_values": projected_parameter_values,
        "learning_rate": args.learning_rate,
        "event_action_anchor_weight": args.event_action_anchor_weight,
        "dense_feature_anchor_weight": args.dense_feature_anchor_weight,
        "dense_action_anchor_weight": args.dense_action_anchor_weight,
        "losses": aggregate,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "microbatches": microbatches,
        "validation_progress": validation_metrics,
        "preupdate_validation_progress": pre_validation_metrics,
        "dense_drift": dense_drift,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "probe_weight_abs_max": float(probe.weight.detach().abs().max().cpu()),
        "probe_bias": float(probe.bias.detach().cpu()),
        "initial_probe_weight_abs_max": initial_probe_weight_abs_max,
        "initial_probe_bias": initial_probe_bias,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "posterior_representation_updates": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
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
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--resume-probe-from-checkpoint", action="store_true")
    parser.add_argument("--initialize-probe-ridge", action="store_true")
    parser.add_argument("--probe-ridge", type=float, default=0.1)
    parser.add_argument("--max-parameter-delta", type=float, default=0.0)
    parser.add_argument("--feature-context", type=int, default=12)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=628)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=633001)
    parser.add_argument("--seed", type=int, default=633)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--event-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--dense-feature-anchor-weight", type=float, default=1.0)
    parser.add_argument("--dense-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--max-gradient-norm", type=float, default=100.0)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(continue_representation(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
