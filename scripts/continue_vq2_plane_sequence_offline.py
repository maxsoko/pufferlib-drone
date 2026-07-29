#!/usr/bin/env python3
"""One deterministic sequence-only near-plane representation update."""

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
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.continue_vq2_plane_representation_offline import (
    _progress_metrics,
    _reference,
    _rolling_samples,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _autocast_context,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


CONTINUATION_SCHEMA = "vq2_deterministic_sequence_plane_representation_v1"


def _fit_standardized_probe(
    feature: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, torch.Tensor | float | int]:
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    x = feature.detach().double().cpu().numpy()
    y = normalized_target.detach().double().cpu().numpy()
    mean = x.mean(0)
    std = x.std(0)
    active = std > 1e-6
    z = (x[:, active] - mean[active]) / std[active]
    design = np.concatenate((z, np.ones((z.shape[0], 1))), 1)
    gram = design.T @ design / z.shape[0]
    rhs = design.T @ y / z.shape[0]
    regularizer = np.eye(gram.shape[0]) * ridge
    regularizer[-1, -1] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, rhs)
    device = feature.device
    return {
        "mean": torch.from_numpy(mean[active]).float().to(device),
        "std": torch.from_numpy(std[active]).float().to(device),
        "indices": torch.from_numpy(np.flatnonzero(active)).long().to(device),
        "weight": torch.from_numpy(coefficient[:-1]).float().to(device),
        "bias": float(coefficient[-1]),
        "ridge": ridge,
        "active_features": int(active.sum()),
    }


def _apply_standardized_probe(
    feature: torch.Tensor, probe: dict[str, torch.Tensor | float | int]
) -> torch.Tensor:
    indices = probe["indices"]
    mean = probe["mean"]
    std = probe["std"]
    weight = probe["weight"]
    assert isinstance(indices, torch.Tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert isinstance(weight, torch.Tensor)
    selected = feature.index_select(-1, indices)
    return ((selected.float() - mean) / std).matmul(weight) + float(probe["bias"])


@torch.no_grad()
def _deterministic_features(
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
        rows.append(output.posterior.deterministic[:, -1].float())
    return torch.cat(rows, 0)


@torch.no_grad()
def _predictions(
    model: VQ2InformedDreamer,
    probe: dict[str, torch.Tensor | float | int],
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    microbatch_size: int,
    amp_dtype: str,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    feature = _deterministic_features(
        model,
        observation,
        action,
        microbatch_size=microbatch_size,
        amp_dtype=amp_dtype,
    )
    return (
        _apply_standardized_probe(feature, probe) * target_std + target_mean
    ).cpu().numpy()


def _allowed_changes(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    changed = [name for name, value in after.items() if not torch.equal(value, before[name])]
    forbidden = [name for name in changed if not name.startswith("rssm.sequence.")]
    return changed, forbidden


def continue_sequence(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("sequence continuation cannot overwrite its parent")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "replay": _replay_hashes(args.replay_dir),
    }
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
    model.eval()
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.rssm.sequence.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)

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
    split = _stratified_group_split(
        groups,
        phases,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    targets = tuple(
        dataset_context - int(value)
        for value in sorted(args.negative_offsets_values, reverse=True)
    ) + (dataset_context,)
    train_events = np.flatnonzero(split == 0)
    validation_events = np.flatnonzero(split == 1)
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
    parent_event_feature = _deterministic_features(
        model,
        event_observation,
        event_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    probe = _fit_standardized_probe(
        parent_event_feature,
        (event_target - target_mean) / target_std,
        ridge=args.probe_ridge,
    )
    pre_validation = _predictions(
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
    pre_validation_metrics = _progress_metrics(validation_target_np, pre_validation)

    replay = QuantizedSequenceReplay(
        int(replay_state["capacity"]),
        int(replay_state["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    observed = replay.state_dict()
    for key in ("schema", "capacity", "agents", "position", "size"):
        if observed[key] != replay_state[key]:
            raise RuntimeError(f"source-locked replay {key} mismatch")
    dense_observation, dense_action, _dense_reward, _dense_continuation = replay.sample(
        args.dense_batch_size,
        args.feature_context,
        device=device,
        rng=random.Random(args.dense_seed),
    )
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
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    losses = {"plane": 0.0, "event_action_anchor": 0.0, "dense_anchor": 0.0}
    microbatches = 0
    for start in range(0, event_observation.shape[0], args.microbatch_size):
        stop = min(start + args.microbatch_size, event_observation.shape[0])
        weight = (stop - start) / event_observation.shape[0]
        with _autocast_context(device, args.amp_dtype):
            output = model.observe_sequence(
                event_observation[start:stop],
                event_action[start:stop],
                deterministic_latent=True,
            )
            deterministic = output.posterior.deterministic[:, -1]
            prediction = _apply_standardized_probe(deterministic, probe)
            target = (event_target[start:stop] - target_mean) / target_std
            plane_loss = F.mse_loss(prediction.float(), target.float())
            actor_action = model.deterministic_actor_action(
                model.actor_distribution(output.posterior.features[:, -1])
            )
            action_loss = F.mse_loss(
                actor_action.float(), event_reference["action"][start:stop]
            )
        ((plane_loss + args.event_action_anchor_weight * action_loss) * weight).backward()
        losses["plane"] += float(plane_loss.detach().cpu()) * weight
        losses["event_action_anchor"] += float(action_loss.detach().cpu()) * weight
        microbatches += 1
    for start in range(0, dense_observation.shape[0], args.microbatch_size):
        stop = min(start + args.microbatch_size, dense_observation.shape[0])
        weight = (stop - start) / dense_observation.shape[0]
        with _autocast_context(device, args.amp_dtype):
            output = model.observe_sequence(
                dense_observation[start:stop],
                dense_action[start:stop],
                deterministic_latent=True,
            )
            deterministic = output.posterior.deterministic[:, -1].float()
            logits = output.posterior.logits[:, -1].float()
            actor_action = model.deterministic_actor_action(
                model.actor_distribution(output.posterior.features[:, -1])
            ).float()
            anchor = F.mse_loss(
                deterministic, dense_reference["deterministic"][start:stop]
            ) + F.mse_loss(logits, dense_reference["logits"][start:stop])
            action_anchor = F.mse_loss(
                actor_action, dense_reference["action"][start:stop]
            )
            dense_loss = args.dense_feature_anchor_weight * anchor + (
                args.dense_action_anchor_weight * action_anchor
            )
        (dense_loss * weight).backward()
        losses["dense_anchor"] += float(dense_loss.detach().cpu()) * weight
        microbatches += 1
    gradient_norm = _finite_clip_grad_norm(
        parameters, args.max_gradient_norm, label="plane-sequence"
    )
    optimizer.step()
    preprojection_max = 0.0
    projected_values = 0
    with torch.no_grad():
        for name, parameter in model.rssm.sequence.named_parameters(prefix="rssm.sequence"):
            parent = parent_state[name]
            delta = parameter - parent
            preprojection_max = max(preprojection_max, float(delta.abs().max().cpu()))
            clipped = delta.clamp(-args.max_parameter_delta, args.max_parameter_delta)
            projected_values += int((clipped != delta).sum().cpu())
            parameter.copy_(parent + clipped)
    model.eval()
    post_validation = _predictions(
        model,
        probe,
        validation_observation,
        validation_action,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
        target_mean=target_mean,
        target_std=target_std,
    ).reshape(len(validation_events), len(targets))
    post_validation_metrics = _progress_metrics(validation_target_np, post_validation)
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
    changed, forbidden = _allowed_changes(parent_state, model.state_dict())
    output_payload = dict(payload)
    output_payload["model"] = model.state_dict()
    output_payload["plane_sequence_aux"] = {
        "schema": CONTINUATION_SCHEMA,
        "probe": {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in probe.items()
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "targets": targets,
        "feature_context": args.feature_context,
        "updates": 1,
    }
    _save_checkpoint(args.output_checkpoint, output_payload)
    source_hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "replay": _replay_hashes(args.replay_dir),
    }
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "train_events": int(len(train_events)),
        "validation_events": int(len(validation_events)),
        "test_events": int((split == 2).sum()),
        "test_evaluations": 0,
        "targets": list(targets),
        "target_mean": target_mean,
        "target_std": target_std,
        "probe_ridge": args.probe_ridge,
        "probe_active_features": int(probe["active_features"]),
        "learning_rate": args.learning_rate,
        "max_parameter_delta": args.max_parameter_delta,
        "preprojection_max_parameter_delta": preprojection_max,
        "projected_parameter_values": projected_values,
        "losses": losses,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "microbatches": microbatches,
        "preupdate_validation_progress": pre_validation_metrics,
        "validation_progress": post_validation_metrics,
        "dense_drift": dense_drift,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "sequence_updates": 1,
        "encoder_updates": 0,
        "posterior_updates": 0,
        "prior_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
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
    parser.add_argument("--feature-context", type=int, default=12)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=628)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--probe-ridge", type=float, default=0.1)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-seed", type=int, default=636001)
    parser.add_argument("--seed", type=int, default=636)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-parameter-delta", type=float, default=0.000001)
    parser.add_argument("--event-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--dense-feature-anchor-weight", type=float, default=1.0)
    parser.add_argument("--dense-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--max-gradient-norm", type=float, default=100.0)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.negative_offsets_values = tuple(
        int(value) for value in args.negative_offsets.split(",")
    )
    if not args.negative_offsets_values or any(
        value <= 0 for value in args.negative_offsets_values
    ):
        parser.error("negative offsets must be positive integers")
    if args.max_parameter_delta <= 0.0:
        parser.error("maximum parameter delta must be positive")
    return args


def main() -> None:
    print(json.dumps(continue_sequence(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
