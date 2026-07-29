#!/usr/bin/env python3
"""Continue the VQ2 world/reward model on dense plus gate-event replay.

The recurrent actor, critic, target critic, source replay, event corpus, and
native/FlightSim environments remain frozen. This path exists to teach the
world model true current-policy ordered-gate transitions before any actor is
allowed to consume learned reward again.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE, PRIVILEGED_SIZE
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _restore_rng,
    _sha256,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _autocast_context,
    _burn_in_state,
    _save_checkpoint,
    _world_model_update,
)


EVENT_SCHEMA = "vq2_actor_frozen_gate_event_windows_v1"


def _resolve_world_learning_rate(
    parent_learning_rate: float,
    override: float,
    *,
    reset_optimizer: bool,
) -> float:
    if parent_learning_rate <= 0.0 or override < 0.0:
        raise ValueError("world learning rates must be positive or zero override")
    if override > 0.0 and not reset_optimizer:
        raise ValueError("world learning-rate override requires a fresh optimizer")
    return override if override > 0.0 else parent_learning_rate


def _load_event_dataset(
    path: Path,
    *,
    context_length: int,
    event_threshold: float,
) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {
            "mask",
            "tail",
            "action",
            "reward",
            "continuation",
            "agent",
            "vector_step",
            "phase_after_event",
        }
        if set(archive.files) != required:
            raise RuntimeError("event corpus schema mismatch")
        dataset = {name: archive[name].copy() for name in archive.files}
    events = dataset["mask"].shape[0]
    length = context_length + 1
    expected = {
        "mask": (events, length, MASK_SIZE),
        "tail": (events, length, ENV_OBS_SIZE - MASK_SIZE),
        "action": (events, length, 4),
        "reward": (events, length),
        "continuation": (events, length),
    }
    for name, shape in expected.items():
        if dataset[name].shape != shape:
            raise RuntimeError(
                f"event corpus {name} shape mismatch: {dataset[name].shape} != {shape}"
            )
        if not np.isfinite(dataset[name]).all():
            raise RuntimeError(f"event corpus {name} contains non-finite values")
    if events <= 0:
        raise RuntimeError("event corpus is empty")
    if not (dataset["reward"][:, -1] > event_threshold).all():
        raise RuntimeError("event corpus final rewards do not satisfy threshold")
    if not (dataset["continuation"][:, :-1] == 1).all():
        raise RuntimeError("event corpus crosses a pre-event terminal")
    if np.abs(dataset["action"]).max() > 1.0:
        raise RuntimeError("event corpus action exceeds normalized bounds")
    return dataset


def _event_batch(
    dataset: dict[str, np.ndarray],
    indices: list[int] | np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = np.asarray(indices, dtype=np.int64)
    mask = torch.from_numpy(dataset["mask"][selected].astype(np.float32)).to(device)
    mask /= 255.0
    tail = torch.from_numpy(dataset["tail"][selected].astype(np.float32)).to(device)
    observation = torch.cat((mask, tail), -1)
    action = torch.from_numpy(dataset["action"][selected].astype(np.float32)).to(
        device
    )
    reward = torch.from_numpy(dataset["reward"][selected].astype(np.float32)).to(
        device
    )
    continuation = torch.from_numpy(
        dataset["continuation"][selected].astype(np.float32)
    ).to(device)
    return observation, action, reward, continuation


def _mix_batch(
    dense: tuple[torch.Tensor, ...],
    event: tuple[torch.Tensor, ...],
    *,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if len(dense) != 4 or len(event) != 4:
        raise ValueError("dense/event batches must contain four tensors")
    mixed = tuple(torch.cat((event_value, dense_value), 0) for dense_value, event_value in zip(dense, event, strict=True))
    if any(value.shape[0] != permutation.numel() for value in mixed):
        raise ValueError("batch permutation size mismatch")
    return tuple(value[permutation] for value in mixed)


@torch.no_grad()
def _world_validation_metrics(
    model: VQ2InformedDreamer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    context_length: int,
    amp_dtype: str,
) -> dict[str, float]:
    observation, action, reward, continuation = batch
    with _autocast_context(observation.device, amp_dtype):
        initial_state = _burn_in_state(
            model,
            observation[:, :context_length],
            action[:, :context_length],
        )
        loss, _output = model.world_model_loss(
            observation[:, context_length:],
            action[:, context_length:],
            reward[:, context_length:],
            continuation[:, context_length:],
            initial_state=initial_state,
            deterministic_latent=True,
        )
    return {
        field.name: float(getattr(loss, field.name).detach().float().cpu())
        for field in fields(loss)
    }


@torch.no_grad()
def _event_prior_metrics(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    *,
    context_length: int,
    microbatch_size: int,
    amp_dtype: str,
) -> dict[str, float]:
    predicted_rewards: list[torch.Tensor] = []
    target_rewards: list[torch.Tensor] = []
    current_phases: list[torch.Tensor] = []
    current_target_phases: list[torch.Tensor] = []
    predicted_phases: list[torch.Tensor] = []
    target_phases: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[start:stop, :context_length],
                action[start:stop, :context_length],
            )
            current_phase = model.privileged_decoder(state.features)[..., 32]
            prior = model.rssm.imagine_step(
                state,
                action[start:stop, context_length],
                deterministic_latent=True,
            )
            prior_features = prior.features
            predicted_reward = model.predict_reward(
                prior_features, action[start:stop, context_length]
            )
            predicted_phase = model.privileged_decoder(prior_features)[..., 32]
        target_phase = observation[
            start:stop, context_length, -PRIVILEGED_SIZE + 32
        ]
        current_target_phase = observation[
            start:stop, context_length - 1, -PRIVILEGED_SIZE + 32
        ]
        predicted_rewards.append(predicted_reward.float().cpu())
        target_rewards.append(reward[start:stop, context_length].float().cpu())
        current_phases.append(current_phase.float().cpu())
        current_target_phases.append(current_target_phase.float().cpu())
        predicted_phases.append(predicted_phase.float().cpu())
        target_phases.append(target_phase.float().cpu())
    predicted_reward = torch.cat(predicted_rewards)
    target_reward = torch.cat(target_rewards)
    current_phase = torch.cat(current_phases)
    current_target_phase = torch.cat(current_target_phases)
    predicted_phase = torch.cat(predicted_phases)
    target_phase = torch.cat(target_phases)
    predicted_crossing = (predicted_phase - current_phase) >= (0.5 / 6.0)
    target_crossing = (target_phase - current_target_phase) >= (0.5 / 6.0)
    return {
        "predicted_reward_mean": float(predicted_reward.mean()),
        "target_reward_mean": float(target_reward.mean()),
        "reward_mae": float((predicted_reward - target_reward).abs().mean()),
        "predicted_phase_mean": float(predicted_phase.mean()),
        "target_phase_mean": float(target_phase.mean()),
        "phase_mae": float((predicted_phase - target_phase).abs().mean()),
        "predicted_crossing_fraction": float(predicted_crossing.float().mean()),
        "target_crossing_fraction_from_decoded_context": float(
            target_crossing.float().mean()
        ),
    }


def continue_world(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("offline event continuation cannot overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parent_sha = _sha256(args.checkpoint)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    event_sha_before = _sha256(args.event_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    replay_state = payload.get("replay")
    if not isinstance(training_args, dict) or not isinstance(replay_state, dict):
        raise RuntimeError("parent checkpoint lacks training or replay state")
    if int(training_args["replay_context"]) != args.context_length:
        raise RuntimeError("event context does not match parent replay context")

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
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    world_parameters = [
        *model.rssm.parameters(),
        *model.privileged_decoder.parameters(),
        *model.reward_predictor.parameters(),
        *model.continue_predictor.parameters(),
    ]
    world_learning_rate = _resolve_world_learning_rate(
        float(training_args["world_lr"]),
        args.world_learning_rate,
        reset_optimizer=args.reset_world_optimizer,
    )
    world_optimizer = torch.optim.AdamW(
        world_parameters,
        lr=world_learning_rate,
        weight_decay=float(training_args.get("weight_decay", 0.0)),
    )
    if not args.reset_world_optimizer:
        world_optimizer.load_state_dict(payload["world_optimizer"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in world_parameters:
        parameter.requires_grad_(True)

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
    event_dataset = _load_event_dataset(
        args.event_dataset,
        context_length=args.context_length,
        event_threshold=args.event_threshold,
    )
    if args.event_batch_size > event_dataset["mask"].shape[0]:
        raise ValueError("event batch exceeds corpus size")
    dense_batch_size = args.batch_size - args.event_batch_size
    if dense_batch_size <= 0:
        raise ValueError("event-balanced update requires dense replay samples")
    rng = _restore_rng(payload, device)
    validation_rng = random.Random()
    validation_rng.setstate(rng.getstate())
    dense_validation_batch = replay.sample(
        args.batch_size,
        args.context_length + 1,
        device=device,
        rng=validation_rng,
    )
    full_event_observation, full_event_action, full_event_reward, _ = _event_batch(
        event_dataset,
        np.arange(event_dataset["mask"].shape[0]),
        device=device,
    )
    pre_event_metrics = _event_prior_metrics(
        model,
        full_event_observation,
        full_event_action,
        full_event_reward,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    pre_dense_metrics = _world_validation_metrics(
        model,
        dense_validation_batch,
        context_length=args.context_length,
        amp_dtype=args.amp_dtype,
    )

    last_loss = None
    last_gradient = None
    microbatches_processed = 0
    started = time.perf_counter()
    for update in range(args.updates):
        event_indices = rng.sample(
            range(event_dataset["mask"].shape[0]), args.event_batch_size
        )
        event_batch = _event_batch(event_dataset, event_indices, device=device)
        dense_batch = replay.sample(
            dense_batch_size,
            args.context_length + 1,
            device=device,
            rng=rng,
        )
        permutation = torch.randperm(args.batch_size, device=device)
        observation, action, reward, continuation = _mix_batch(
            dense_batch,
            event_batch,
            permutation=permutation,
        )
        loss, _starts, gradient, microbatches = _world_model_update(
            model=model,
            optimizer=world_optimizer,
            observation=observation,
            action=action,
            reward=reward,
            continuation=continuation,
            replay_context=args.context_length,
            microbatch_size=args.microbatch_size,
            amp_dtype=args.amp_dtype,
            free_nats=float(training_args.get("free_nats", 1.0)),
            world_parameters=world_parameters,
            deterministic_latent=False,
        )
        last_loss = loss
        last_gradient = gradient
        microbatches_processed += microbatches
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "offline_event_world_updates": update + 1,
                        "world_total": float(loss.total.cpu()),
                        "world_reward": float(loss.reward.cpu()),
                        "wall_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_loss is not None and last_gradient is not None
    post_event_metrics = _event_prior_metrics(
        model,
        full_event_observation,
        full_event_action,
        full_event_reward,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    post_dense_metrics = _world_validation_metrics(
        model,
        dense_validation_batch,
        context_length=args.context_length,
        amp_dtype=args.amp_dtype,
    )
    child_model_state = model.state_dict()
    changed_model_keys = sorted(
        name
        for name, value in child_model_state.items()
        if not torch.equal(value, parent_model_state[name])
    )
    allowed_prefixes = (
        "rssm.",
        "privileged_decoder.",
        "reward_predictor.",
        "continue_predictor.",
    )
    forbidden_changes = [
        name for name in changed_model_keys if not name.startswith(allowed_prefixes)
    ]
    if not changed_model_keys:
        raise RuntimeError("event-balanced update changed no world tensor")
    if forbidden_changes:
        raise RuntimeError(
            f"event-balanced update changed forbidden tensors: {forbidden_changes}"
        )
    if _replay_hashes(args.replay_dir) != replay_hashes_before:
        raise RuntimeError("event-balanced update changed dense replay")
    if _sha256(args.event_dataset) != event_sha_before:
        raise RuntimeError("event-balanced update changed event corpus")
    if _sha256(args.checkpoint) != parent_sha:
        raise RuntimeError("event-balanced update changed parent checkpoint")

    previous_offline_updates = int(payload.get("offline_event_world_updates", 0))
    report: dict[str, object] = {
        "contract": "vq2_event_balanced_world_continuation_v1",
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha,
        "dense_replay_dir": str(args.replay_dir),
        "dense_replay_hashes": replay_hashes_before,
        "dense_replay_unchanged": 1,
        "event_dataset": str(args.event_dataset),
        "event_dataset_sha256": event_sha_before,
        "event_dataset_unchanged": 1,
        "event_dataset_count": int(event_dataset["mask"].shape[0]),
        "updates": args.updates,
        "cumulative_offline_event_world_updates": previous_offline_updates
        + args.updates,
        "batch_size": args.batch_size,
        "event_batch_size": args.event_batch_size,
        "dense_batch_size": dense_batch_size,
        "event_batch_fraction": args.event_batch_size / args.batch_size,
        "context_length": args.context_length,
        "world_sequence_length": 1,
        "microbatch_size": args.microbatch_size,
        "microbatches_processed": microbatches_processed,
        "amp_dtype": args.amp_dtype,
        "world_learning_rate": world_optimizer.param_groups[0]["lr"],
        "world_optimizer_reset": int(args.reset_world_optimizer),
        "pre_event_prior_metrics": pre_event_metrics,
        "post_event_prior_metrics": post_event_metrics,
        "pre_dense_validation": pre_dense_metrics,
        "post_dense_validation": post_dense_metrics,
        "last_world_gradient_norm": float(last_gradient.cpu()),
        "changed_model_keys": changed_model_keys,
        "changed_model_key_count": len(changed_model_keys),
        "forbidden_model_key_changes": forbidden_changes,
        "actor_tensor_changes": sum(name.startswith("actor.") for name in changed_model_keys),
        "critic_tensor_changes": sum(
            name.startswith(("critic.", "target_critic."))
            for name in changed_model_keys
        ),
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    report.update(
        {
            f"last_world_{field.name}": float(getattr(last_loss, field.name).cpu())
            for field in fields(last_loss)
        }
    )
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )

    output = dict(payload)
    output["model"] = child_model_state
    output["world_optimizer"] = world_optimizer.state_dict()
    output["offline_event_world_updates"] = previous_offline_updates + args.updates
    output["offline_event_world_parent_checkpoint"] = str(args.checkpoint)
    output["offline_event_world_parent_sha256"] = parent_sha
    output["offline_event_dataset"] = str(args.event_dataset)
    output["offline_event_dataset_sha256"] = event_sha_before
    output["rng_state"] = {
        "python": rng.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    output["metrics"] = report
    _save_checkpoint(args.output_checkpoint, output)
    report["output_checkpoint"] = str(args.output_checkpoint)
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--event-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--reset-world-optimizer", action="store_true")
    parser.add_argument(
        "--world-learning-rate",
        type=float,
        default=0.0,
        help="positive override allowed only with --reset-world-optimizer",
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for name in ("updates", "batch_size", "event_batch_size", "context_length", "microbatch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.event_batch_size >= args.batch_size:
        parser.error("--event-batch-size must leave at least one dense sample")
    if args.microbatch_size > args.batch_size:
        parser.error("--microbatch-size cannot exceed batch size")
    if args.event_threshold <= 0.0:
        parser.error("--event-threshold must be positive")
    if args.world_learning_rate < 0.0:
        parser.error("--world-learning-rate cannot be negative")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("offline event continuation refuses to overwrite outputs")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_world(parse_args()), indent=2, sort_keys=True))
