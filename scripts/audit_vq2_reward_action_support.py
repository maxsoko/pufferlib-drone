#!/usr/bin/env python3
"""Audit N604 reward-head action resolution and replay gate-event support.

The audit is read-only. It reconstructs one source-locked posterior batch,
measures fixed-state and deterministic-one-step learned-reward sensitivity to
small collective perturbations, and inventories native reward events in the
replay. It creates no native environment and performs no optimizer step.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _restore_rng,
    _sha256,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _burn_in_state,
    _select_imagination_starts,
)


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("audit received empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _center_bin_fraction(values: np.ndarray, first_positive_bin: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or first_positive_bin <= 0.0:
        raise ValueError("center-bin inputs must be nonempty with a positive bound")
    return float((np.abs(array) < first_positive_bin).mean())


def _candidate_starts(
    valid: np.ndarray,
    continuation: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Return reset-free logical ``(start, agent)`` sequence indices."""

    if valid.ndim != 2 or continuation.shape != valid.shape:
        raise ValueError("valid and continuation must be aligned [time, agent]")
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    time = valid.shape[0]
    if time < sequence_length:
        return np.empty((0, 2), dtype=np.int64)
    valid_i = valid.astype(np.int32)
    valid_prefix = np.concatenate(
        (np.zeros((1, valid.shape[1]), dtype=np.int32), np.cumsum(valid_i, 0)), 0
    )
    good = (
        valid_prefix[sequence_length:] - valid_prefix[:-sequence_length]
        == sequence_length
    )
    if sequence_length > 1:
        continuation_i = continuation.astype(np.int32)
        continuation_prefix = np.concatenate(
            (
                np.zeros((1, continuation.shape[1]), dtype=np.int32),
                np.cumsum(continuation_i, 0),
            ),
            0,
        )
        continuation_counts = (
            continuation_prefix[sequence_length - 1 : time]
            - continuation_prefix[: time - sequence_length + 1]
        )
        good &= continuation_counts == sequence_length - 1
    return np.argwhere(good)


def _event_support(
    *,
    valid: np.ndarray,
    continuation: np.ndarray,
    reward: np.ndarray,
    sequence_length: int,
    context: int,
    event_threshold: float,
    sampled_sequences: int,
) -> dict[str, float | int]:
    """Measure event exposure under uniform reset-free sequence sampling."""

    if reward.shape != valid.shape:
        raise ValueError("reward and valid must be aligned")
    if not 0 <= context < sequence_length:
        raise ValueError("context must be inside the sequence")
    candidates = _candidate_starts(valid, continuation, sequence_length)
    if not len(candidates):
        raise RuntimeError("replay has no reset-free candidate sequence")
    events = valid & (reward > event_threshold)
    event_prefix = np.concatenate(
        (
            np.zeros((1, events.shape[1]), dtype=np.int32),
            np.cumsum(events.astype(np.int32), 0),
        ),
        0,
    )
    start = candidates[:, 0]
    agent = candidates[:, 1]
    event_counts = (
        event_prefix[start + sequence_length, agent]
        - event_prefix[start + context, agent]
    )
    candidate_mask = np.zeros(
        (valid.shape[0] - sequence_length + 1, valid.shape[1]), dtype=bool
    )
    candidate_mask[start, agent] = True
    reachable = 0
    for event_time, event_agent in np.argwhere(events):
        low = max(0, int(event_time) - sequence_length + 1)
        high = min(
            candidate_mask.shape[0] - 1,
            int(event_time) - context,
        )
        if low <= high and candidate_mask[low : high + 1, event_agent].any():
            reachable += 1
    mean_count = float(event_counts.mean())
    event_sequence_fraction = float((event_counts > 0).mean())
    return {
        "valid_transition_count": int(valid.sum()),
        "native_event_count": int(events.sum()),
        "native_event_fraction": float(events.sum() / max(1, valid.sum())),
        "candidate_sequence_count": int(len(candidates)),
        "candidate_sequences_with_event": int((event_counts > 0).sum()),
        "candidate_sequence_event_fraction": event_sequence_fraction,
        "mean_event_targets_per_sampled_sequence": mean_count,
        "expected_event_targets_at_training_sample_count": float(
            mean_count * sampled_sequences
        ),
        "expected_event_sequences_at_training_sample_count": float(
            event_sequence_fraction * sampled_sequences
        ),
        "distinct_events_reachable_in_world_window": reachable,
    }


def _surface_row(
    reward: torch.Tensor,
    derivative: torch.Tensor,
    *,
    offset: float,
) -> dict[str, object]:
    reward_np = reward.detach().float().cpu().numpy()
    derivative_np = derivative.detach().float().cpu().numpy()
    return {
        "collective_offset": offset,
        "reward": _summary(reward_np),
        "collective_reward_derivative": _summary(derivative_np),
        "negative_derivative_count": int((derivative_np < 0.0).sum()),
        "negative_derivative_fraction": float((derivative_np < 0.0).mean()),
    }


def _local_surface(
    model: VQ2InformedDreamer,
    starts,
    base_action: torch.Tensor,
    *,
    offsets: tuple[float, ...],
    channel: int,
    one_step: bool,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for offset in offsets:
        action = base_action.detach().clone()
        action[:, channel] += offset
        clipped = (action[:, channel].abs() > 1.0).sum().item()
        action[:, channel].clamp_(-1.0, 1.0)
        action.requires_grad_(True)
        if one_step:
            state = model.rssm.imagine_step(
                starts.detached(), action, deterministic_latent=True
            )
            reward = model.predict_reward(state.features, action)
        else:
            reward = model.predict_reward(starts.features.detach(), action)
        derivative = torch.autograd.grad(reward.sum(), action)[0][:, channel]
        row = _surface_row(reward, derivative, offset=offset)
        row["clipped_action_count"] = int(clipped)
        rows.append(row)
        del action, reward, derivative
        gc.collect()
    zero = next(row for row in rows if row["collective_offset"] == 0.0)
    zero_mean = float(zero["reward"]["mean"])
    for row in rows:
        row["mean_reward_delta_from_zero"] = float(row["reward"]["mean"]) - zero_mean
    negative = next(row for row in rows if row["collective_offset"] == -0.1)
    positive = next(row for row in rows if row["collective_offset"] == 0.1)
    return {
        "mode": "deterministic_one_step" if one_step else "fixed_feature_direct",
        "rows": rows,
        "negative_0p1_beats_zero": bool(
            float(negative["reward"]["mean"]) > zero_mean
        ),
        "zero_beats_positive_0p1": bool(
            zero_mean > float(positive["reward"]["mean"])
        ),
        "ordered_negative_zero_positive": bool(
            float(negative["reward"]["mean"])
            > zero_mean
            > float(positive["reward"]["mean"])
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha_before = _sha256(args.checkpoint)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    replay_state = payload.get("replay")
    if not isinstance(training_args, dict) or not isinstance(replay_state, dict):
        raise RuntimeError("checkpoint lacks training arguments or replay state")
    if not bool(training_args.get("distributional_reward", False)):
        raise RuntimeError("audit requires the distributional reward head")
    if not bool(training_args.get("action_conditioned_reward", False)):
        raise RuntimeError("audit requires the action-conditioned reward head")

    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(
            training_args.get("actor_distribution_mode", "legacy_tanh_normal")
        ),
        distributional_reward=True,
        action_conditioned_reward=True,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    source_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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

    chronological = replay._physical(np.arange(replay.size))
    valid = np.asarray(replay.valid[chronological], dtype=bool)
    continuation = np.asarray(replay.continuation[chronological], dtype=bool)
    reward = np.asarray(replay.reward[chronological], dtype=np.float32)
    action = np.asarray(replay.action[chronological], dtype=np.float32)
    valid_reward = reward[valid]
    event = valid & (reward > args.event_threshold)
    event_action = action[event]
    non_event_action = action[valid & ~event]
    first_positive_bin = float(model.reward_bins[model.reward_bins > 0.0][0].cpu())
    replay_context = int(training_args["replay_context"])
    world_length = int(training_args["world_sequence_length"])
    training_updates = int(args.training_updates)
    training_batch_size = int(training_args["batch_size"])
    support = _event_support(
        valid=valid,
        continuation=continuation,
        reward=reward,
        sequence_length=replay_context + world_length,
        context=replay_context,
        event_threshold=args.event_threshold,
        sampled_sequences=training_updates * training_batch_size,
    )
    support.update(
        {
            "training_updates": training_updates,
            "training_batch_size": training_batch_size,
            "supervised_targets_per_sequence": world_length,
            "total_supervised_reward_targets": (
                training_updates * training_batch_size * world_length
            ),
            "first_positive_reward_bin": first_positive_bin,
            "fraction_valid_rewards_inside_center_bin": _center_bin_fraction(
                valid_reward, first_positive_bin
            ),
            "distinct_float16_reward_values": int(np.unique(valid_reward).size),
            "valid_reward": _summary(valid_reward),
            "valid_reward_percentiles": {
                str(percentile): float(np.percentile(valid_reward, percentile))
                for percentile in (0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)
            },
            "distinct_event_agents": int(np.unique(np.argwhere(event)[:, 1]).size)
            if event.any()
            else 0,
            "distinct_event_vector_steps": int(
                np.unique(np.argwhere(event)[:, 0]).size
            )
            if event.any()
            else 0,
            "event_reward_values": valid_reward[valid_reward > args.event_threshold]
            .astype(float)
            .tolist(),
            "event_action_mean": (
                event_action.mean(0).astype(float).tolist() if len(event_action) else []
            ),
            "event_action_std": (
                event_action.std(0).astype(float).tolist() if len(event_action) else []
            ),
            "non_event_action_mean": non_event_action.mean(0).astype(float).tolist(),
            "non_event_action_std": non_event_action.std(0).astype(float).tolist(),
        }
    )

    replay_rng = _restore_rng(payload, device)
    observation, sampled_action, _sampled_reward, _sampled_continuation = replay.sample(
        training_batch_size,
        replay_context + world_length,
        device=device,
        rng=replay_rng,
    )
    with torch.no_grad():
        initial_state = _burn_in_state(
            model,
            observation[:, :replay_context],
            sampled_action[:, :replay_context],
        )
        posterior = model.observe_sequence(
            observation[:, replay_context:],
            sampled_action[:, replay_context:],
            initial_state=initial_state,
            deterministic_latent=True,
        )
        starts = _select_imagination_starts(
            posterior.posterior,
            mode="all",
            count=0,
        )
        base_action = model.deterministic_actor_action(
            model.actor_distribution(starts.features)
        ).detach()

    offsets = tuple(float(value) for value in args.collective_offsets)
    surfaces = [
        _local_surface(
            model,
            starts,
            base_action,
            offsets=offsets,
            channel=args.actor_mean_channel,
            one_step=one_step,
        )
        for one_step in (False, True)
    ]

    final_model_state = model.state_dict()
    changed_model_keys = [
        name
        for name, value in final_model_state.items()
        if not torch.equal(value, source_model_state[name])
    ]
    replay_hashes_after = _replay_hashes(args.replay_dir)
    checkpoint_sha_after = _sha256(args.checkpoint)
    if changed_model_keys:
        raise RuntimeError(f"audit changed model tensors: {changed_model_keys}")
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("audit changed source replay")
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("audit changed source checkpoint")

    report: dict[str, object] = {
        "contract": "vq2_reward_subbin_and_event_support_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "replay_dir": str(args.replay_dir),
        "replay_hashes": replay_hashes_after,
        "fixed_posterior_start_count": int(starts.deterministic.shape[0]),
        "actor_mean_channel": args.actor_mean_channel,
        "actor_distribution_mode": model.actor_distribution_mode,
        "base_action_mean": base_action.float().mean(0).cpu().tolist(),
        "base_action_abs_max": base_action.float().abs().amax(0).cpu().tolist(),
        "collective_offsets": list(offsets),
        "event_threshold": args.event_threshold,
        "replay_event_support": support,
        "learned_reward_surfaces": surfaces,
        "model_tensor_changes": changed_model_keys,
        "checkpoint_unchanged": 1,
        "replay_records_unchanged": 1,
        "persisted_actor_updates": 0,
        "persisted_critic_updates": 0,
        "world_optimizer_steps": 0,
        "reward_optimizer_steps": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "privileged_actor_input": 0,
        "runtime_reward_dependency": 0,
        "device": str(device),
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--training-updates", type=int, default=114)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--actor-mean-channel", type=int, default=2)
    parser.add_argument(
        "--collective-offsets",
        type=float,
        nargs="+",
        default=(-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, -0.01, 0.0,
                 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.training_updates <= 0:
        parser.error("--training-updates must be positive")
    if args.event_threshold <= 0.0:
        parser.error("--event-threshold must be positive")
    if not 0 <= args.actor_mean_channel < 4:
        parser.error("--actor-mean-channel must be in [0,3]")
    if len(set(args.collective_offsets)) != len(args.collective_offsets):
        parser.error("--collective-offsets must be unique")
    for required in (-0.1, 0.0, 0.1):
        if required not in args.collective_offsets:
            parser.error("--collective-offsets must include -0.1, 0, and 0.1")
    if any(abs(value) > 0.5 for value in args.collective_offsets):
        parser.error("--collective-offsets exceed the local audit range")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
