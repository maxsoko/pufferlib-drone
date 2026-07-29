#!/usr/bin/env python3
"""Measure VQ2 learned-reward actor-gradient sign without saving a child.

The audit fixes one replay posterior batch, samples independent prior/action
imaginations, and measures the action-mean change from a virtual first Adam
step on one actor mean-head row. It restores every model tensor after each
trial, creates no native environment, and persists only a JSON report.
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
    _autocast_context,
    _burn_in_state,
    _repeat_imagination_starts,
    _select_imagination_starts,
)


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not values or not np.isfinite(array).all():
        raise RuntimeError("gradient audit received empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _virtual_adam_first_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    *,
    learning_rate: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return Adam's bias-corrected first-step value with zero weight decay."""

    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient shapes must match")
    if learning_rate <= 0.0 or epsilon <= 0.0:
        raise ValueError("learning rate and epsilon must be positive")
    if not torch.isfinite(gradient).all():
        raise RuntimeError("virtual Adam gradient is non-finite")
    return parameter - learning_rate * gradient / (gradient.abs() + epsilon)


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha_before = _sha256(args.checkpoint)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint does not contain training arguments")

    actor_distribution_mode = str(
        training_args.get("actor_distribution_mode", "legacy_tanh_normal")
    )
    if actor_distribution_mode != args.actor_distribution_mode:
        raise RuntimeError(
            "audit refuses an action-distribution migration: "
            f"parent={actor_distribution_mode}, requested={args.actor_distribution_mode}"
        )
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=actor_distribution_mode,
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    normalizer = payload.get("dreamerv3_return_normalizer", {})
    if isinstance(normalizer, dict):
        model.return_normalizer_low.fill_(float(normalizer.get("low", 0.0)))
        model.return_normalizer_high.fill_(float(normalizer.get("high", 0.0)))
    source_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    source_low = model.return_normalizer_low.detach().clone()
    source_high = model.return_normalizer_high.detach().clone()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.actor[-1]
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("actor output head must be linear")
    if args.actor_mean_channel < 0 or args.actor_mean_channel >= model.action_size:
        raise ValueError("actor mean channel is out of range")
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    source_head_weight = head.weight.detach().clone()
    source_head_bias = head.bias.detach().clone()

    replay_state = payload.get("replay")
    if not isinstance(replay_state, dict):
        raise RuntimeError("checkpoint does not contain replay state")
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

    replay_rng = _restore_rng(payload, device)
    replay_context = int(training_args["replay_context"])
    world_length = int(training_args["world_sequence_length"])
    batch_size = int(training_args["batch_size"])
    amp_dtype = str(training_args["amp_dtype"])
    observation, action, _reward, _continuation = replay.sample(
        batch_size,
        replay_context + world_length,
        device=device,
        rng=replay_rng,
    )
    with torch.no_grad():
        initial_state = _burn_in_state(
            model,
            observation[:, :replay_context],
            action[:, :replay_context],
        )
        posterior = model.observe_sequence(
            observation[:, replay_context:],
            action[:, replay_context:],
            initial_state=initial_state,
            deterministic_latent=True,
        )
        fixed_starts = _select_imagination_starts(
            posterior.posterior,
            mode="all",
            count=0,
        )
        fixed_features = fixed_starts.features.detach().clone()
        initial_distribution = model.actor_distribution(fixed_features)
        initial_action_mean = model.deterministic_actor_action(
            initial_distribution
        ).mean(0)

    combinations: list[dict[str, object]] = []
    for normalization in args.advantage_normalizations:
        for repeats in args.imagination_repeats:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            trials: list[dict[str, object]] = []
            for trial in range(args.trials):
                trial_seed = args.trial_seed + trial
                torch.manual_seed(trial_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(trial_seed)
                with torch.no_grad():
                    head.weight.copy_(source_head_weight)
                    head.bias.copy_(source_head_bias)
                    model.return_normalizer_low.copy_(source_low)
                    model.return_normalizer_high.copy_(source_high)
                model.zero_grad(set_to_none=True)
                starts = _repeat_imagination_starts(fixed_starts, repeats)
                with _autocast_context(device, amp_dtype):
                    imagination = model.imagination_loss(
                        starts,
                        horizon=args.imagination_horizon,
                        gamma=args.gamma,
                        lambda_=args.return_lambda,
                        entropy_weight=args.entropy_weight,
                        smoothness_weight=args.smoothness_weight,
                        reward_source="learned",
                        action_effort_weights=(0.0, 0.0, 0.0, 0.0),
                        advantage_normalization=normalization,
                        control_frequency_hz=args.control_frequency_hz,
                    )
                imagination.actor.backward()
                if head.weight.grad is None or head.bias.grad is None:
                    raise RuntimeError("actor head gradient is missing")
                channel = args.actor_mean_channel
                weight_gradient = head.weight.grad[channel].detach().clone()
                bias_gradient = head.bias.grad[channel].detach().clone()
                if not torch.isfinite(weight_gradient).all() or not torch.isfinite(
                    bias_gradient
                ).all():
                    raise RuntimeError("actor head gradient is non-finite")
                with torch.no_grad():
                    head.weight[channel].copy_(
                        _virtual_adam_first_step(
                            source_head_weight[channel],
                            weight_gradient,
                            learning_rate=args.learning_rate,
                        )
                    )
                    head.bias[channel].copy_(
                        _virtual_adam_first_step(
                            source_head_bias[channel],
                            bias_gradient,
                            learning_rate=args.learning_rate,
                        )
                    )
                    final_distribution = model.actor_distribution(fixed_features)
                    final_action_mean = model.deterministic_actor_action(
                        final_distribution
                    ).mean(0)
                    delta = final_action_mean - initial_action_mean
                other_exact = all(
                    bool(delta[index].item() == 0.0)
                    for index in range(model.action_size)
                    if index != channel
                )
                scalar_values = [
                    float(imagination.actor.detach().cpu()),
                    float(imagination.mean_return.detach().cpu()),
                    float(imagination.return_scale.detach().cpu()),
                    float(delta[channel].detach().cpu()),
                ]
                if not _finite(scalar_values):
                    raise RuntimeError("audit produced a non-finite scalar")
                trials.append(
                    {
                        "trial": trial,
                        "seed": trial_seed,
                        "collective_virtual_mean_delta": scalar_values[3],
                        "actor_loss": scalar_values[0],
                        "mean_return": scalar_values[1],
                        "return_scale": scalar_values[2],
                        "collective_weight_gradient_l2": float(
                            weight_gradient.float().norm().cpu()
                        ),
                        "collective_bias_gradient": float(bias_gradient.float().cpu()),
                        "normalizer_low_after": float(
                            model.return_normalizer_low.detach().cpu()
                        ),
                        "normalizer_high_after": float(
                            model.return_normalizer_high.detach().cpu()
                        ),
                        "other_action_mean_deltas_bit_exact_zero": other_exact,
                    }
                )
                with torch.no_grad():
                    head.weight.copy_(source_head_weight)
                    head.bias.copy_(source_head_bias)
                    model.return_normalizer_low.copy_(source_low)
                    model.return_normalizer_high.copy_(source_high)
                model.zero_grad(set_to_none=True)
                del starts, imagination, weight_gradient, bias_gradient
                gc.collect()

            deltas = [
                float(trial["collective_virtual_mean_delta"]) for trial in trials
            ]
            gradients = [
                float(trial["collective_weight_gradient_l2"]) for trial in trials
            ]
            combination: dict[str, object] = {
                "advantage_normalization": normalization,
                "imagination_repeats": repeats,
                "trials": trials,
                "collective_virtual_mean_delta": _summary(deltas),
                "collective_weight_gradient_l2": _summary(gradients),
                "negative_direction_count": sum(delta < 0.0 for delta in deltas),
                "negative_direction_fraction": sum(delta < 0.0 for delta in deltas)
                / len(deltas),
                "other_action_mean_deltas_bit_exact_zero": all(
                    bool(trial["other_action_mean_deltas_bit_exact_zero"])
                    for trial in trials
                ),
            }
            if device.type == "cuda":
                combination["cuda_peak_memory_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )
            combinations.append(combination)

    with torch.no_grad():
        head.weight.copy_(source_head_weight)
        head.bias.copy_(source_head_bias)
        model.return_normalizer_low.copy_(source_low)
        model.return_normalizer_high.copy_(source_high)
    final_model_state = model.state_dict()
    changed_model_keys = [
        name
        for name, value in final_model_state.items()
        if not torch.equal(value, source_model_state[name])
    ]
    replay_hashes_after = _replay_hashes(args.replay_dir)
    checkpoint_sha_after = _sha256(args.checkpoint)
    if changed_model_keys:
        raise RuntimeError(f"audit failed to restore model tensors: {changed_model_keys}")
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("audit changed source replay")
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("audit changed source checkpoint")

    report: dict[str, object] = {
        "contract": "fixed_posterior_virtual_adam_collective_gradient_snr_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "replay_dir": str(args.replay_dir),
        "replay_hashes": replay_hashes_after,
        "fixed_posterior_start_count": int(fixed_starts.deterministic.shape[0]),
        "fixed_posterior_batch_size": batch_size,
        "fixed_posterior_world_length": world_length,
        "actor_distribution_mode": actor_distribution_mode,
        "actor_mean_channel": args.actor_mean_channel,
        "initial_reference_action_mean": initial_action_mean.cpu().tolist(),
        "trials_per_combination": args.trials,
        "trial_seed": args.trial_seed,
        "imagination_horizon": args.imagination_horizon,
        "learning_rate": args.learning_rate,
        "combinations": combinations,
        "persisted_actor_updates": 0,
        "persisted_critic_updates": 0,
        "world_optimizer_steps": 0,
        "reward_optimizer_steps": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "privileged_actor_input": 0,
        "runtime_reward_dependency": 0,
        "model_tensor_changes_after_restore": changed_model_keys,
        "replay_records_unchanged": 1,
        "checkpoint_unchanged": 1,
        "device": str(device),
        "amp_dtype": amp_dtype,
    }
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
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--trial-seed", type=int, default=613000)
    parser.add_argument("--imagination-repeats", type=int, nargs="+", default=(1, 4))
    parser.add_argument(
        "--advantage-normalizations",
        choices=("legacy_center_std", "dreamerv3_percentile"),
        nargs="+",
        default=("legacy_center_std", "dreamerv3_percentile"),
    )
    parser.add_argument("--actor-mean-channel", type=int, default=2)
    parser.add_argument("--actor-distribution-mode", default="legacy_tanh_normal")
    parser.add_argument("--imagination-horizon", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--return-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-weight", type=float, default=3e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.002)
    parser.add_argument("--control-frequency-hz", type=float, default=64.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.trial_seed < 0:
        parser.error("--trial-seed cannot be negative")
    if any(repeats <= 0 for repeats in args.imagination_repeats):
        parser.error("--imagination-repeats values must be positive")
    if args.imagination_horizon < 2:
        parser.error("--imagination-horizon must be at least two")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
