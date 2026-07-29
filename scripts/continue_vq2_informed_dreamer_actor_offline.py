#!/usr/bin/env python3
"""Train only the recurrent Puffer actor/critic from frozen VQ2 imagination.

The executable creates no native environment and collects no transitions. It
samples posterior start states from source-locked replay, freezes the entire
world model, and updates the actor and critic using decoder-derived progress in
    prior-only imagination. Mutation and replay hashes are checked before writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer  # noqa: E402
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _autocast_context,
    _burn_in_state,
    _finite_clip_grad_norm,
    _repeat_imagination_starts,
    _save_checkpoint,
    _select_imagination_starts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _restore_rng(payload: dict, device: torch.device) -> random.Random:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        raise RuntimeError("parent checkpoint does not contain RNG state")
    rng = random.Random()
    rng.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if not cuda_state:
            raise RuntimeError("parent checkpoint does not contain CUDA RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    return rng


def _configure_actor_training(
    model: VQ2InformedDreamer,
    mean_channels: tuple[int, ...] | None,
    *,
    weight_decay: float,
) -> list[torch.nn.Parameter]:
    """Select whole-actor or source-locked mean-head row training."""

    if mean_channels is None:
        parameters = list(model.actor.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters
    if weight_decay != 0.0:
        raise ValueError("restricted actor-row training requires zero weight decay")
    if not mean_channels or len(set(mean_channels)) != len(mean_channels):
        raise ValueError("actor mean channels must be nonempty and unique")
    if any(channel < 0 or channel >= model.action_size for channel in mean_channels):
        raise ValueError("actor mean channel index is out of range")
    for parameter in model.actor.parameters():
        parameter.requires_grad_(False)
    head = model.actor[-1]
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("actor output head must be linear")
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    weight_mask = torch.zeros_like(head.weight)
    bias_mask = torch.zeros_like(head.bias)
    weight_mask[list(mean_channels)] = 1.0
    bias_mask[list(mean_channels)] = 1.0
    head.weight.register_hook(lambda gradient: gradient * weight_mask)
    head.bias.register_hook(lambda gradient: gradient * bias_mask)
    return [head.weight, head.bias]


def continue_actor(args: argparse.Namespace) -> dict:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("offline actor training must not overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    parent_sha256 = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("parent checkpoint does not contain training arguments")

    actor_distribution_mode = (
        args.actor_distribution_mode
        or str(training_args.get("actor_distribution_mode", "legacy_tanh_normal"))
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
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    critic_parameters = list(model.critic.parameters())
    actor_mean_channels = (
        None
        if args.actor_mean_channels is None
        else tuple(args.actor_mean_channels)
    )
    actor_parameters = _configure_actor_training(
        model,
        actor_mean_channels,
        weight_decay=args.weight_decay,
    )
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    actor_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic_parameters,
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )

    replay_state = payload.get("replay")
    if not isinstance(replay_state, dict):
        raise RuntimeError("parent checkpoint does not contain replay state")
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

    rng = _restore_rng(payload, device)
    replay_context = int(training_args["replay_context"])
    world_length = int(training_args["world_sequence_length"])
    sample_length = replay_context + world_length
    batch_size = int(training_args["batch_size"])
    amp_dtype = str(training_args["amp_dtype"])
    last_imagination = None
    last_actor_gradient = None
    last_critic_gradient = None
    reference_start_features = None
    initial_reference_action_mean = None
    started = time.perf_counter()

    for update in range(args.updates):
        observation, action, _reward, _continuation = replay.sample(
            batch_size, sample_length, device=device, rng=rng
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
            starts = _select_imagination_starts(
                posterior.posterior,
                mode=args.imagination_start_mode,
                count=args.imagination_start_count,
            )
            starts = _repeat_imagination_starts(starts, args.imagination_repeats)
            if reference_start_features is None:
                reference_start_features = starts.features.detach().clone()
                initial_reference_distribution = model.actor_distribution(
                    reference_start_features
                )
                initial_reference_action_mean = model.deterministic_actor_action(
                    initial_reference_distribution
                ).mean(0)
        with _autocast_context(device, amp_dtype):
            imagination = model.imagination_loss(
                starts,
                horizon=args.imagination_horizon,
                gamma=args.gamma,
                lambda_=args.return_lambda,
                entropy_weight=args.entropy_weight,
                smoothness_weight=args.smoothness_weight,
                reward_source=args.imagination_reward_source,
                action_effort_weights=tuple(args.action_effort_weights),
                advantage_normalization=args.advantage_normalization,
                control_frequency_hz=args.imagination_control_frequency_hz,
            )
        actor_optimizer.zero_grad(set_to_none=True)
        imagination.actor.backward()
        last_actor_gradient = _finite_clip_grad_norm(
            actor_parameters, 100.0, label="offline actor"
        )
        actor_optimizer.step()
        critic_optimizer.zero_grad(set_to_none=True)
        imagination.critic.backward()
        last_critic_gradient = _finite_clip_grad_norm(
            critic_parameters, 100.0, label="offline critic"
        )
        critic_optimizer.step()
        model.update_target_critic()
        last_imagination = imagination
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "offline_actor_updates": update + 1,
                        "actor_loss": float(imagination.actor.detach().cpu()),
                        "mean_return": float(
                            imagination.mean_return.detach().cpu()
                        ),
                        "wall_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_imagination is not None
    assert last_actor_gradient is not None and last_critic_gradient is not None
    assert reference_start_features is not None
    assert initial_reference_action_mean is not None
    child_model_state = model.state_dict()
    changed_model_keys = sorted(
        name
        for name, value in child_model_state.items()
        if not torch.equal(value, parent_model_state[name])
    )
    allowed_prefixes = ("actor.", "critic.", "target_critic.")
    unexpected = [
        name
        for name in changed_model_keys
        if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"offline actor changed frozen world tensors: {unexpected}")
    if not any(name.startswith("actor.") for name in changed_model_keys):
        raise RuntimeError("offline actor training did not change the actor")
    frozen_actor_row_changes = 0
    if actor_mean_channels is not None:
        allowed_rows = set(actor_mean_channels)
        parent_head = parent_model_state["actor.5.weight"]
        child_head = child_model_state["actor.5.weight"]
        parent_bias = parent_model_state["actor.5.bias"]
        child_bias = child_model_state["actor.5.bias"]
        for row in range(parent_head.shape[0]):
            if row not in allowed_rows:
                frozen_actor_row_changes += int(
                    not torch.equal(parent_head[row], child_head[row])
                )
                frozen_actor_row_changes += int(
                    not torch.equal(parent_bias[row], child_bias[row])
                )
        nonhead_actor_changes = [
            name
            for name in changed_model_keys
            if name.startswith("actor.")
            and name not in ("actor.5.weight", "actor.5.bias")
        ]
        frozen_actor_row_changes += len(nonhead_actor_changes)
        if frozen_actor_row_changes:
            raise RuntimeError("restricted actor training changed frozen actor rows")
    replay_hashes_after = _replay_hashes(args.replay_dir)
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("source-locked replay changed during actor training")

    elapsed = time.perf_counter() - started
    previous_offline_updates = int(payload.get("offline_actor_updates", 0))
    previous_actor_updates = int(payload.get("actor_updates", 0))
    with torch.no_grad():
        zero_state = model.rssm.initial(batch_size, device=device)
        final_distribution = model.actor_distribution(zero_state.features)
        final_mean_action = model.deterministic_actor_action(final_distribution).mean(0)
        final_std = final_distribution.stddev.mean(0)
        final_reference_distribution = model.actor_distribution(
            reference_start_features
        )
        final_reference_action_mean = model.deterministic_actor_action(
            final_reference_distribution
        ).mean(0)
    report: dict[str, object] = {
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "replay_dir": str(args.replay_dir),
        "replay_hashes": replay_hashes_after,
        "replay_records_unchanged": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "world_optimizer_steps": 0,
        "reward_optimizer_steps": 0,
        "offline_actor_updates": args.updates,
        "offline_critic_updates": args.updates,
        "cumulative_offline_actor_updates": previous_offline_updates + args.updates,
        "changed_model_keys": changed_model_keys,
        "changed_model_key_count": len(changed_model_keys),
        "forbidden_world_model_key_changes": len(unexpected),
        "imagination_reward_source": args.imagination_reward_source,
        "imagination_control_frequency_hz": (
            args.imagination_control_frequency_hz
        ),
        "imagination_start_mode": args.imagination_start_mode,
        "imagination_start_count": starts.deterministic.shape[0],
        "imagination_repeats": args.imagination_repeats,
        "actor_distribution_mode": actor_distribution_mode,
        "advantage_normalization": args.advantage_normalization,
        "optimizer_weight_decay": args.weight_decay,
        "actor_mean_channels_trained": (
            "all" if actor_mean_channels is None else list(actor_mean_channels)
        ),
        "frozen_actor_row_changes": frozen_actor_row_changes,
        "privileged_actor_input": 0,
        "runtime_reward_dependency": 0,
        "actor_optimizer_reset_for_new_reward": 1,
        "critic_optimizer_reset_for_new_reward": 1,
        "actor_loss": float(last_imagination.actor.detach().cpu()),
        "critic_loss": float(last_imagination.critic.detach().cpu()),
        "mean_return": float(last_imagination.mean_return.detach().cpu()),
        "return_scale": float(last_imagination.return_scale.detach().cpu()),
        "return_normalizer_low": float(model.return_normalizer_low.cpu()),
        "return_normalizer_high": float(model.return_normalizer_high.cpu()),
        "mean_continuation_weight": float(
            last_imagination.mean_continuation_weight.detach().cpu()
        ),
        "entropy": float(last_imagination.entropy.detach().cpu()),
        "smoothness": float(last_imagination.smoothness.detach().cpu()),
        "mean_action_effort": float(
            last_imagination.mean_effort.detach().cpu()
        ),
        "action_effort_weights": list(args.action_effort_weights),
        "last_actor_gradient_norm": float(last_actor_gradient.detach().cpu()),
        "last_critic_gradient_norm": float(last_critic_gradient.detach().cpu()),
        "final_zero_state_action_mean": final_mean_action.cpu().tolist(),
        "final_zero_state_action_std": final_std.cpu().tolist(),
        "reference_start_count": int(reference_start_features.shape[0]),
        "initial_reference_start_action_mean": (
            initial_reference_action_mean.cpu().tolist()
        ),
        "final_reference_start_action_mean": (
            final_reference_action_mean.cpu().tolist()
        ),
        "wall_seconds": elapsed,
        "device": str(device),
        "amp_dtype": amp_dtype,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )

    output = dict(payload)
    output["args"] = dict(training_args)
    output["args"]["imagination_reward_source"] = args.imagination_reward_source
    output["args"]["imagination_control_frequency_hz"] = (
        args.imagination_control_frequency_hz
    )
    output["args"]["actor_distribution_mode"] = actor_distribution_mode
    output["args"]["advantage_normalization"] = args.advantage_normalization
    output["args"]["optimizer_weight_decay"] = args.weight_decay
    output["model"] = child_model_state
    output["actor_optimizer"] = actor_optimizer.state_dict()
    output["critic_optimizer"] = critic_optimizer.state_dict()
    output["actor_updates"] = previous_actor_updates + args.updates
    output["offline_actor_updates"] = previous_offline_updates + args.updates
    output["offline_actor_parent_checkpoint"] = str(args.checkpoint)
    output["offline_actor_parent_sha256"] = parent_sha256
    output["dreamerv3_return_normalizer"] = {
        "low": float(model.return_normalizer_low.cpu()),
        "high": float(model.return_normalizer_high.cpu()),
        "rate": 0.01,
        "limit": 1.0,
        "perclo": 5.0,
        "perchi": 95.0,
        "debias": False,
    }
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
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--actor-learning-rate", type=float, default=4e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--actor-distribution-mode",
        choices=("legacy_tanh_normal", "dreamerv3_bounded_normal"),
        help="override parent policy distribution for a source-locked migration",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("legacy_center_std", "dreamerv3_percentile"),
        default="legacy_center_std",
    )
    parser.add_argument(
        "--actor-mean-channels",
        type=int,
        nargs="+",
        help="train only these actor mean-head rows; freezes trunk/std/other means",
    )
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument(
        "--imagination-start-mode",
        choices=("last", "all"),
        default="last",
        help="last preserves the local legacy; all samples replay-time posteriors",
    )
    parser.add_argument(
        "--imagination-start-count",
        type=int,
        default=0,
        help="bounded subset for all mode; zero uses all batch*time states",
    )
    parser.add_argument(
        "--imagination-repeats",
        type=int,
        default=1,
        help="independent prior/action samples per selected posterior start",
    )
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--return-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-weight", type=float, default=3e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.002)
    parser.add_argument(
        "--imagination-reward-source",
        choices=(
            "learned",
            "informed_decoder_progress",
            "informed_decoder_skydreamer",
        ),
        default="informed_decoder_progress",
        help="versioned decoder-derived task reward; never a runtime input",
    )
    parser.add_argument(
        "--imagination-control-frequency-hz",
        type=float,
        default=64.0,
        help="native control frequency used by the paper's body-rate penalty",
    )
    parser.add_argument(
        "--action-effort-weights",
        type=float,
        nargs=4,
        metavar=("PITCH", "ROLL", "COLLECTIVE", "YAW"),
        default=(0.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.actor_learning_rate <= 0.0 or args.critic_learning_rate <= 0.0:
        parser.error("learning rates must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative")
    if args.imagination_horizon < 2:
        parser.error("--imagination-horizon must be at least two")
    if args.imagination_control_frequency_hz <= 0.0:
        parser.error("--imagination-control-frequency-hz must be positive")
    if args.imagination_start_count < 0:
        parser.error("--imagination-start-count cannot be negative")
    if args.imagination_repeats <= 0:
        parser.error("--imagination-repeats must be positive")
    if args.imagination_start_mode == "last" and args.imagination_start_count:
        parser.error("--imagination-start-count requires --imagination-start-mode all")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0,1]")
    if any(weight < 0.0 for weight in args.action_effort_weights):
        parser.error("--action-effort-weights cannot be negative")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_actor(parse_args()), indent=2, sort_keys=True))
