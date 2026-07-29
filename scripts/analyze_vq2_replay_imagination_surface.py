#!/usr/bin/env python3
"""Audit VQ2 action ordering over the actor's replay-time start distribution.

This diagnostic is native-replay only. Privileged decoder output is used only
to compute the same training reward as latent imagination; it never feeds the
actor, plant, or a deployed preprocessing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer  # noqa: E402
from scripts.analyze_vq2_informed_reward_channels import (  # noqa: E402
    ACTION_VALUES,
    CHANNELS,
)
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _burn_in_state,
    _select_imagination_starts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concatenate(states: list[RSSMState]) -> RSSMState:
    return RSSMState(
        *(torch.cat([value[index] for value in states], 0) for index in range(3))
    )


@torch.no_grad()
def _constant_action_return(
    model: VQ2InformedDreamer,
    starts: RSSMState,
    *,
    channel: int,
    value: float,
    horizon: int,
    gamma: float,
    effort_weights: torch.Tensor,
    deterministic_latent: bool,
    seed: int,
    reward_source: str,
    control_frequency_hz: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return decoder task reward for one complete constant CTBR vector."""

    torch.manual_seed(seed)
    if starts.deterministic.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    state = starts.detached()
    action = torch.zeros(
        starts.deterministic.shape[0],
        model.action_size,
        device=starts.deterministic.device,
        dtype=starts.deterministic.dtype,
    )
    action[:, channel] = value
    total = torch.zeros(action.shape[0], device=action.device)
    progress_total = torch.zeros_like(total)
    rate_total = torch.zeros_like(total)
    gate_total = torch.zeros_like(total)
    crossing_count = torch.zeros_like(total)
    for step in range(horizon):
        current_feature = state.features
        state = model.rssm.imagine_step(
            state, action, deterministic_latent=deterministic_latent
        )
        if reward_source == "learned":
            reward = model.predict_reward(state.features, action)
        elif reward_source == "informed_decoder_skydreamer":
            terms = model.decoded_skydreamer_reward_terms(
                current_feature,
                state.features,
                control_frequency_hz=control_frequency_hz,
            )
            reward = terms.total
            progress_total += (gamma**step) * terms.progress
            rate_total += (gamma**step) * terms.rate_penalty
            gate_total += (gamma**step) * terms.gate
            crossing_count += terms.crossed.float()
        else:
            reward = model.decoded_progress_reward(current_feature, state.features)
            progress_total += (gamma**step) * reward
        reward -= (action.square() * effort_weights).sum(-1)
        total += (gamma**step) * reward
    diagnostics = {
        "mean_discounted_progress": float(progress_total.float().mean().cpu()),
        "mean_discounted_rate_penalty": float(rate_total.float().mean().cpu()),
        "mean_discounted_gate_reward": float(gate_total.float().mean().cpu()),
        "crossing_event_fraction": float(
            (crossing_count > 0.0).float().mean().cpu()
        ),
        "mean_crossing_events_per_start": float(crossing_count.mean().cpu()),
    }
    return total.float().cpu(), diagnostics


def _statistics(values: torch.Tensor, zero: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(torch.quantile(values, 0.10)),
        "p90": float(torch.quantile(values, 0.90)),
        "fraction_better_than_zero": float((values > zero).float().mean()),
    }


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = checkpoint.get("args")
    replay_state = checkpoint.get("replay")
    if not isinstance(training_args, dict) or not isinstance(replay_state, dict):
        raise RuntimeError("checkpoint lacks training arguments or replay state")
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
    model.load_state_dict(checkpoint["model"])
    model.eval()

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

    context = int(training_args["replay_context"])
    world_length = int(training_args["world_sequence_length"])
    if args.restore_checkpoint_rng:
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, dict) or "python" not in rng_state:
            raise RuntimeError("checkpoint lacks source-locked Python RNG state")
        rng = random.Random()
        rng.setstate(rng_state["python"])
    else:
        rng = random.Random(args.seed)
    starts = []
    for batch_index in range(args.batches):
        observation, action, _reward, _continuation = replay.sample(
            args.batch_size,
            context + world_length,
            device=device,
            rng=rng,
        )
        torch.manual_seed(args.seed + batch_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + batch_index)
        with torch.no_grad():
            initial = _burn_in_state(
                model, observation[:, :context], action[:, :context]
            )
            posterior = model.observe_sequence(
                observation[:, context:],
                action[:, context:],
                initial_state=initial,
                deterministic_latent=args.deterministic_posterior,
            )
            starts.append(
                _select_imagination_starts(
                    posterior.posterior,
                    mode="all",
                    count=args.starts_per_batch,
                )
            )
    selected = _concatenate(starts)
    effort_weights = torch.tensor(
        args.action_effort_weights,
        device=device,
        dtype=selected.deterministic.dtype,
    )
    with torch.no_grad():
        actor_distribution = model.actor_distribution(selected.features)
        actor_mean = model.deterministic_actor_action(actor_distribution)

    modes = {}
    for mode, deterministic_latent in (
        ("deterministic_prior", True),
        ("stochastic_prior", False),
    ):
        channels = []
        for channel_name in args.channels:
            channel = CHANNELS.index(channel_name)
            returns = {
                value: _constant_action_return(
                    model,
                    selected,
                    channel=channel,
                    value=value,
                    horizon=args.horizon,
                    gamma=args.gamma,
                    effort_weights=effort_weights,
                    deterministic_latent=deterministic_latent,
                    seed=args.seed + 10_000 * channel,
                    reward_source=args.reward_source,
                    control_frequency_hz=args.control_frequency_hz,
                )
                for value in ACTION_VALUES
            }
            zero = returns[0.0][0]
            rows = [
                {
                    "action_value": value,
                    **_statistics(returns[value][0], zero),
                    **returns[value][1],
                }
                for value in ACTION_VALUES
            ]
            channels.append(
                {
                    "channel": CHANNELS[channel],
                    "best_mean_value": max(rows, key=lambda row: row["mean"])[
                        "action_value"
                    ],
                    "rows": rows,
                }
            )
        modes[mode] = channels

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "replay_dir": str(args.replay_dir),
        "replay_state": observed,
        "contract": "sampled_all_time_replay_posterior_constant_ctbr_surface_v1",
        "device": str(device),
        "seed": args.seed,
        "restore_checkpoint_rng": args.restore_checkpoint_rng,
        "deterministic_posterior": args.deterministic_posterior,
        "channels": list(args.channels),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "starts_per_batch": args.starts_per_batch,
        "total_starts": int(selected.deterministic.shape[0]),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "action_effort_weights": list(args.action_effort_weights),
        "reward_source": args.reward_source,
        "control_frequency_hz": args.control_frequency_hz,
        "actor_mean_action": actor_mean.mean(0).float().cpu().tolist(),
        "actor_abs_max_action": actor_mean.abs().amax(0).float().cpu().tolist(),
        "modes": modes,
        "privileged_actor_input": False,
        "runtime_reward_dependency": False,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--starts-per-batch", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument(
        "--reward-source",
        choices=(
            "learned",
            "informed_decoder_progress",
            "informed_decoder_skydreamer",
        ),
        default="informed_decoder_progress",
    )
    parser.add_argument("--control-frequency-hz", type=float, default=64.0)
    parser.add_argument(
        "--action-effort-weights",
        type=float,
        nargs=4,
        default=(0.0, 0.05, 0.10, 0.5),
        metavar=("PITCH", "ROLL", "COLLECTIVE", "YAW"),
    )
    parser.add_argument("--seed", type=int, default=579)
    parser.add_argument(
        "--restore-checkpoint-rng",
        action="store_true",
        help="sample replay from the checkpoint's exact saved Python RNG state",
    )
    parser.add_argument(
        "--deterministic-posterior",
        action="store_true",
        help="match offline actor training's deterministic posterior starts",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=CHANNELS,
        default=CHANNELS,
        help="restrict the hardware-bounded constant-action audit",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("batches", "batch_size", "starts_per_batch", "horizon"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.starts_per_batch > args.batch_size * 64:
        parser.error("--starts-per-batch exceeds the fixed world sequence capacity")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0,1]")
    if args.control_frequency_hz <= 0.0:
        parser.error("--control-frequency-hz must be positive")
    if any(weight < 0.0 for weight in args.action_effort_weights):
        parser.error("--action-effort-weights cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
