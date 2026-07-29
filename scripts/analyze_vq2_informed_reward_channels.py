#!/usr/bin/env python3
"""Compare native and informed reward causality for all four VQ2 CTBR axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib import _C  # noqa: E402
from pufferlib.vq2_dreamer import VQ2InformedDreamer  # noqa: E402
from pufferlib.vq2_informed import (  # noqa: E402
    ACTION_SCHEMA,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    OBSERVATION_SCHEMA,
)
from scripts.analyze_vq2_n540_reward_surface import (  # noqa: E402
    PITCHES as ACTION_VALUES,
    _exact_config,
)
from scripts.train_vq2_informed_dreamer import _tensor_from_pointer  # noqa: E402


CHANNELS = ("pitch", "roll", "collective", "yaw")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_channel(
    model: VQ2InformedDreamer,
    training_args: dict,
    *,
    channel: int,
    horizon: int,
    gamma: float,
    device: torch.device,
    reward_source: str,
    action_effort_weights: tuple[float, float, float, float],
) -> dict:
    vec = _C.create_vec(_exact_config(str(training_args["env_name"])), gpu=0)
    if vec.total_agents != 64 or vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError("native diagnostic ABI mismatch")
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    native_reward = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)
    terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)

    actions = torch.zeros((64, 4), dtype=torch.float32)
    group_size = 64 // len(ACTION_VALUES)
    for group, value in enumerate(ACTION_VALUES):
        actions[group * group_size : (group + 1) * group_size, channel] = value
    device_actions = actions.to(device)
    effort_weights = torch.tensor(action_effort_weights, device=device)
    with torch.no_grad():
        state = model.rssm.initial(64, device=device)
        state, _ = model.rssm.observe_step(
            state,
            torch.zeros_like(device_actions),
            observations[:, :LEGAL_OBS_SIZE].to(device),
            deterministic_latent=True,
        )
        posterior_state = state
        prior_return = torch.zeros(64, device=device)
        posterior_return = torch.zeros(64, device=device)
        native_return = torch.zeros(64)
        native_terminal = torch.zeros(64, dtype=torch.bool)

    try:
        for step in range(horizon):
            discount = gamma**step
            with torch.no_grad():
                current_feature = state.features
                state = model.rssm.imagine_step(
                    state, device_actions, deterministic_latent=True
                )
                if reward_source == "learned":
                    predicted_reward = model.predict_reward(
                        state.features, device_actions
                    )
                else:
                    predicted_reward = model.decoded_progress_reward(
                        current_feature, state.features
                    )
                predicted_reward -= (
                    device_actions.square() * effort_weights
                ).sum(-1)
                prior_return += discount * predicted_reward
            vec.cpu_step(actions.data_ptr())
            native_return += discount * native_reward.clone()
            native_terminal |= terminals.clone().bool()
            with torch.no_grad():
                current_feature = posterior_state.features
                posterior_state, _ = model.rssm.observe_step(
                    posterior_state,
                    device_actions,
                    observations[:, :LEGAL_OBS_SIZE].to(device),
                    deterministic_latent=True,
                )
                if reward_source == "learned":
                    predicted_reward = model.predict_reward(
                        posterior_state.features, device_actions
                    )
                else:
                    predicted_reward = model.decoded_progress_reward(
                        current_feature, posterior_state.features
                    )
                predicted_reward -= (
                    device_actions.square() * effort_weights
                ).sum(-1)
                posterior_return += discount * predicted_reward
    finally:
        vec.close()

    rows = []
    for group, value in enumerate(ACTION_VALUES):
        selected = slice(group * group_size, (group + 1) * group_size)
        rows.append(
            {
                "action_value": value,
                "native_discounted_return": float(native_return[selected].mean()),
                "prior_informed_discounted_return": float(
                    prior_return[selected].mean().cpu()
                ),
                "posterior_informed_discounted_return": float(
                    posterior_return[selected].mean().cpu()
                ),
                "native_terminal_rate": float(
                    native_terminal[selected].float().mean()
                ),
            }
        )
    return {
        "channel": CHANNELS[channel],
        "rows": rows,
        "best_native_value": max(
            rows, key=lambda row: row["native_discounted_return"]
        )["action_value"],
        "best_prior_informed_value": max(
            rows, key=lambda row: row["prior_informed_discounted_return"]
        )["action_value"],
        "best_posterior_informed_value": max(
            rows, key=lambda row: row["posterior_informed_discounted_return"]
        )["action_value"],
    }


def analyze(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("observation_schema") != OBSERVATION_SCHEMA:
        raise RuntimeError("checkpoint observation schema mismatch")
    if checkpoint.get("action_schema") != ACTION_SCHEMA:
        raise RuntimeError("checkpoint action schema mismatch")
    training_args = checkpoint["args"]
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    channels = [
        _one_channel(
            model,
            training_args,
            channel=channel,
            horizon=args.horizon,
            gamma=args.gamma,
            device=device,
            reward_source=args.reward_source,
            action_effort_weights=tuple(args.action_effort_weights),
        )
        for channel in range(4)
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "progress_scale": 5.0,
        "contract": "exact_full_start_complete_ctbr_channel_surfaces",
        "randomization_disabled": True,
        "privileged_actor_input": False,
        "runtime_reward_dependency": False,
        "reward_source": args.reward_source,
        "action_effort_weights": list(args.action_effort_weights),
        "learned_reward_target_scale": float(
            checkpoint.get("offline_reward_target_scale", 1.0)
        ),
        "channels": channels,
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
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument(
        "--reward-source",
        choices=("informed_decoder_progress", "learned"),
        default="informed_decoder_progress",
    )
    parser.add_argument(
        "--action-effort-weights",
        type=float,
        nargs=4,
        metavar=("PITCH", "ROLL", "COLLECTIVE", "YAW"),
        default=(0.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if any(weight < 0.0 for weight in args.action_effort_weights):
        parser.error("--action-effort-weights cannot be negative")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0,1]")
    return args


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
