#!/usr/bin/env python3
"""Compare N540 latent and native short-horizon pitch returns.

This is training-only diagnosis. It disables every native randomization, starts
all agents from the same full-course state, applies constant complete CTBR
vectors, and compares discounted native reward with prior-only world-model
reward. Privileged values never enter the actor or imagined transition.
"""

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
from pufferlib.vq2_dreamer import VQ2InformedDreamer, symexp  # noqa: E402
from pufferlib.vq2_informed import (  # noqa: E402
    ACTION_SCHEMA,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    OBSERVATION_SCHEMA,
    configure_full_start_evaluation,
)
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    _load_config,
    _tensor_from_pointer,
)


PITCHES = (-0.50, -0.25, -0.10, 0.0, 0.04, 0.10, 0.25, 0.50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_config(name: str) -> dict:
    config = configure_full_start_evaluation(
        _load_config(name), episodes_per_agent=1, episode_offset=0
    )
    config["vec"]["total_agents"] = 64
    config["env"].update(
        {
            "evaluation_episode_limit": 0,
            "evaluation_episode_offset": 0,
            "gate_position_domain_randomize": 0,
            "course_geometry_scale_randomize": 0,
            "gate_radius_randomize": 0,
            "reset_position_noise_xy": 0.0,
            "reset_position_noise_z": 0.0,
            "visual_camera_roll_jitter_rad": 0.0,
            "visual_camera_pitch_jitter_rad": 0.0,
            "visual_camera_yaw_jitter_rad": 0.0,
            "visual_camera_dropout_prob": 0.0,
            "visual_edge_dropout_prob": 0.0,
            "visual_edge_corrupt_prob": 0.0,
            "visual_false_segments": 0,
            "visual_rolling_shutter_s": 0.0,
            "sitl_plant_domain_randomize": 0,
        }
    )
    return config


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
        distributional_reward=bool(
            training_args.get("distributional_reward", False)
        ),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    config = _exact_config(str(training_args["env_name"]))
    vec = _C.create_vec(config, gpu=0)
    if vec.total_agents != 64 or vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError("native diagnostic ABI mismatch")
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    rewards = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)
    terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)

    actions = torch.zeros((64, 4), dtype=torch.float32)
    group_size = 64 // len(PITCHES)
    for group, pitch in enumerate(PITCHES):
        actions[group * group_size : (group + 1) * group_size, 0] = pitch

    with torch.no_grad():
        legal = observations[:, :LEGAL_OBS_SIZE].to(device)
        previous_action = torch.zeros((64, 4), device=device)
        state = model.rssm.initial(64, device=device)
        distribution, state = model.policy_distribution_step(
            legal, previous_action, state
        )
        observed_state = state
        initial_policy_mean = torch.tanh(distribution.mean).cpu()
        imagined_actions = actions.to(device)
        imagined_return = torch.zeros(64, device=device)
        imagined_discount = torch.ones(64, device=device)
        imagined_continue_sum = torch.zeros(64, device=device)
        posterior_return = torch.zeros(64, device=device)
        posterior_kl_sum = torch.zeros(64, device=device)
        posterior_kl_max = torch.zeros(64, device=device)
        posterior_kl_below_free_nats = torch.zeros(64, device=device)

    native_return = torch.zeros(64, dtype=torch.float32)
    native_terminal = torch.zeros(64, dtype=torch.bool)
    try:
        for step in range(args.horizon):
            with torch.no_grad():
                state = model.rssm.imagine_step(
                    state, imagined_actions, deterministic_latent=True
                )
                predicted_reward = model.predict_reward(
                    state.features, imagined_actions
                )
                predicted_continue = torch.sigmoid(
                    model.continue_predictor(state.features).squeeze(-1)
                )
                imagined_return += imagined_discount * predicted_reward
                imagined_discount *= args.gamma * predicted_continue
                imagined_continue_sum += predicted_continue

            vec.cpu_step(actions.data_ptr())
            native_return += (args.gamma**step) * rewards.clone()
            native_terminal |= terminals.clone().bool()
            with torch.no_grad():
                next_legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                observed_state, prior_logits = model.rssm.observe_step(
                    observed_state,
                    imagined_actions,
                    next_legal,
                    deterministic_latent=True,
                )
                posterior_reward = model.predict_reward(
                    observed_state.features, imagined_actions
                )
                posterior_return += (args.gamma**step) * posterior_reward
                raw_kl = model._categorical_kl(
                    observed_state.logits, prior_logits
                )
                posterior_kl_sum += raw_kl
                posterior_kl_max = torch.maximum(posterior_kl_max, raw_kl)
                posterior_kl_below_free_nats += (raw_kl < 1.0).float()
    finally:
        vec.close()

    rows = []
    for group, pitch in enumerate(PITCHES):
        selected = slice(group * group_size, (group + 1) * group_size)
        rows.append(
            {
                "pitch": pitch,
                "native_discounted_return": float(native_return[selected].mean()),
                "imagined_discounted_return": float(
                    imagined_return[selected].mean().cpu()
                ),
                "imagined_mean_continue": float(
                    (imagined_continue_sum[selected] / args.horizon).mean().cpu()
                ),
                "posterior_discounted_return": float(
                    posterior_return[selected].mean().cpu()
                ),
                "posterior_prior_raw_kl_mean": float(
                    (posterior_kl_sum[selected] / args.horizon).mean().cpu()
                ),
                "posterior_prior_raw_kl_max": float(
                    posterior_kl_max[selected].max().cpu()
                ),
                "posterior_prior_kl_below_one_fraction": float(
                    (
                        posterior_kl_below_free_nats[selected] / args.horizon
                    ).mean().cpu()
                ),
                "native_terminal_rate": float(
                    native_terminal[selected].float().mean()
                ),
                "initial_policy_pitch_mean": float(
                    initial_policy_mean[selected, 0].mean()
                ),
            }
        )

    best_native = max(rows, key=lambda row: row["native_discounted_return"])
    best_imagined = max(rows, key=lambda row: row["imagined_discounted_return"])
    best_posterior = max(
        rows, key=lambda row: row["posterior_discounted_return"]
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "contract": "exact_full_start_constant_complete_ctbr",
        "randomization_disabled": True,
        "privileged_actor_input": False,
        "rows": rows,
        "best_native_pitch": best_native["pitch"],
        "best_imagined_pitch": best_imagined["pitch"],
        "best_posterior_pitch": best_posterior["pitch"],
        "pitch_ranking_agrees": best_native["pitch"] == best_imagined["pitch"],
        "posterior_pitch_ranking_agrees": (
            best_native["pitch"] == best_posterior["pitch"]
        ),
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
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0,1]")
    return args


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2, sort_keys=True))
