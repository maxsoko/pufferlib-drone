#!/usr/bin/env python3
"""Causally admit decoder-derived imagined progress for compact VQ2 Dreamer."""

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
    PITCHES,
    _exact_config,
)
from scripts.train_vq2_informed_dreamer import _tensor_from_pointer  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    vec = _C.create_vec(_exact_config(str(training_args["env_name"])), gpu=0)
    if vec.total_agents != 64 or vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError("native diagnostic ABI mismatch")
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    native_reward = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)

    actions = torch.zeros((64, 4), dtype=torch.float32)
    group_size = 64 // len(PITCHES)
    for group, pitch in enumerate(PITCHES):
        actions[group * group_size : (group + 1) * group_size, 0] = pitch
    device_actions = actions.to(device)

    with torch.no_grad():
        initial = model.rssm.initial(64, device=device)
        state, _ = model.rssm.observe_step(
            initial,
            torch.zeros_like(device_actions),
            observations[:, :LEGAL_OBS_SIZE].to(device),
            deterministic_latent=True,
        )
        posterior_state = state
        prior_return = torch.zeros(64, device=device)
        posterior_return = torch.zeros(64, device=device)
        native_return = torch.zeros(64)

    try:
        for step in range(args.horizon):
            discount = args.gamma**step
            with torch.no_grad():
                current_feature = state.features
                state = model.rssm.imagine_step(
                    state, device_actions, deterministic_latent=True
                )
                prior_return += discount * model.decoded_progress_reward(
                    current_feature, state.features
                )

            vec.cpu_step(actions.data_ptr())
            native_return += discount * native_reward.clone()

            with torch.no_grad():
                current_posterior_feature = posterior_state.features
                posterior_state, _ = model.rssm.observe_step(
                    posterior_state,
                    device_actions,
                    observations[:, :LEGAL_OBS_SIZE].to(device),
                    deterministic_latent=True,
                )
                posterior_return += discount * model.decoded_progress_reward(
                    current_posterior_feature, posterior_state.features
                )
    finally:
        vec.close()

    rows = []
    for group, pitch in enumerate(PITCHES):
        selected = slice(group * group_size, (group + 1) * group_size)
        rows.append(
            {
                "pitch": pitch,
                "native_discounted_return": float(native_return[selected].mean()),
                "prior_informed_discounted_return": float(
                    prior_return[selected].mean().cpu()
                ),
                "posterior_informed_discounted_return": float(
                    posterior_return[selected].mean().cpu()
                ),
            }
        )

    best_native = max(rows, key=lambda row: row["native_discounted_return"])
    best_prior = max(
        rows, key=lambda row: row["prior_informed_discounted_return"]
    )
    best_posterior = max(
        rows, key=lambda row: row["posterior_informed_discounted_return"]
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "progress_scale": 5.0,
        "contract": "exact_full_start_decoder_progress_reward",
        "randomization_disabled": True,
        "privileged_actor_input": False,
        "runtime_reward_dependency": False,
        "rows": rows,
        "best_native_pitch": best_native["pitch"],
        "best_prior_informed_pitch": best_prior["pitch"],
        "best_posterior_informed_pitch": best_posterior["pitch"],
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
