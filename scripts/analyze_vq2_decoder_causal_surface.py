#!/usr/bin/env python3
"""Separate learned VQ2 dynamics direction from reward-head calibration.

This training-only diagnostic applies paired constant CTBR pitch commands from
one exact native start. It compares true active-gate range progress with range
progress decoded from prior-only and posterior RSSM states. Privileged values
are decoder targets and diagnostics only; they never enter the actor or RSSM.
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
from pufferlib.vq2_dreamer import VQ2InformedDreamer  # noqa: E402
from pufferlib.vq2_informed import (  # noqa: E402
    ACTION_SCHEMA,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    OBSERVATION_SCHEMA,
    split_environment_observation,
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


def _gate_range(normalized_privileged: torch.Tensor) -> torch.Tensor:
    """Decode tanh-normalized active-gate body vector to range in metres."""

    gate_body = normalized_privileged[..., 3:6].clamp(-0.999999, 0.999999)
    return (10.0 * torch.atanh(gate_body)).square().sum(-1).sqrt()


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

    config = _exact_config(str(training_args["env_name"]))
    vec = _C.create_vec(config, gpu=0)
    if vec.total_agents != 64 or vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError("native diagnostic ABI mismatch")
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)

    actions = torch.zeros((64, 4), dtype=torch.float32)
    group_size = 64 // len(PITCHES)
    for group, pitch in enumerate(PITCHES):
        actions[group * group_size : (group + 1) * group_size, 0] = pitch
    device_actions = actions.to(device)

    with torch.no_grad():
        legal = observations[:, :LEGAL_OBS_SIZE].to(device)
        state = model.rssm.initial(64, device=device)
        state, _ = model.rssm.observe_step(
            state,
            torch.zeros_like(device_actions),
            legal,
            deterministic_latent=True,
        )
        posterior_state = state
        prior_range = _gate_range(model.privileged_decoder(state.features))
        posterior_range = prior_range.clone()
        native_range = _gate_range(
            split_environment_observation(observations).privileged
        )
        initial_prior_range = prior_range.clone()
        initial_posterior_range = posterior_range.clone()
        initial_native_range = native_range.clone()
        prior_progress = torch.zeros(64, device=device)
        posterior_progress = torch.zeros(64, device=device)
        native_progress = torch.zeros(64)

    try:
        for step in range(args.horizon):
            discount = args.gamma**step
            with torch.no_grad():
                state = model.rssm.imagine_step(
                    state, device_actions, deterministic_latent=True
                )
                next_prior_range = _gate_range(
                    model.privileged_decoder(state.features)
                )
                prior_progress += discount * (prior_range - next_prior_range)
                prior_range = next_prior_range

            vec.cpu_step(actions.data_ptr())
            next_native_range = _gate_range(
                split_environment_observation(observations).privileged
            )
            native_progress += discount * (native_range - next_native_range)
            native_range = next_native_range

            with torch.no_grad():
                posterior_state, _ = model.rssm.observe_step(
                    posterior_state,
                    device_actions,
                    observations[:, :LEGAL_OBS_SIZE].to(device),
                    deterministic_latent=True,
                )
                next_posterior_range = _gate_range(
                    model.privileged_decoder(posterior_state.features)
                )
                posterior_progress += discount * (
                    posterior_range - next_posterior_range
                )
                posterior_range = next_posterior_range
    finally:
        vec.close()

    rows = []
    for group, pitch in enumerate(PITCHES):
        selected = slice(group * group_size, (group + 1) * group_size)
        rows.append(
            {
                "pitch": pitch,
                "native_initial_range_m": float(initial_native_range[selected].mean()),
                "native_final_range_m": float(native_range[selected].mean()),
                "native_discounted_progress_m": float(native_progress[selected].mean()),
                "prior_initial_decoded_range_m": float(
                    initial_prior_range[selected].mean().cpu()
                ),
                "prior_final_decoded_range_m": float(prior_range[selected].mean().cpu()),
                "prior_discounted_decoded_progress_m": float(
                    prior_progress[selected].mean().cpu()
                ),
                "posterior_initial_decoded_range_m": float(
                    initial_posterior_range[selected].mean().cpu()
                ),
                "posterior_final_decoded_range_m": float(
                    posterior_range[selected].mean().cpu()
                ),
                "posterior_discounted_decoded_progress_m": float(
                    posterior_progress[selected].mean().cpu()
                ),
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "horizon": args.horizon,
        "gamma": args.gamma,
        "contract": "exact_full_start_constant_complete_ctbr_decoder_only",
        "randomization_disabled": True,
        "privileged_actor_input": False,
        "rows": rows,
        "best_native_pitch": max(
            rows, key=lambda row: row["native_discounted_progress_m"]
        )["pitch"],
        "best_prior_decoded_pitch": max(
            rows, key=lambda row: row["prior_discounted_decoded_progress_m"]
        )["pitch"],
        "best_posterior_decoded_pitch": max(
            rows, key=lambda row: row["posterior_discounted_decoded_progress_m"]
        )["pitch"],
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
