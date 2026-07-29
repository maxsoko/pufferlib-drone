#!/usr/bin/env python3
"""Deterministically evaluate a legal VQ2 Informed-Dreamer checkpoint natively.

This executable uses only the dedicated Puffer environment. It cannot connect
to FlightSim and never emits MAVLink traffic.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib import _C, pufferl  # noqa: E402
from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer  # noqa: E402
from pufferlib.vq2_informed import (  # noqa: E402
    ACTION_SCHEMA,
    configure_full_start_evaluation,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    OBSERVATION_SCHEMA,
    PRIVILEGED_NAMES,
    PRIVILEGED_SIZE,
)


@dataclass
class DecoderErrorAccumulator:
    """Training-only decoder diagnostics that never feed the actor or plant."""

    samples: int = 0
    absolute_sum: torch.Tensor = field(
        default_factory=lambda: torch.zeros(PRIVILEGED_SIZE, dtype=torch.float64)
    )
    square_sum: torch.Tensor = field(
        default_factory=lambda: torch.zeros(PRIVILEGED_SIZE, dtype=torch.float64)
    )

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.shape[-1] != PRIVILEGED_SIZE:
            raise ValueError("decoder prediction/target must match privileged ABI")
        if prediction.shape[0] == 0:
            return
        error = (prediction - target).detach().double().cpu()
        self.samples += int(error.shape[0])
        self.absolute_sum += error.abs().sum(0)
        self.square_sum += error.square().sum(0)

    def report(self) -> dict[str, float | int]:
        denominator = max(1, self.samples)
        result: dict[str, float | int] = {
            "decoder_samples": self.samples,
            "decoder_mean_absolute_error": float(
                self.absolute_sum.sum() / (denominator * PRIVILEGED_SIZE)
            ),
            "decoder_root_mean_square_error": float(
                torch.sqrt(
                    self.square_sum.sum() / (denominator * PRIVILEGED_SIZE)
                )
            ),
        }
        for index, name in enumerate(PRIVILEGED_NAMES):
            result[f"decoder_{name}_mean_absolute_error"] = float(
                self.absolute_sum[index] / denominator
            )
            result[f"decoder_{name}_root_mean_square_error"] = float(
                torch.sqrt(self.square_sum[index] / denominator)
            )
        return result


def _load_config(name: str) -> dict:
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(name)
    finally:
        sys.argv = saved_argv


def _tensor_from_pointer(pointer: int, length: int) -> torch.Tensor:
    return torch.frombuffer(
        (ctypes.c_float * length).from_address(pointer), dtype=torch.float32
    )


def _reset_state_where(
    model: VQ2InformedDreamer,
    state: RSSMState,
    reset: torch.Tensor,
) -> RSSMState:
    initial = model.rssm.initial(
        reset.shape[0], device=reset.device, dtype=state.deterministic.dtype
    )
    values = []
    for current, fresh in zip(state, initial, strict=True):
        mask = reset.bool().reshape(reset.shape[0], *([1] * (current.ndim - 1)))
        values.append(torch.where(mask, fresh, current))
    return RSSMState(*values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("legal_observation_size") != LEGAL_OBS_SIZE:
        raise RuntimeError("checkpoint legal-observation ABI mismatch")
    if checkpoint.get("environment_observation_size") != ENV_OBS_SIZE:
        raise RuntimeError("checkpoint environment-observation ABI mismatch")
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
        actor_distribution_mode=str(
            training_args.get("actor_distribution_mode", "legacy_tanh_normal")
        ),
        distributional_reward=bool(
            training_args.get("distributional_reward", False)
        ),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    env_name = args.env_name or str(training_args["env_name"])
    config = configure_full_start_evaluation(
        _load_config(env_name),
        episodes_per_agent=args.episodes_per_agent,
        episode_offset=args.episode_offset,
    )
    vec = _C.create_vec(config, gpu=0)
    if vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError(
            f"drone_race_vision ABI mismatch: expected {ENV_OBS_SIZE}, got {vec.obs_size}"
        )
    vec.reset()
    total_agents = vec.total_agents
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)
    actions_cpu = torch.zeros((vec.total_agents, 4), dtype=torch.float32)
    episode_counts = torch.zeros(vec.total_agents, dtype=torch.int64)
    pending_reset = torch.zeros(vec.total_agents, dtype=torch.bool)
    state = model.rssm.initial(vec.total_agents, device=device)
    previous_action = torch.zeros((vec.total_agents, 4), device=device)
    action_sum = torch.zeros(4, dtype=torch.float64)
    action_abs_max = torch.zeros(4, dtype=torch.float32)
    action_count = 0
    decoder_errors = DecoderErrorAccumulator()
    max_steps = args.max_vector_steps
    if max_steps is None:
        max_steps = (
            args.episodes_per_agent * (int(config["env"]["max_steps"]) + 2) + 16
        )
    started = time.perf_counter()

    try:
        for vector_step in range(max_steps):
            legal = observations[:, :LEGAL_OBS_SIZE].to(device)
            with torch.no_grad():
                action, next_state = model.policy_step(
                    legal, previous_action, state
                )
                privileged_prediction = model.privileged_decoder(
                    next_state.features
                )
            if pending_reset.any():
                action = torch.where(
                    pending_reset.to(device)[:, None],
                    torch.zeros_like(action),
                    action,
                )
            if not torch.isfinite(action).all():
                raise RuntimeError("checkpoint produced a non-finite action")
            actions_cpu.copy_(action.cpu())
            vec.cpu_step(actions_cpu.data_ptr())
            terminal_copy = terminals.clone().bool()
            episode_counts += terminal_copy.to(torch.int64)
            active_action = ~pending_reset
            if active_action.any():
                selected = actions_cpu[active_action]
                action_sum += selected.double().sum(0)
                action_abs_max = torch.maximum(
                    action_abs_max, selected.abs().amax(0)
                )
                action_count += int(selected.shape[0])
                decoder_errors.update(
                    privileged_prediction[active_action.to(device)],
                    observations[
                        active_action, LEGAL_OBS_SIZE:
                    ].to(device),
                )

            reset = terminal_copy.to(device) | pending_reset.to(device)
            state = _reset_state_where(model, next_state, reset)
            previous_action = torch.where(
                reset[:, None], torch.zeros_like(action), action
            )
            pending_reset = terminal_copy
            if int(episode_counts.min()) >= args.episodes_per_agent:
                break
        else:
            raise RuntimeError(
                f"evaluation did not finish within {max_steps} vector steps"
            )
        log = dict(vec.log())
    finally:
        vec.close()

    elapsed = time.perf_counter() - started
    metrics: dict[str, float | int | str] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "episodes_per_agent": args.episodes_per_agent,
        "episode_offset": args.episode_offset,
        "total_episodes": int(episode_counts.sum()),
        "vector_steps": vector_step + 1,
        "agent_steps": (vector_step + 1) * total_agents,
        "wall_seconds": elapsed,
        "agent_steps_per_second": (
            (vector_step + 1) * total_agents / max(elapsed, 1e-9)
        ),
        "action_count": action_count,
        "evaluation_start_contract": "uninterrupted_full_course",
        "gate_local_start_curriculum": int(
            config["env"]["gate_local_start_curriculum"]
        ),
        "mixed_start_curriculum": int(
            config["env"]["mixed_start_curriculum"]
        ),
    }
    if action_count:
        for channel in range(4):
            metrics[f"action_{channel}_mean"] = float(
                action_sum[channel] / action_count
            )
            metrics[f"action_{channel}_abs_max"] = float(action_abs_max[channel])
    metrics.update({f"env_{key}": float(value) for key, value in log.items()})
    if log.get("reset_count", 0.0) > 0.0:
        metrics["env_gate_local_reset_fraction"] = (
            float(log.get("gate_local_reset_count", 0.0))
            / float(log["reset_count"])
        )
    metrics.update(decoder_errors.report())
    if metrics.get("env_gate_local_reset_fraction", 0.0) != 0.0:
        raise RuntimeError("full-start evaluation observed a gate-local reset")
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-name")
    parser.add_argument("--episodes-per-agent", type=int, default=1)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-vector-steps", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.episodes_per_agent <= 0:
        parser.error("--episodes-per-agent must be positive")
    if args.episode_offset < 0:
        parser.error("--episode-offset cannot be negative")
    if args.max_vector_steps is not None and args.max_vector_steps <= 0:
        parser.error("--max-vector-steps must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
