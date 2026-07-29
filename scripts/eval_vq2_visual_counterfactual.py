#!/usr/bin/env python3
"""Measure causal visual dependence of a VQ2 Dreamer checkpoint offline.

The native ``drone_race_vision`` environment supplies the normal trajectory.
At every active policy decision, blank-mask and horizontal-flip actions are
computed from the same nonvisual observation values, previous action, and
normal recurrent prefix. Counterfactual actions never reach the plant. This
executable has no FlightSim or MAVLink dependency.
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
    MASK_SIZE,
    OBSERVATION_SCHEMA,
    VISUAL_HEIGHT,
    VISUAL_WIDTH,
)


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


def visual_variant(legal: torch.Tensor, kind: str) -> torch.Tensor:
    """Return a visual intervention while preserving every nonvisual value."""

    if legal.shape[-1] != LEGAL_OBS_SIZE:
        raise ValueError(
            f"expected legal observation width {LEGAL_OBS_SIZE}, got {legal.shape[-1]}"
        )
    changed = legal.clone()
    visual = changed[..., :MASK_SIZE]
    if kind == "blank":
        visual.zero_()
    elif kind == "horizontal_flip":
        flipped = visual.reshape(
            *visual.shape[:-1], VISUAL_HEIGHT, VISUAL_WIDTH
        ).flip(-1)
        visual.copy_(flipped.reshape_as(visual))
    else:
        raise ValueError(f"unsupported visual intervention: {kind}")
    return changed


@dataclass
class DeltaAccumulator:
    """Streaming action-difference summary for one visual intervention."""

    threshold: float
    samples: int = 0
    exact_changed_samples: int = 0
    threshold_changed_samples: int = 0
    absolute_sum: float = 0.0
    square_sum: float = 0.0
    absolute_max: float = 0.0
    channel_absolute_sum: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    channel_absolute_max: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )

    def update(self, normal: torch.Tensor, changed: torch.Tensor) -> None:
        if normal.shape != changed.shape or normal.ndim != 2 or normal.shape[-1] != 4:
            raise ValueError("action batches must have matching shape [samples, 4]")
        if normal.shape[0] == 0:
            return
        delta = (changed - normal).abs().detach().double().cpu()
        per_sample_max = delta.amax(-1)
        self.samples += int(delta.shape[0])
        self.exact_changed_samples += int((per_sample_max > 0.0).sum())
        self.threshold_changed_samples += int(
            (per_sample_max > self.threshold).sum()
        )
        self.absolute_sum += float(delta.sum())
        self.square_sum += float(delta.square().sum())
        self.absolute_max = max(self.absolute_max, float(delta.max()))
        channel_sum = delta.sum(0)
        channel_max = delta.amax(0)
        for channel in range(4):
            self.channel_absolute_sum[channel] += float(channel_sum[channel])
            self.channel_absolute_max[channel] = max(
                self.channel_absolute_max[channel], float(channel_max[channel])
            )

    def report(self) -> dict[str, float | int]:
        values = max(1, self.samples * 4)
        result: dict[str, float | int] = {
            "samples": self.samples,
            "exact_changed_samples": self.exact_changed_samples,
            "threshold": self.threshold,
            "threshold_changed_samples": self.threshold_changed_samples,
            "mean_absolute_action_delta": self.absolute_sum / values,
            "rms_action_delta": (self.square_sum / values) ** 0.5,
            "max_absolute_action_delta": self.absolute_max,
        }
        for channel in range(4):
            result[f"channel_{channel}_mean_absolute_delta"] = (
                self.channel_absolute_sum[channel] / max(1, self.samples)
            )
            result[f"channel_{channel}_max_absolute_delta"] = (
                self.channel_absolute_max[channel]
            )
        return result


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[VQ2InformedDreamer, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
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
        distributional_reward=bool(
            training_args.get("distributional_reward", False)
        ),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, training_args


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model, training_args = _load_model(args.checkpoint, device)
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
        vec.obs_ptr, total_agents * vec.obs_size
    ).reshape(total_agents, vec.obs_size)
    terminals = _tensor_from_pointer(vec.terminals_ptr, total_agents)
    actions_cpu = torch.zeros((total_agents, 4), dtype=torch.float32)
    episode_counts = torch.zeros(total_agents, dtype=torch.int64)
    pending_reset = torch.zeros(total_agents, dtype=torch.bool)
    state = model.rssm.initial(total_agents, device=device)
    previous_action = torch.zeros((total_agents, 4), device=device)
    all_deltas = {
        kind: DeltaAccumulator(args.change_threshold)
        for kind in ("blank", "horizontal_flip")
    }
    initial_deltas = {
        kind: DeltaAccumulator(args.change_threshold)
        for kind in ("blank", "horizontal_flip")
    }
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
                normal_action, next_state = model.policy_step(
                    legal, previous_action, state
                )
                changed_actions = {
                    kind: model.policy_step(
                        visual_variant(legal, kind), previous_action, state
                    )[0]
                    for kind in all_deltas
                }
            active = ~pending_reset
            if active.any():
                active_device = active.to(device)
                selected_normal = normal_action[active_device]
                for kind, changed_action in changed_actions.items():
                    selected_changed = changed_action[active_device]
                    all_deltas[kind].update(selected_normal, selected_changed)
                    if vector_step == 0:
                        initial_deltas[kind].update(
                            selected_normal, selected_changed
                        )
            if pending_reset.any():
                normal_action = torch.where(
                    pending_reset.to(device)[:, None],
                    torch.zeros_like(normal_action),
                    normal_action,
                )
            if not torch.isfinite(normal_action).all():
                raise RuntimeError("checkpoint produced a non-finite normal action")
            actions_cpu.copy_(normal_action.cpu())
            vec.cpu_step(actions_cpu.data_ptr())
            terminal_copy = terminals.clone().bool()
            episode_counts += terminal_copy.to(torch.int64)
            reset = terminal_copy.to(device) | pending_reset.to(device)
            state = _reset_state_where(model, next_state, reset)
            previous_action = torch.where(
                reset[:, None], torch.zeros_like(normal_action), normal_action
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
    metrics = {
        "schema": "vq2_visual_counterfactual_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": str(device),
        "episode_offset": args.episode_offset,
        "episodes_per_agent": args.episodes_per_agent,
        "total_episodes": int(episode_counts.sum()),
        "vector_steps": vector_step + 1,
        "normal_agent_steps": (vector_step + 1) * total_agents,
        "wall_seconds": elapsed,
        "normal_trajectory_only_reaches_plant": True,
        "counterfactual_recurrent_prefix": "normal",
        "counterfactual_nonvisual_values": "identical",
        "counterfactual_previous_action": "identical",
        "evaluation_start_contract": "uninterrupted_full_course",
        "gate_local_start_curriculum": int(
            config["env"]["gate_local_start_curriculum"]
        ),
        "mixed_start_curriculum": int(
            config["env"]["mixed_start_curriculum"]
        ),
        "counterfactuals": {
            kind: {
                "all_active_decisions": all_deltas[kind].report(),
                "initial_decision": initial_deltas[kind].report(),
            }
            for kind in all_deltas
        },
        "normal_environment": {
            key: float(value) for key, value in log.items()
        },
    }
    normal_environment = metrics["normal_environment"]
    if normal_environment.get("reset_count", 0.0) > 0.0:
        normal_environment["gate_local_reset_fraction"] = (
            normal_environment.get("gate_local_reset_count", 0.0)
            / normal_environment["reset_count"]
        )
    if normal_environment.get("gate_local_reset_fraction", 0.0) != 0.0:
        raise RuntimeError("full-start counterfactual observed a gate-local reset")
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    parser.add_argument("--change-threshold", type=float, default=1e-6)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.episodes_per_agent <= 0:
        parser.error("--episodes-per-agent must be positive")
    if args.episode_offset < 0:
        parser.error("--episode-offset cannot be negative")
    if args.max_vector_steps is not None and args.max_vector_steps <= 0:
        parser.error("--max-vector-steps must be positive")
    if args.change_threshold < 0:
        parser.error("--change-threshold cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
