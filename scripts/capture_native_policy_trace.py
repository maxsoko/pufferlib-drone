#!/usr/bin/env python3
"""Capture exact native recurrent rollout tensors for one deterministic policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from eval_drone_race_checkpoint import _deterministic_checkpoint, _load_config
except ModuleNotFoundError:  # Imported as scripts.capture_native_policy_trace.
    from scripts.eval_drone_race_checkpoint import (
        _deterministic_checkpoint,
        _load_config,
    )


OFFICIAL_FIT_OVERRIDES = (
    "--env.num-gates", "4",
    "--env.course-geometry-scale", "1",
    "--env.course-geometry-scale-randomize", "0",
    "--env.gate-radius", "0.75",
    "--env.gate-radius-randomize", "0",
    "--env.sitl-gate-transition-min-forward-speed-randomize", "0",
    "--env.sitl-gate-transition-from-gate-index", "2",
    "--env.sitl-gate-transition-max-accel-m-s2", "0",
    "--env.sitl-gate-transition-preserve-velocity-direction", "0",
    "--env.sitl-gate-transition-delay-steps", "0",
    "--env.sitl-gate-obs-dropout-from-index", "1",
    "--env.sitl-gate-obs-dropout-range-m", "0",
    "--env.gate-position-domain-randomize", "0",
)


def build_capture_overrides(
    *,
    total_agents: int,
    horizon: int,
    floor: float,
    config_overrides: tuple[str, ...] = (),
) -> list[str]:
    """Build native trace overrides, with caller settings taking precedence."""
    if len(config_overrides) % 2:
        raise ValueError("config_overrides must contain key/value pairs")
    return [
        "--vec.total-agents", str(total_agents),
        "--vec.num-buffers", "1",
        "--vec.num-threads", "1",
        "--train.horizon", str(horizon),
        "--train.minibatch-size", str(total_agents * horizon),
        *OFFICIAL_FIT_OVERRIDES,
        "--env.sitl-gate-transition-min-forward-speed", repr(float(floor)),
        *config_overrides,
    ]


def parse_config_overrides(values: list[str]) -> tuple[str, ...]:
    flattened: list[str] = []
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.startswith("--") or not value:
            raise ValueError(
                "config overrides must use --section.key=value syntax"
            )
        flattened.extend((key, value))
    return tuple(flattened)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble_trace_blocks(
    blocks: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not blocks:
        raise ValueError("at least one rollout block is required")
    required = {"observations", "actions", "rewards", "terminals", "initial_states"}
    for block in blocks:
        missing = required.difference(block)
        if missing:
            raise ValueError(f"rollout block is missing {sorted(missing)}")

    observations = np.concatenate([b["observations"] for b in blocks], axis=0)
    actions = np.concatenate([b["actions"] for b in blocks], axis=0)
    rewards = np.concatenate([b["rewards"] for b in blocks], axis=0)
    terminals = np.concatenate([b["terminals"] for b in blocks], axis=0)
    rollout_initial_states = np.stack([b["initial_states"] for b in blocks])
    if observations.shape[:2] != actions.shape[:2] or terminals.shape != observations.shape[:2]:
        raise ValueError("rollout tensor time/agent dimensions do not agree")

    time_steps, agents = terminals.shape
    lengths = np.full(agents, time_steps, dtype=np.int32)
    valid = np.zeros_like(terminals, dtype=np.bool_)
    for agent in range(agents):
        done = np.flatnonzero(terminals[:, agent] > 0.5)
        if done.size:
            lengths[agent] = int(done[0]) + 1
        valid[: lengths[agent], agent] = True

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "valid": valid,
        "lengths": lengths,
        "rollout_initial_states": rollout_initial_states,
    }


def verify_trace_replay(
    checkpoint: Path,
    trace_path: Path,
    *,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
    action_atol: float = 5e-5,
    state_atol: float = 5e-5,
) -> dict[str, Any]:
    """Replay a native trace and verify action/state/reset parity."""
    try:
        from policy_callable_checkpoint import CheckpointPolicy
    except ModuleNotFoundError:
        from scripts.policy_callable_checkpoint import CheckpointPolicy

    with np.load(trace_path, allow_pickle=False) as trace:
        observations = np.asarray(trace["observations"], dtype=np.float32)
        expected_actions = np.asarray(trace["actions"], dtype=np.float32)
        terminals = np.asarray(trace["terminals"], dtype=np.float32)
        initial_states = np.asarray(
            trace["rollout_initial_states"], dtype=np.float32
        )
        metadata = json.loads(str(trace["metadata"]))

    if metadata["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("trace checkpoint hash does not match replay checkpoint")
    horizon = int(metadata["horizon"])
    if observations.shape[:2] != expected_actions.shape[:2]:
        raise ValueError("trace observation/action time and agent axes disagree")
    if terminals.shape != observations.shape[:2]:
        raise ValueError("trace terminal shape does not match observations")
    expected_blocks = (observations.shape[0] + horizon - 1) // horizon
    expected_state_shape = (
        expected_blocks,
        num_layers,
        observations.shape[1],
        hidden_dim,
    )
    if initial_states.shape != expected_state_shape:
        raise ValueError(
            f"expected rollout initial states {expected_state_shape}, "
            f"got {initial_states.shape}"
        )

    action_square_error = 0.0
    state_square_error = 0.0
    action_values = 0
    state_values = 0
    action_max_error = 0.0
    state_max_error = 0.0
    reset_count = 0
    raw_action_saturation_values = int(
        np.count_nonzero((expected_actions < -1.0) | (expected_actions > 1.0))
    )
    for agent in range(observations.shape[1]):
        policy = CheckpointPolicy.load(
            str(checkpoint),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            layout_precision_bytes=layout_precision_bytes,
        )
        for step in range(observations.shape[0]):
            if step % horizon == 0:
                block = step // horizon
                state_delta = policy.state - initial_states[block, :, agent, :]
                state_max_error = max(
                    state_max_error, float(np.max(np.abs(state_delta)))
                )
                state_square_error += float(
                    np.square(state_delta, dtype=np.float64).sum()
                )
                state_values += state_delta.size
            # Native inference resets an agent before the action whose current
            # observation carries the terminal flag.
            if terminals[step, agent] > 0.5:
                policy.reset_state()
                reset_count += 1
            replayed = np.asarray(
                policy.infer(observations[step, agent]), dtype=np.float32
            )
            # The native rollout buffer retains the unsquashed Gaussian mean;
            # both the environment and CheckpointPolicy clamp it to the
            # official [-1, 1] action contract before applying commands.
            operational_expected = np.clip(
                expected_actions[step, agent], -1.0, 1.0
            )
            action_delta = replayed - operational_expected
            action_max_error = max(
                action_max_error, float(np.max(np.abs(action_delta)))
            )
            action_square_error += float(
                np.square(action_delta, dtype=np.float64).sum()
            )
            action_values += action_delta.size

    action_rms_error = float(np.sqrt(action_square_error / action_values))
    state_rms_error = float(np.sqrt(state_square_error / state_values))
    passed = action_max_error <= action_atol and state_max_error <= state_atol
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "time_steps": int(observations.shape[0]),
        "agents": int(observations.shape[1]),
        "rollout_blocks": int(initial_states.shape[0]),
        "terminal_resets_replayed": reset_count,
        "raw_action_saturation_values": raw_action_saturation_values,
        "action_rms_error": action_rms_error,
        "action_max_error": action_max_error,
        "state_rms_error": state_rms_error,
        "state_max_error": state_max_error,
        "action_atol": action_atol,
        "state_atol": state_atol,
        "passed": passed,
    }


def capture_native_trace(
    checkpoint: Path,
    output: Path,
    *,
    floor: float,
    total_agents: int = 8,
    horizon: int = 32,
    episode_offset: int = 0,
    max_rollouts: int = 64,
    env_name: str = "drone_race_full_policy_official_fit",
    layout_precision_bytes: int = 4,
    config_overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    if total_agents <= 0 or horizon <= 0 or max_rollouts <= 0:
        raise ValueError("agents, horizon, and max rollouts must be positive")
    if episode_offset < 0:
        raise ValueError("episode offset must be nonnegative")

    from pufferlib import _C, pufferl

    if not hasattr(_C, "rollout_trace"):
        raise RuntimeError(
            "native backend lacks rollout_trace; rebuild with "
            "`CUDA_HOME=/usr/local/cuda NVCC_ARCH=sm_86 bash build.sh drone_race --float`"
        )
    native_precision = int(getattr(_C, "precision_bytes", 0))
    if native_precision != layout_precision_bytes:
        raise ValueError(
            f"native/layout precision mismatch: {native_precision}/{layout_precision_bytes}"
        )

    overrides = build_capture_overrides(
        total_agents=total_agents,
        horizon=horizon,
        floor=floor,
        config_overrides=config_overrides,
    )
    cfg = _load_config(pufferl, env_name, overrides)
    cfg["env"]["evaluation_episode_limit"] = 1
    cfg["env"]["evaluation_episode_offset"] = int(episode_offset)

    deterministic_path = _deterministic_checkpoint(
        checkpoint,
        input_dim=32,
        hidden_dim=int(cfg["policy"]["hidden_size"]),
        num_actions=4,
        layout_precision_bytes=layout_precision_bytes,
    )
    runner = _C.create_pufferl(cfg)
    blocks: list[dict[str, np.ndarray]] = []
    flat_log: dict[str, float] = {}
    try:
        _C.load_weights(runner, deterministic_path)
        for _ in range(max_rollouts):
            _C.rollouts(runner)
            native_block = _C.rollout_trace(runner)
            blocks.append({key: np.asarray(value).copy() for key, value in native_block.items()})
            nested_log = _C.eval_log(runner)
            flat_log = dict(pufferl.unroll_nested_dict(nested_log))
            if float(flat_log.get("env/n", 0.0)) >= total_agents:
                break
    finally:
        _C.close(runner)
        Path(deterministic_path).unlink(missing_ok=True)

    if float(flat_log.get("env/n", 0.0)) != total_agents:
        raise RuntimeError(
            f"trace episode mismatch: expected {total_agents}, got {flat_log.get('env/n', 0.0)}"
        )
    assembled = assemble_trace_blocks(blocks)
    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "floor": float(floor),
        "total_agents": total_agents,
        "horizon": horizon,
        "episode_offset": episode_offset,
        "rollout_blocks": len(blocks),
        "native_precision_bytes": native_precision,
        "checkpoint_layout_precision_bytes": layout_precision_bytes,
        "hidden_state_granularity": "exact_native_rollout_start",
        "config_overrides": overrides,
        "metrics": flat_log,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **assembled,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--total-agents", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-rollouts", type=int, default=64)
    parser.add_argument("--env-name", default="drone_race_full_policy_official_fit")
    parser.add_argument("--layout-precision-bytes", type=int, choices=(2, 4), default=4)
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="append one PufferLib config override; later values take precedence",
    )
    args = parser.parse_args()
    metadata = capture_native_trace(
        args.checkpoint,
        args.output,
        floor=args.floor,
        total_agents=args.total_agents,
        horizon=args.horizon,
        episode_offset=args.episode_offset,
        max_rollouts=args.max_rollouts,
        env_name=args.env_name,
        layout_precision_bytes=args.layout_precision_bytes,
        config_overrides=parse_config_overrides(args.config_override),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
