#!/usr/bin/env python3
"""Collect paired exact-start CTBR channel sweeps into bounded VQ2 replay."""

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
from pufferlib.vq2_informed import ENV_OBS_SIZE  # noqa: E402
from scripts.analyze_vq2_informed_reward_channels import (  # noqa: E402
    ACTION_VALUES,
    CHANNELS,
)
from scripts.analyze_vq2_n540_reward_surface import _exact_config  # noqa: E402
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _tensor_from_pointer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> dict:
    records_per_channel = args.steps_per_channel + 1
    capacity = len(CHANNELS) * records_per_channel + len(CHANNELS) - 1
    replay = QuantizedSequenceReplay(
        capacity,
        64,
        storage_dir=args.replay_dir,
        resume=False,
    )
    group_size = 64 // len(ACTION_VALUES)
    total_valid = 0
    total_terminal = 0
    reward_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")
    channel_reports = []

    for channel, channel_name in enumerate(CHANNELS):
        vec = _C.create_vec(_exact_config(args.env_name), gpu=0)
        if vec.total_agents != 64 or vec.obs_size != ENV_OBS_SIZE:
            vec.close()
            raise RuntimeError("native causal collector ABI mismatch")
        vec.reset()
        observations = _tensor_from_pointer(
            vec.obs_ptr, vec.total_agents * vec.obs_size
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)
        terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)
        actions = torch.zeros((64, 4), dtype=torch.float32)
        for group, value in enumerate(ACTION_VALUES):
            actions[group * group_size : (group + 1) * group_size, channel] = value
        replay.add(
            observations.clone(),
            torch.zeros_like(actions),
            torch.zeros(64),
            torch.ones(64),
            torch.ones(64, dtype=torch.bool),
        )
        total_valid += 64
        channel_valid = 64
        pending_reset = torch.zeros(64, dtype=torch.bool)
        channel_terminal = 0
        boundary_observation = None
        try:
            for _step in range(args.steps_per_channel):
                vec.cpu_step(actions.data_ptr())
                reward_copy = rewards.clone()
                terminal_copy = terminals.clone().bool()
                valid = ~pending_reset
                continuation = 1.0 - terminal_copy.float()
                replay.add(
                    observations.clone(),
                    actions,
                    reward_copy,
                    continuation,
                    valid,
                )
                selected_reward = reward_copy[valid]
                if selected_reward.numel():
                    reward_sum += float(selected_reward.double().sum())
                    reward_min = min(reward_min, float(selected_reward.min()))
                    reward_max = max(reward_max, float(selected_reward.max()))
                    count = int(selected_reward.numel())
                    total_valid += count
                    channel_valid += count
                count_terminal = int(terminal_copy.sum())
                total_terminal += count_terminal
                channel_terminal += count_terminal
                pending_reset = terminal_copy
            log = dict(vec.log())
            boundary_observation = observations.clone()
        finally:
            vec.close()
        channel_reports.append(
            {
                "channel": channel_name,
                "valid_transitions": channel_valid,
                "terminal_events": channel_terminal,
                "env_gates_passed": float(log.get("gates_passed", 0.0)),
                "env_crash": float(log.get("crash", 0.0)),
            }
        )
        if channel + 1 < len(CHANNELS):
            assert boundary_observation is not None
            replay.add(
                boundary_observation,
                torch.zeros_like(actions),
                torch.zeros(64),
                torch.zeros(64),
                torch.zeros(64, dtype=torch.bool),
            )

    replay.flush()
    if replay.size != capacity:
        raise RuntimeError(f"causal replay size mismatch: {replay.size} != {capacity}")
    replay_hashes = {
        path.name: _sha256(path)
        for path in sorted(args.replay_dir.iterdir())
        if path.is_file()
    }
    report: dict[str, object] = {
        "schema": "vq2_exact_paired_ctbr_channel_replay_v2",
        "env_name": args.env_name,
        "channels": list(CHANNELS),
        "action_values": list(ACTION_VALUES),
        "steps_per_channel": args.steps_per_channel,
        "initial_records_per_channel": 1,
        "vector_records": replay.size,
        "agents": 64,
        "valid_transitions": total_valid,
        "terminal_events": total_terminal,
        "reward_mean": reward_sum / max(total_valid, 1),
        "reward_min": reward_min,
        "reward_max": reward_max,
        "randomization_disabled": True,
        "privileged_actor_input": False,
        "native_environment_created": len(CHANNELS),
        "flightsim_packets": 0,
        "channel_reports": channel_reports,
        "replay_state": replay.state_dict(),
        "replay_hashes": replay_hashes,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name", default="drone_race_vq2_informed_dreamer"
    )
    parser.add_argument("--steps-per-channel", type=int, default=64)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.steps_per_channel < 48:
        parser.error("--steps-per-channel must be at least 48")
    return args


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), indent=2, sort_keys=True))
