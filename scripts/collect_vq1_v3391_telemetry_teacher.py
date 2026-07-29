#!/usr/bin/env python3
"""Generate exact six-gate VQ1 v3391 velocity-teacher demonstrations.

The rollout mirrors the native interface-3 plant and observation contract in
``ocean/drone_race/drone_race.c``. Each float32 record contains 32 observable
policy inputs, four normalized teacher actions, and an episode-reset flag.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1
HALF_FOV_RAD = math.pi / 4.0
GATE_INNER_WIDTH_M = 1.5


def load_course(config_path: Path) -> tuple[np.ndarray, dict[str, float]]:
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(config_path)
    env = parser["env"]
    num_gates = env.getint("num_gates")
    gates = np.asarray(
        [
            [
                env.getfloat(f"gate{gate}_x"),
                env.getfloat(f"gate{gate}_y"),
                env.getfloat(f"gate{gate}_z"),
            ]
            for gate in range(num_gates)
        ],
        dtype=np.float64,
    )
    settings = {
        "dt": env.getfloat("dt"),
        "gate_radius": env.getfloat("gate_radius"),
        "max_steps": float(env.getint("max_steps")),
        "time_limit_seconds": env.getfloat("time_limit_seconds"),
        "max_cmd_forward": env.getfloat("max_cmd_forward"),
        "max_cmd_lateral": env.getfloat("max_cmd_lateral"),
        "max_cmd_vertical": env.getfloat("max_cmd_vertical"),
        "response_tau_s": env.getfloat("telemetry_velocity_response_tau_s"),
        "max_accel_m_s2": env.getfloat("telemetry_velocity_max_accel_m_s2"),
        "teacher_speed_m_s": env.getfloat("telemetry_velocity_teacher_speed_m_s"),
        "gate_index_denominator": env.getfloat("observable_gate_index_denominator"),
    }
    return gates, settings


def build_observation(
    position: np.ndarray,
    velocity: np.ndarray,
    gate_index: int,
    elapsed_time: float,
    last_action: np.ndarray,
    gates: np.ndarray,
    settings: dict[str, float],
) -> np.ndarray:
    observation = np.zeros(OBSERVATIONS, dtype=np.float32)
    observation[0] = np.tanh(-velocity[0] * 0.2)
    observation[1] = np.tanh(-velocity[1] / 3.0)
    observation[2] = np.tanh(velocity[2] / 3.0)
    observation[6] = 1.0

    if gate_index < len(gates):
        relative = gates[gate_index] - position
        forward, right, relative_up = relative
        down = -relative_up
        distance = max(float(np.linalg.norm(relative)), 1e-3)
        yaw_error = math.atan2(right, max(forward, 1e-4))
        elevation = math.atan2(-down, max(forward, 1e-4))
        observation[10] = 1.0
        observation[11] = np.tanh(forward * 0.1)
        observation[12] = np.tanh(right * 0.2)
        observation[13] = np.tanh(down * 0.2)
        observation[14] = yaw_error / HALF_FOV_RAD
        observation[15] = elevation / HALF_FOV_RAD
        observation[16] = np.clip(GATE_INNER_WIDTH_M / distance, 0.0, 1.0)
        yaw_score = 1.0 - abs(yaw_error) / HALF_FOV_RAD
        pitch_score = 1.0 - abs(elevation) / HALF_FOV_RAD
        observation[17] = np.clip(0.5 * (yaw_score + pitch_score), 0.0, 1.0)

    observation[18] = np.clip(
        elapsed_time / settings["time_limit_seconds"], 0.0, 1.0
    )
    observation[19:23] = last_action
    observation[23] = np.clip(
        gate_index / settings["gate_index_denominator"], 0.0, 1.0
    )
    if gate_index < 6:
        observation[24 + gate_index] = 1.0
    return np.clip(observation, -1.0, 1.0)


def teacher_action(
    position: np.ndarray,
    gate_index: int,
    gates: np.ndarray,
    settings: dict[str, float],
) -> np.ndarray:
    relative = gates[gate_index] - position
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-5:
        return np.zeros(ACTIONS, dtype=np.float32)
    speed = min(settings["teacher_speed_m_s"], distance * 2.0)
    desired = relative * (speed / distance)
    action = np.asarray(
        [
            desired[0] / settings["max_cmd_forward"],
            desired[1] / settings["max_cmd_lateral"],
            -desired[2] / settings["max_cmd_vertical"],
            0.0,
        ],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0)


def crossing_radial(
    previous: np.ndarray, current: np.ndarray, gate: np.ndarray
) -> float | None:
    previous_distance = previous[0] - gate[0]
    current_distance = current[0] - gate[0]
    if not (previous_distance <= 0.0 and current_distance >= 0.0):
        return None
    denominator = previous_distance - current_distance
    if abs(denominator) < 1e-8 or current[0] - previous[0] <= 0.01:
        return None
    fraction = np.clip(previous_distance / denominator, 0.0, 1.0)
    hit = previous + (current - previous) * fraction
    return float(np.linalg.norm(hit[1:] - gate[1:]))


def rollout_episode(
    rng: np.random.Generator,
    gates: np.ndarray,
    settings: dict[str, float],
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    position = np.asarray(
        [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)],
        dtype=np.float64,
    )
    velocity = np.zeros(3, dtype=np.float64)
    last_action = np.zeros(ACTIONS, dtype=np.float32)
    gate_index = 0
    elapsed_time = 0.0
    records: list[np.ndarray] = []
    maximum_radial = 0.0
    dt = settings["dt"]
    alpha = 1.0 - math.exp(-dt / max(settings["response_tau_s"], dt))
    maximum_step = settings["max_accel_m_s2"] * dt

    for step in range(int(settings["max_steps"])):
        observation = build_observation(
            position, velocity, gate_index, elapsed_time, last_action, gates, settings
        )
        action = teacher_action(position, gate_index, gates, settings)
        record = np.zeros(RECORD_WIDTH, dtype=np.float32)
        record[:OBSERVATIONS] = observation
        record[OBSERVATIONS:OBSERVATIONS + ACTIONS] = action
        record[-1] = float(step == 0)
        records.append(record)

        desired_velocity = np.asarray(
            [
                action[0] * settings["max_cmd_forward"],
                action[1] * settings["max_cmd_lateral"],
                -action[2] * settings["max_cmd_vertical"],
            ],
            dtype=np.float64,
        )
        velocity += np.clip((desired_velocity - velocity) * alpha, -maximum_step, maximum_step)
        previous_position = position.copy()
        position += velocity * dt
        elapsed_time += dt
        last_action = action

        radial = crossing_radial(previous_position, position, gates[gate_index])
        if radial is not None:
            maximum_radial = max(maximum_radial, radial)
            if radial > settings["gate_radius"]:
                break
            gate_index += 1
            if gate_index == len(gates):
                return np.stack(records), {
                    "success": True,
                    "gates": gate_index,
                    "steps": step + 1,
                    "completion_time_s": elapsed_time,
                    "maximum_crossing_radial_m": maximum_radial,
                }

    return np.stack(records), {
        "success": False,
        "gates": gate_index,
        "steps": len(records),
        "completion_time_s": elapsed_time,
        "maximum_crossing_radial_m": maximum_radial,
    }


def collect_dataset(
    config_path: Path, episodes: int, seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    gates, settings = load_course(config_path)
    rng = np.random.default_rng(seed)
    trajectories: list[np.ndarray] = []
    episode_reports: list[dict[str, float | int | bool]] = []
    for _ in range(episodes):
        trajectory, report = rollout_episode(rng, gates, settings)
        trajectories.append(trajectory)
        episode_reports.append(report)
    failures = [report for report in episode_reports if not report["success"]]
    if failures:
        raise RuntimeError(f"analytic teacher failed {len(failures)}/{episodes} episodes")
    records = np.concatenate(trajectories)
    report = {
        "config": str(config_path),
        "seed": seed,
        "episodes": episodes,
        "records": int(len(records)),
        "record_width": RECORD_WIDTH,
        "success_rate": 1.0,
        "mean_steps": float(np.mean([item["steps"] for item in episode_reports])),
        "mean_completion_time_s": float(
            np.mean([item["completion_time_s"] for item in episode_reports])
        ),
        "maximum_crossing_radial_m": float(
            max(item["maximum_crossing_radial_m"] for item in episode_reports)
        ),
        "gates": gates.tolist(),
        "settings": settings,
    }
    return records, report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "drone_race_vq1_v3391_telemetry.ini",
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3391)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    records, report = collect_dataset(args.config, args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(args.output)
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
