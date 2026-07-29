#!/usr/bin/env python3
"""Evaluate the privileged native VQ2 oracle without collecting labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import torch

from pufferlib.vq2_informed import configure_full_start_evaluation


ENV_NAME = "drone_race_vq2_informed_dreamer"
BACKEND_ENV_NAME = "drone_race_vision"
DT_SECONDS = 0.015625
EPISODE_SECONDS = 40.0
EPISODE_STEPS = int(round(EPISODE_SECONDS / DT_SECONDS))


def fixed_overrides(*, agents: int, seed: int) -> list[str]:
    """Return loader-visible vector and seed overrides."""

    return [
        "--seed", str(seed),
        "--vec.total-agents", str(agents),
        "--vec.num-buffers", "2",
        "--vec.num-threads", "4",
    ]


def fixed_environment(
    controller: str = "spline",
    *,
    target_speed_m_s: float = 4.0,
    episode_seconds: float = EPISODE_SECONDS,
    roll_per_m: float = 0.35,
    roll_rate_per_m_s: float = 0.10,
    true_velocity_damping: bool = False,
    randomized_course: bool = False,
) -> dict[str, int | float]:
    """Return binding-level values, including fields absent from the INI."""

    if controller not in {"spline", "direct", "segment", "governed"}:
        raise ValueError(f"unknown oracle controller {controller!r}")
    if (
        target_speed_m_s <= 0.0
        or episode_seconds <= 0.0
        or roll_per_m < 0.0
        or roll_rate_per_m_s < 0.0
    ):
        raise ValueError("target speed and episode duration must be positive")
    episode_steps = int(round(episode_seconds / DT_SECONDS))
    values: dict[str, int | float] = {
        "num_drones": 1,
        "dt": DT_SECONDS,
        "num_gates": 6,
        "gate_radius": 0.75,
        "gate_radius_randomize": 0,
        "gate_radius_profile_mix": 0,
        "gate_position_domain_randomize": 0,
        "course_geometry_scale_randomize": 0,
        "sitl_plant_domain_randomize": 0,
        "use_custom_start": 0,
        "gate_local_start_curriculum": 0,
        "gate_local_start_probability": 0.0,
        "mixed_start_curriculum": 0,
        "segment_start_probability": 0.0,
        "max_steps": episode_steps,
        "time_limit_seconds": episode_seconds,
        "visual_camera_roll_jitter_rad": 0.0,
        "visual_camera_pitch_jitter_rad": 0.0,
        "visual_camera_yaw_jitter_rad": 0.0,
        "visual_camera_dropout_prob": 0.0,
        "visual_edge_dropout_prob": 0.0,
        "visual_edge_corrupt_prob": 0.0,
        "visual_false_segments": 0,
        "visual_rolling_shutter_s": 0.0,
        "teacher_action_blend": 1.0,
        "teacher_action_blend_from_step": 0,
        "teacher_course_spline": int(controller == "spline"),
        "teacher_segment_minimum_jerk": int(controller == "segment"),
        "teacher_alignment_governor": int(controller == "governed"),
        "teacher_from_gate_index": 0,
        "teacher_pitch_from_gate_index": 0,
        "teacher_pitch_speed_control": 1,
        "teacher_pitch_speed_target_m_s": target_speed_m_s,
        "teacher_pitch_speed_gain": 0.35,
        "teacher_pitch_speed_scale": 0.24,
        "teacher_roll_from_gate_index": 0,
        "teacher_roll_until_gate_index": 6,
        "teacher_thrust_from_gate_index": 0,
        "teacher_yaw_control": 1,
        "teacher_yaw_from_gate_index": 0,
        "teacher_yaw_action": 0.0,
    }
    if controller == "direct":
        values.update({
            "teacher_roll_per_m": roll_per_m,
            "teacher_roll_rate_per_m_s": roll_rate_per_m_s,
            "teacher_true_velocity_damping": int(true_velocity_damping),
            "teacher_thrust_bias": 0.0,
            "teacher_thrust_per_m": 0.30,
            "teacher_thrust_rate_per_m_s": 0.10,
            "teacher_thrust_world_frame": 1,
        })
    if randomized_course:
        values.update({
            "gate_position_domain_randomize": 1,
            "gate_position_domain_randomize_probability": 1.0,
            "gate_position_randomize_from_index": 0,
            "gate_position_jitter_x": 3.0,
            "gate_position_jitter_y": 5.0,
            "gate_position_jitter_z": 0.5,
            "course_geometry_scale_randomize": 1,
            "course_geometry_scale_min": 0.35,
            "course_geometry_scale_max": 1.0,
        })
    return values


def load_fixed_config(
    pufferl_module: Any,
    *,
    agents: int,
    episodes: int,
    seed: int,
    controller: str = "spline",
    target_speed_m_s: float = 4.0,
    episode_seconds: float = EPISODE_SECONDS,
    roll_per_m: float = 0.35,
    roll_rate_per_m_s: float = 0.10,
    true_velocity_damping: bool = False,
    randomized_course: bool = False,
):
    if agents <= 0 or episodes <= 0 or episodes % agents:
        raise ValueError("episodes must be a positive multiple of agents")
    saved_argv = sys.argv
    overrides = fixed_overrides(agents=agents, seed=seed)
    try:
        sys.argv = [sys.argv[0], *overrides]
        config = pufferl_module.load_config(ENV_NAME)
    finally:
        sys.argv = saved_argv

    if config.get("backend_env_name") != BACKEND_ENV_NAME:
        raise RuntimeError("oracle diagnostic requires the drone_race_vision backend")
    if int(config["vec"]["total_agents"]) != agents:
        raise RuntimeError("vector agent count differs from the fixed override")
    config.setdefault("env", {}).update(fixed_environment(
        controller,
        target_speed_m_s=target_speed_m_s,
        episode_seconds=episode_seconds,
        roll_per_m=roll_per_m,
        roll_rate_per_m_s=roll_rate_per_m_s,
        true_velocity_damping=true_velocity_damping,
        randomized_course=randomized_course,
    ))
    configure_full_start_evaluation(
        config,
        episodes_per_agent=episodes // agents,
        episode_offset=0,
    )
    return config, overrides


def flatten_log(pufferl_module: Any, raw_log: dict[str, Any]) -> dict[str, float]:
    flat = dict(pufferl_module.unroll_nested_dict(raw_log))
    normalized = {}
    for key, value in flat.items():
        name = str(key)
        if "/" not in name:
            name = f"env/{name}"
        normalized[name] = float(value)
    return normalized


def native_step_budget(*, max_steps: int, episodes: int, agents: int) -> int:
    """Cover every exact-count episode, including its reset-only boundary."""

    if max_steps <= 0 or agents <= 0 or episodes <= 0 or episodes % agents:
        raise ValueError("invalid exact-count native step budget")
    return (max_steps + 1) * (episodes // agents)


def oracle_passes(metrics: dict[str, float], *, episodes: int) -> bool:
    return (
        metrics.get("env/n", 0.0) == float(episodes)
        and metrics.get("env/success_rate", 0.0) == 1.0
        and metrics.get("env/gates_passed", 0.0) == 6.0
        and metrics.get("env/crash", 0.0) == 0.0
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/missed_gate", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/valid_run_rate", 0.0) == 1.0
        and metrics.get("env/crossing_margin_violation", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
    )


def run_diagnostic(
    *,
    agents: int,
    episodes: int,
    seed: int,
    controller: str,
    target_speed_m_s: float,
    episode_seconds: float,
    roll_per_m: float = 0.35,
    roll_rate_per_m_s: float = 0.10,
    true_velocity_damping: bool = False,
    randomized_course: bool = False,
) -> dict[str, Any]:
    from pufferlib import _C
    from pufferlib import pufferl

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError(
            f"compiled backend is {getattr(_C, 'env_name', None)!r}; "
            f"expected {BACKEND_ENV_NAME!r}"
        )
    config, overrides = load_fixed_config(
        pufferl,
        agents=agents,
        episodes=episodes,
        seed=seed,
        controller=controller,
        target_speed_m_s=target_speed_m_s,
        episode_seconds=episode_seconds,
        roll_per_m=roll_per_m,
        roll_rate_per_m_s=roll_rate_per_m_s,
        true_velocity_damping=true_velocity_damping,
        randomized_course=randomized_course,
    )
    vector = _C.create_vec(config, gpu=0)
    actions = torch.zeros((vector.total_agents, vector.num_atns), dtype=torch.float32)
    started = time.perf_counter()
    try:
        vector.reset()
        # One extra step covers the visual environment's terminal/reset boundary.
        native_steps = native_step_budget(
            max_steps=int(config["env"]["max_steps"]),
            episodes=episodes,
            agents=agents,
        )
        for _ in range(native_steps):
            vector.cpu_step(actions.data_ptr())
        metrics = flatten_log(pufferl, dict(vector.log()))
    finally:
        vector.close()
    elapsed = time.perf_counter() - started

    return {
        "schema": "vq2_native_oracle_diagnostic_v1",
        "tag": (
            "vq2_sf002_native_oracle_baseline_64"
            if controller == "spline"
            else (
                (
                    "vq2_sf010_native_oracle_randomized_course_512"
                    if randomized_course and episodes == 512
                    else (
                        "vq2_sf011_native_oracle_randomized_course_4096"
                        if randomized_course and episodes == 4096
                        else "vq2_sf009_native_oracle_alignment_governed_64"
                    )
                )
                if controller == "governed"
                else (
                    "vq2_sf008_native_oracle_minimum_jerk_64"
                    if controller == "segment"
                    else (
                        "vq2_sf004_native_oracle_direct_slow_64"
                        if target_speed_m_s == 2.5 and episode_seconds == 60.0
                        else "vq2_sf003_native_oracle_direct_64"
                    )
                )
            )
        ),
        "controller": controller,
        "environment": ENV_NAME,
        "compiled_backend": BACKEND_ENV_NAME,
        "agents": agents,
        "episodes": episodes,
        "seed": seed,
        "native_steps": native_steps,
        "wall_time_seconds": elapsed,
        "zero_external_actions": True,
        "teacher_labels_written": 0,
        "loader_overrides": overrides,
        "fixed_environment": fixed_environment(
            controller,
            target_speed_m_s=target_speed_m_s,
            episode_seconds=episode_seconds,
            roll_per_m=roll_per_m,
            roll_rate_per_m_s=roll_rate_per_m_s,
            true_velocity_damping=true_velocity_damping,
            randomized_course=randomized_course,
        ),
        "metrics": metrics,
        "baseline_passed": oracle_passes(metrics, episodes=episodes),
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42002)
    parser.add_argument(
        "--controller",
        choices=("spline", "direct", "segment", "governed"),
        default="spline",
    )
    parser.add_argument("--target-speed-m-s", type=float, default=4.0)
    parser.add_argument("--episode-seconds", type=float, default=EPISODE_SECONDS)
    parser.add_argument("--roll-per-m", type=float, default=0.35)
    parser.add_argument("--roll-rate-per-m-s", type=float, default=0.10)
    parser.add_argument("--true-velocity-damping", action="store_true")
    parser.add_argument("--randomized-course", action="store_true")
    args = parser.parse_args()
    report = run_diagnostic(
        agents=args.agents,
        episodes=args.episodes,
        seed=args.seed,
        controller=args.controller,
        target_speed_m_s=args.target_speed_m_s,
        episode_seconds=args.episode_seconds,
        roll_per_m=args.roll_per_m,
        roll_rate_per_m_s=args.roll_rate_per_m_s,
        true_velocity_damping=args.true_velocity_damping,
        randomized_course=args.randomized_course,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["baseline_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
