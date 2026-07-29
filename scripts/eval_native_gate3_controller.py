#!/usr/bin/env python3
"""Replay and screen a Gate-3-only roll controller in the native SITL plant.

Each official trace is initialized at its first visible Gate-3 sample inside
the configured admission range.  Pitch, thrust, and yaw are replayed from the
official trace.  ``replay`` also replays roll and is the model-authority test;
``controller`` replaces only roll with one smooth, anchored projected-path PD
law.  Candidate 045 can therefore remain a true holdout.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from collections.abc import Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclasses.dataclass(frozen=True)
class ControllerParameters:
    path_power: float = 2.0
    terminal_right_m: float = -0.10
    position_gain: float = 1.0
    rate_gain: float = 0.70
    max_roll_norm: float = 0.90
    max_roll_slew_norm_s: float = 2.0
    initial_rate_blend_tau_s: float = 0.60


@dataclasses.dataclass(frozen=True)
class PlantParameters:
    vertical_accel_per_thrust: float
    horizontal_accel_scale: float
    braking_accel_scale: float
    lateral_accel_scale: float
    gravity_m_s2: float
    linear_drag_per_s: float
    rate_lag_tau_s: float


@dataclasses.dataclass(frozen=True)
class TraceScenario:
    path: str
    tag: str
    official_pass: bool
    official_active_gate_index: int
    admission_elapsed_s: float
    forward_m: float
    right_m: float
    down_m: float
    forward_rate_m_s: float
    right_rate_m_s: float
    down_rate_m_s: float
    quat_wxyz: tuple[float, float, float, float]
    omega_rad_s: tuple[float, float, float]
    velocity_world_m_s: tuple[float, float, float]
    velocity_source: str
    start_position_world_m: tuple[float, float, float]
    initial_action: tuple[float, float, float, float]
    profile_times_s: tuple[float, ...]
    profile_actions: tuple[tuple[float, float, float, float], ...]


@dataclasses.dataclass
class ControllerState:
    initial_forward_m: torch.Tensor
    initial_right_m: torch.Tensor
    initial_forward_rate_m_s: torch.Tensor
    initial_right_rate_m_s: torch.Tensor
    previous_roll_norm: torch.Tensor
    elapsed_s: float = 0.0


def _clip_atanh(value: float, scale: float) -> float:
    return float(np.arctanh(np.clip(float(value), -0.999999, 0.999999)) * scale)


def _rotate_by_quaternion(
    vector: Sequence[float], quat_wxyz: Sequence[float]
) -> tuple[float, float, float]:
    vx, vy, vz = (float(value) for value in vector)
    qw, qx, qy, qz = (float(value) for value in quat_wxyz)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_from_report(
    report: dict,
    *,
    path: str,
    admission_forward_m: float,
    target_world_m: tuple[float, float, float],
    velocity_source_mode: str = "observable_rate",
) -> TraceScenario:
    samples = report.get("policy_trace", {}).get("samples", [])
    candidates: list[tuple[float, dict, list[float]]] = []
    for sample in samples:
        observation = sample.get("observation") or []
        if (
            int(sample.get("official_active_gate_index") or -1) != 2
            or len(observation) < 23
            or float(observation[10]) < 0.5
        ):
            continue
        forward_m = _clip_atanh(observation[11], 10.0)
        closing_m_s = -_clip_atanh(observation[0], 5.0)
        if 0.0 < forward_m <= admission_forward_m and closing_m_s >= 1.0:
            candidates.append((float(sample["elapsed_s"]), sample, observation))
    if not candidates:
        raise ValueError(f"{path}: no visible Gate-3 sample inside admission range")
    candidates.sort(key=lambda row: row[0])
    admission_elapsed_s, first, observation = candidates[0]

    forward_m = _clip_atanh(observation[11], 10.0)
    right_m = _clip_atanh(observation[12], 5.0)
    down_m = _clip_atanh(observation[13], 5.0)
    forward_rate_m_s = _clip_atanh(observation[0], 5.0)
    right_rate_m_s = _clip_atanh(observation[1], 3.0)
    down_rate_m_s = _clip_atanh(observation[2], 3.0)
    quat = tuple(float(value) for value in observation[6:10])
    quat_norm = math.sqrt(sum(value * value for value in quat))
    if quat_norm <= 1e-8:
        raise ValueError(f"{path}: invalid zero attitude quaternion")
    quat = tuple(value / quat_norm for value in quat)
    omega = tuple(float(observation[index]) * 20.0 for index in range(3, 6))

    # Reconstruct instantaneous velocity causally from the associated gate
    # poses available up to admission.  Observation 0..2 is deliberately a
    # lagged control feature and must not be mistaken for physical velocity.
    history: list[tuple[float, tuple[float, float, float]]] = []
    for sample in samples:
        elapsed = float(sample.get("elapsed_s", -1.0))
        row = sample.get("observation") or []
        if (
            elapsed > admission_elapsed_s + 1e-9
            or elapsed < admission_elapsed_s - 1.0
            or int(sample.get("official_active_gate_index") or -1) != 2
            or len(row) < 23
            or float(row[10]) < 0.5
        ):
            continue
        row_quat = tuple(float(value) for value in row[6:10])
        row_relative_body_up = (
            _clip_atanh(row[11], 10.0),
            _clip_atanh(row[12], 5.0),
            -_clip_atanh(row[13], 5.0),
        )
        history.append(
            (
                elapsed - admission_elapsed_s,
                _rotate_by_quaternion(row_relative_body_up, row_quat),
            )
        )
    history.sort(key=lambda row: row[0])
    history = history[-7:]
    causal_velocity: tuple[float, float, float] | None = None
    causal_velocity_source = "causal_gate_pose_unavailable"
    if len(history) >= 3 and len({row[0] for row in history}) == len(history):
        history_times = np.asarray([row[0] for row in history], dtype=np.float64)
        history_vectors = np.asarray([row[1] for row in history], dtype=np.float64)
        degree = 2 if len(history) >= 5 else 1
        reconstructed = []
        for axis in range(3):
            coefficients = np.polyfit(history_times, history_vectors[:, axis], degree)
            derivative = float(coefficients[-2])
            reconstructed.append(float(np.clip(-derivative, -20.0, 20.0)))
        causal_velocity = tuple(reconstructed)
        causal_velocity_source = (
            f"causal_gate_pose_polynomial_degree_{degree}_n_{len(history)}"
        )

    gate_rate_body_up = (
        forward_rate_m_s,
        right_rate_m_s,
        -down_rate_m_s,
    )
    gate_rate_world = _rotate_by_quaternion(gate_rate_body_up, quat)
    observable_rate_velocity = tuple(-value for value in gate_rate_world)
    if velocity_source_mode == "causal_pose" and causal_velocity is not None:
        velocity = causal_velocity
        velocity_source = causal_velocity_source
    elif velocity_source_mode == "observable_rate":
        velocity = observable_rate_velocity
        velocity_source = "observable_gate_rate"
    elif velocity_source_mode == "causal_pose":
        velocity = observable_rate_velocity
        velocity_source = "observable_gate_rate_fallback"
    else:
        raise ValueError(f"unsupported velocity source: {velocity_source_mode}")

    relative_body_up = (forward_m, right_m, -down_m)
    relative_world = _rotate_by_quaternion(relative_body_up, quat)
    start_position = tuple(
        float(target_world_m[index]) - relative_world[index] for index in range(3)
    )

    profile: list[tuple[float, tuple[float, float, float, float]]] = []
    for sample in samples:
        action = sample.get("normalized_action") or []
        elapsed = float(sample.get("elapsed_s", -1.0))
        if (
            elapsed + 1e-9 < admission_elapsed_s
            or int(sample.get("official_active_gate_index") or -1) != 2
            or len(action) != 4
        ):
            continue
        profile.append(
            (
                elapsed - admission_elapsed_s,
                tuple(float(np.clip(value, -1.0, 1.0)) for value in action),
            )
        )
    if not profile:
        raise ValueError(f"{path}: no action profile after admission")
    profile.sort(key=lambda row: row[0])
    if profile[0][0] > 1e-6:
        profile.insert(0, (0.0, profile[0][1]))

    official_index = int(report.get("official_active_gate_index") or 0)
    initial_action = tuple(float(value) for value in observation[19:23])
    return TraceScenario(
        path=path,
        tag=Path(path).stem,
        official_pass=official_index >= 3,
        official_active_gate_index=official_index,
        admission_elapsed_s=admission_elapsed_s,
        forward_m=forward_m,
        right_m=right_m,
        down_m=down_m,
        forward_rate_m_s=forward_rate_m_s,
        right_rate_m_s=right_rate_m_s,
        down_rate_m_s=down_rate_m_s,
        quat_wxyz=quat,
        omega_rad_s=omega,
        velocity_world_m_s=tuple(float(value) for value in velocity),
        velocity_source=velocity_source,
        start_position_world_m=start_position,
        initial_action=initial_action,
        profile_times_s=tuple(row[0] for row in profile),
        profile_actions=tuple(row[1] for row in profile),
    )


def _plant_from_dynamics_fit(report: dict) -> PlantParameters:
    axes = {row["axis"]: row for row in report.get("axis_fits", [])}
    if not {"x", "z"}.issubset(axes):
        raise ValueError("dynamics fit must contain x and z axis fits")
    vertical = abs(float(axes["z"]["thrust_accel_per_unit"]))
    if vertical <= 1e-6:
        raise ValueError("dynamics fit has zero vertical thrust authority")
    horizontal = abs(float(axes["x"]["thrust_accel_per_unit"])) / vertical
    lateral_fit = report.get("gate3_lateral_fit") or {}
    lateral_gain = abs(float(lateral_fit.get("accel_per_tan_roll_m_s2", 0.0)))
    if lateral_gain <= 1e-6:
        raise ValueError("dynamics fit must contain Gate-3 lateral authority")
    # Around Gate 3 the command is close to the measured 0.27 hover thrust.
    lateral = lateral_gain / (vertical * 0.27)
    gravity = -float(axes["z"].get("gravity_intercept_m_s2", 0.0))
    if gravity <= 0.0:
        raise ValueError("dynamics fit must contain a negative world-z intercept")
    drag = max(float(axes["x"].get("linear_drag_per_s", 0.0)), 0.0)
    tau = float(report.get("attitude_tau_s", {}).get("pitch") or 0.0)
    if tau <= 0.0:
        raise ValueError("dynamics fit must contain a positive pitch time constant")
    return PlantParameters(
        vertical_accel_per_thrust=vertical,
        horizontal_accel_scale=horizontal,
        braking_accel_scale=horizontal,
        lateral_accel_scale=lateral,
        gravity_m_s2=gravity,
        linear_drag_per_s=drag,
        rate_lag_tau_s=tau,
    )


def _load_config(env_name: str) -> dict:
    from pufferlib import pufferl

    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(env_name)
    finally:
        sys.argv = saved_argv


def _scenario_config(
    env_name: str,
    scenario: TraceScenario,
    *,
    replicates: int,
    episode_duration_s: float,
    gate_radius_m: float,
    plant: PlantParameters | None,
) -> dict:
    cfg = _load_config(env_name)
    cfg["vec"]["total_agents"] = int(replicates)
    cfg["vec"]["num_buffers"] = min(int(replicates), 8)
    cfg["vec"]["num_threads"] = min(int(replicates), 8)
    env = cfg["env"]
    env.update(
        {
            "num_drones": 1,
            "num_gates": 3,
            "gate_radius": float(gate_radius_m),
            "gate_radius_randomize": 0,
            "gate_radius_profile_mix": 0,
            "gate0_radius": float(gate_radius_m),
            "gate1_radius": float(gate_radius_m),
            "gate2_radius": float(gate_radius_m),
            "use_custom_start": 1,
            "start_gate_index": 2,
            "start_elapsed_time": float(scenario.admission_elapsed_s),
            "mixed_start_curriculum": 0,
            "start_x": float(scenario.start_position_world_m[0]),
            "start_y": float(scenario.start_position_world_m[1]),
            "start_z": float(scenario.start_position_world_m[2]),
            "start_vx": float(scenario.velocity_world_m_s[0]),
            "start_vy": float(scenario.velocity_world_m_s[1]),
            "start_vz": float(scenario.velocity_world_m_s[2]),
            "start_qw": float(scenario.quat_wxyz[0]),
            "start_qx": float(scenario.quat_wxyz[1]),
            "start_qy": float(scenario.quat_wxyz[2]),
            "start_qz": float(scenario.quat_wxyz[3]),
            "start_wx": float(scenario.omega_rad_s[0]),
            "start_wy": float(scenario.omega_rad_s[1]),
            "start_wz": float(scenario.omega_rad_s[2]),
            "gate_position_domain_randomize": 0,
            "course_geometry_scale_randomize": 0,
            "sitl_plant_domain_randomize": 0,
            "sitl_gate_obs_dropout_prob": 0.0,
            "sitl_gate_obs_dropout_range_m": 0.0,
            "evaluation_episode_limit": 1,
            "evaluation_episode_offset": 0,
            "strict_missed_gate": 1,
            "time_limit_seconds": float(
                scenario.admission_elapsed_s + episode_duration_s
            ),
            "max_steps": int(math.ceil(episode_duration_s / float(env["dt"]))) + 2,
        }
    )
    if plant is not None:
        env.update(
            {
                "sitl_vertical_accel_per_thrust": plant.vertical_accel_per_thrust,
                "sitl_horizontal_accel_scale": plant.horizontal_accel_scale,
                "sitl_braking_accel_scale": plant.braking_accel_scale,
                "sitl_lateral_accel_scale": plant.lateral_accel_scale,
                "sitl_gravity_m_s2": plant.gravity_m_s2,
                "sitl_linear_drag": plant.linear_drag_per_s,
                "sitl_rate_lag_tau_s": plant.rate_lag_tau_s,
            }
        )
    return cfg


def _buffer(ptr: int, count: int) -> torch.Tensor:
    return torch.frombuffer(
        (ctypes.c_float * count).from_address(ptr), dtype=torch.float32
    )


def _flat_log(log: dict) -> dict[str, float]:
    from pufferlib import pufferl

    flattened: dict[str, float] = {}
    for key, value in pufferl.unroll_nested_dict(log):
        normalized = str(key)
        if "/" not in normalized:
            normalized = f"env/{normalized}"
        flattened[normalized] = float(value)
    return flattened


def _profile_action(scenario: TraceScenario, elapsed_s: float) -> np.ndarray:
    times = np.asarray(scenario.profile_times_s, dtype=np.float64)
    actions = np.asarray(scenario.profile_actions, dtype=np.float64)
    return np.asarray(
        [
            np.interp(elapsed_s, times, actions[:, index])
            for index in range(4)
        ],
        dtype=np.float32,
    )


def _decode_native_geometry(
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clipped = observations.clamp(-0.999999, 0.999999)
    forward = torch.atanh(clipped[:, 11]) * 10.0
    right = torch.atanh(clipped[:, 12]) * 5.0
    forward_rate = torch.atanh(clipped[:, 0]) * 5.0
    right_rate = torch.atanh(clipped[:, 1]) * 3.0
    return forward, right, forward_rate, right_rate


def _initialize_controller_state(
    observations: torch.Tensor,
    scenario: TraceScenario,
) -> ControllerState:
    forward, right, _forward_rate, _right_rate = _decode_native_geometry(observations)
    return ControllerState(
        initial_forward_m=forward.clone().clamp_min(0.25),
        initial_right_m=right.clone(),
        initial_forward_rate_m_s=torch.full_like(
            forward, float(scenario.forward_rate_m_s)
        ),
        initial_right_rate_m_s=torch.full_like(
            right, float(scenario.right_rate_m_s)
        ),
        previous_roll_norm=torch.full_like(
            right, float(scenario.initial_action[1])
        ),
    )


def _controller_roll(
    observations: torch.Tensor,
    state: ControllerState,
    parameters: ControllerParameters,
    *,
    dt_s: float,
) -> torch.Tensor:
    forward, right, native_forward_rate, native_right_rate = (
        _decode_native_geometry(observations)
    )
    ratio = (forward / state.initial_forward_m).clamp(0.0, 1.0)
    power = float(parameters.path_power)
    desired_right = float(parameters.terminal_right_m) + (
        state.initial_right_m - float(parameters.terminal_right_m)
    ) * ratio.pow(power)

    blend_tau = max(float(parameters.initial_rate_blend_tau_s), 1e-6)
    blend = math.exp(-state.elapsed_s / blend_tau)
    forward_rate = (
        blend * state.initial_forward_rate_m_s
        + (1.0 - blend) * native_forward_rate
    )
    right_rate = (
        blend * state.initial_right_rate_m_s
        + (1.0 - blend) * native_right_rate
    )
    closing = (-forward_rate).clamp_min(0.0)
    derivative_scale = torch.where(
        ratio > 0.0,
        ratio.pow(max(power - 1.0, 0.0)),
        torch.zeros_like(ratio),
    )
    desired_right_rate = -(
        state.initial_right_m - float(parameters.terminal_right_m)
    ) * power / state.initial_forward_m * derivative_scale * closing

    requested = (
        float(parameters.position_gain) * (desired_right - right)
        + float(parameters.rate_gain) * (desired_right_rate - right_rate)
    )
    cap = abs(float(parameters.max_roll_norm))
    requested = requested.clamp(-cap, cap)
    max_delta = abs(float(parameters.max_roll_slew_norm_s)) * dt_s
    requested = torch.maximum(
        torch.minimum(requested, state.previous_roll_norm + max_delta),
        state.previous_roll_norm - max_delta,
    )
    visible = observations[:, 10] >= 0.5
    requested = torch.where(visible, requested, state.previous_roll_norm)
    state.previous_roll_norm.copy_(requested)
    state.elapsed_s += dt_s
    return requested


def _add_derived_metrics(metrics: dict[str, float]) -> None:
    sampled = metrics.get("env/terminal_crossing_sampled", 0.0)
    for axis in ("radial", "right", "vertical"):
        total = metrics.get(f"env/terminal_crossing_{axis}", 0.0)
        metrics[f"env/avg_terminal_crossing_{axis}"] = (
            total / sampled if sampled else 0.0
        )


def _run_scenario(
    scenario: TraceScenario,
    *,
    mode: str,
    parameters: ControllerParameters,
    env_name: str,
    replicates: int,
    episode_duration_s: float,
    gate_radius_m: float,
    plant: PlantParameters | None,
) -> dict:
    from pufferlib import _C

    cfg = _scenario_config(
        env_name,
        scenario,
        replicates=replicates,
        episode_duration_s=episode_duration_s,
        gate_radius_m=gate_radius_m,
        plant=plant,
    )
    if getattr(_C, "env_name", None) != cfg.get("backend_env_name", "drone_race"):
        raise RuntimeError(
            f"compiled native backend is {getattr(_C, 'env_name', None)!r}, "
            f"expected {cfg.get('backend_env_name', 'drone_race')!r}"
        )
    vec = _C.create_vec(cfg, gpu=0)
    vec.reset()
    observations = _buffer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    terminals = _buffer(vec.terminals_ptr, vec.total_agents)
    actions = torch.zeros((vec.total_agents, vec.num_atns), dtype=torch.float32)
    controller_state = _initialize_controller_state(observations, scenario)
    dt_s = float(cfg["env"]["dt"])
    max_steps = int(cfg["env"]["max_steps"]) + 2
    metrics: dict[str, float] = {}
    steps = 0
    started = time.monotonic()
    try:
        for steps in range(1, max_steps + 1):
            profile = torch.from_numpy(
                _profile_action(scenario, (steps - 1) * dt_s)
            )
            actions.copy_(profile.unsqueeze(0).expand_as(actions))
            if mode == "controller":
                actions[:, 1] = _controller_roll(
                    observations,
                    controller_state,
                    parameters,
                    dt_s=dt_s,
                )
            vec.cpu_step(actions.data_ptr())
            if bool(torch.all(terminals >= 0.5)):
                break
        metrics = _flat_log(dict(vec.log()))
    finally:
        vec.close()
    _add_derived_metrics(metrics)
    completed = metrics.get("env/n", 0.0)
    # StaticVec normalizes episode metrics by their sampled episode count; n is
    # retained as the exact aggregate count.  success_rate is already a rate.
    success_rate = metrics.get("env/success_rate", 0.0) if completed else 0.0
    predicted_pass = success_rate >= 0.5
    return {
        "trace": scenario.path,
        "tag": scenario.tag,
        "official_pass": scenario.official_pass,
        "official_active_gate_index": scenario.official_active_gate_index,
        "predicted_pass": predicted_pass,
        "classification_match": predicted_pass == scenario.official_pass,
        "success_rate": success_rate,
        "steps": steps,
        "elapsed_wall_s": round(time.monotonic() - started, 6),
        "initial_state": {
            "elapsed_s": scenario.admission_elapsed_s,
            "forward_m": scenario.forward_m,
            "right_m": scenario.right_m,
            "down_m": scenario.down_m,
            "forward_rate_m_s": scenario.forward_rate_m_s,
            "right_rate_m_s": scenario.right_rate_m_s,
            "down_rate_m_s": scenario.down_rate_m_s,
            "velocity_world_m_s": list(scenario.velocity_world_m_s),
            "velocity_source": scenario.velocity_source,
            "start_position_world_m": list(scenario.start_position_world_m),
        },
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("replay", "controller"), default="replay")
    parser.add_argument(
        "--env-name", default="drone_race_full_policy_six_gate_bootstrap"
    )
    parser.add_argument("--admission-forward-m", type=float, default=12.0)
    parser.add_argument("--gate-radius-m", type=float, default=0.75)
    parser.add_argument(
        "--velocity-source",
        choices=("observable_rate", "causal_pose"),
        default="observable_rate",
    )
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--episode-duration-s", type=float, default=4.0)
    parser.add_argument(
        "--dynamics-fit",
        type=Path,
        help="Frozen official-observation dynamics fit used for evaluator-only plant parameters.",
    )
    parser.add_argument("--path-power", type=float, default=2.0)
    parser.add_argument("--terminal-right-m", type=float, default=-0.10)
    parser.add_argument("--position-gain", type=float, default=1.0)
    parser.add_argument("--rate-gain", type=float, default=0.70)
    parser.add_argument("--max-roll-norm", type=float, default=0.90)
    parser.add_argument("--max-roll-slew-norm-s", type=float, default=2.0)
    parser.add_argument("--initial-rate-blend-tau-s", type=float, default=0.60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.episode_duration_s <= 0.0:
        parser.error("--episode-duration-s must be positive")
    if args.path_power < 1.0:
        parser.error("--path-power must be at least 1")

    parameters = ControllerParameters(
        path_power=args.path_power,
        terminal_right_m=args.terminal_right_m,
        position_gain=args.position_gain,
        rate_gain=args.rate_gain,
        max_roll_norm=args.max_roll_norm,
        max_roll_slew_norm_s=args.max_roll_slew_norm_s,
        initial_rate_blend_tau_s=args.initial_rate_blend_tau_s,
    )
    plant = None
    if args.dynamics_fit:
        with args.dynamics_fit.open("r", encoding="utf-8") as handle:
            plant = _plant_from_dynamics_fit(json.load(handle))
    target = (67.53733, -1.24074, -10.23080)
    scenarios: list[TraceScenario] = []
    for path in args.trace:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        scenarios.append(
            _scenario_from_report(
                report,
                path=str(path),
                admission_forward_m=args.admission_forward_m,
                target_world_m=target,
                velocity_source_mode=args.velocity_source,
            )
        )

    rows = [
        _run_scenario(
            scenario,
            mode=args.mode,
            parameters=parameters,
            env_name=args.env_name,
            replicates=args.replicates,
            episode_duration_s=args.episode_duration_s,
            gate_radius_m=args.gate_radius_m,
            plant=plant,
        )
        for scenario in scenarios
    ]
    matches = sum(bool(row["classification_match"]) for row in rows)
    official_passes = sum(scenario.official_pass for scenario in scenarios)
    pass_matches = sum(
        bool(row["official_pass"] and row["predicted_pass"]) for row in rows
    )
    failure_matches = sum(
        bool(not row["official_pass"] and not row["predicted_pass"]) for row in rows
    )
    official_failures = len(rows) - official_passes
    accuracy = matches / len(rows) if rows else 0.0
    pass_recall = pass_matches / official_passes if official_passes else 0.0
    failure_recall = failure_matches / official_failures if official_failures else 0.0
    replay_authority_passed = bool(
        args.mode == "replay"
        and rows
        and accuracy >= 0.80
        and pass_recall >= 0.80
        and failure_recall >= 0.80
    )
    result = {
        "passed": bool(
            args.mode == "controller"
            and rows
            and all(float(row["success_rate"]) >= 0.95 for row in rows)
        ),
        "mode": args.mode,
        "parameters": dataclasses.asdict(parameters),
        "configuration": {
            "env_name": args.env_name,
            "compiled_backend": None,
            "admission_forward_m": args.admission_forward_m,
            "gate_radius_m": args.gate_radius_m,
            "velocity_source": args.velocity_source,
            "replicates_per_trace": args.replicates,
            "episode_duration_s": args.episode_duration_s,
            "trace_count": len(rows),
            "official_pass_count": official_passes,
            "official_failure_count": len(rows) - official_passes,
            "plant_parameters": None if plant is None else dataclasses.asdict(plant),
        },
        "authority": {
            "classification_matches": matches,
            "classification_total": len(rows),
            "classification_accuracy": accuracy,
            "official_pass_recall": pass_recall,
            "official_failure_recall": failure_recall,
            "required_replay_accuracy": 0.80,
            "required_pass_recall": 0.80,
            "required_failure_recall": 0.80,
            "replay_authority_passed": replay_authority_passed,
        },
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "trace_sha256": {
                str(path): _sha256(path.resolve()) for path in args.trace
            },
            "dynamics_fit": None if args.dynamics_fit is None else str(args.dynamics_fit),
            "dynamics_fit_sha256": (
                None
                if args.dynamics_fit is None
                else _sha256(args.dynamics_fit.resolve())
            ),
        },
        "results": rows,
    }
    try:
        from pufferlib import _C

        result["configuration"]["compiled_backend"] = getattr(_C, "env_name", None)
        result["configuration"]["native_precision_bytes"] = int(
            getattr(_C, "precision_bytes", 0)
        )
    except ImportError:
        pass
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.mode == "replay":
        return 0 if result["authority"]["replay_authority_passed"] else 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
