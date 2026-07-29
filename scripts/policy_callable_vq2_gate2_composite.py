#!/usr/bin/env python3
"""Two-checkpoint recurrent Puffer policy for bounded VQ2 Gate-2 work.

The frozen prefix checkpoint emits the complete action vector while official
Gate 1 is active; the Gate-2 checkpoint emits the complete vector after
official progress reaches index 1.  Gate 2 can either consume the live prefix
or begin from a fixed recurrent state warmed only by an archived legal public
observation trace.  No action blend, analytic controller, or native-coordinate
input exists here.
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from policy_callable_checkpoint import CheckpointPolicy


@dataclass
class _CompositeState:
    prefix: CheckpointPolicy
    gate2: CheckpointPolicy
    gate2_warm_state: object | None = None
    runtime_duration_s: float = 0.0
    gate2_time_limit_s: float = 0.0
    world_frame_gate_features: bool = False
    linear_gate_features: bool = False
    predict_forward_gate_rate: bool = False
    initial_forward_gate_rate_world_x: float = 0.0
    forward_gate_rate_world_x: float = 0.0
    forward_predictor_dt_s: float = 1.0 / 60.0
    forward_predictor_started: bool = False
    predict_gate_kinematics: bool = False
    initial_gate_rate_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_gate_position_world_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    initial_gate_position_world: tuple[float, float, float] | None = None
    gate_position_world: tuple[float, float, float] | None = None
    gate_rate_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    association_jump_threshold_m: float = 0.0
    reseed_bearing_on_association: bool = False
    association_ready: bool = True
    association_reseed_pending: bool = False
    last_gate_pose_body_ned: tuple[float, float, float] | None = None


_STATE: _CompositeState | None = None
_KEY: tuple[object, ...] | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _required_path(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _optional_path(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _gate0_prefix_observations(report_path: str) -> list[list[float]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    samples = report.get("policy_trace", {}).get("samples", [])
    observations: list[list[float]] = []
    for index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 0:
            continue
        observation = [float(value) for value in sample.get("observation", [])]
        if len(observation) != 32:
            raise ValueError(
                f"fixed Gate-2 prefix sample {index} must contain 32 values")
        observations.append(observation)
    if not observations:
        raise ValueError("fixed Gate-2 prefix report contains no Gate-1 samples")
    return observations


def _gate2_observation(
    observation: Sequence[float],
    *,
    runtime_duration_s: float,
    gate2_time_limit_s: float,
    world_frame_gate_features: bool = False,
    linear_gate_features: bool = False,
    forward_gate_rate_world_x: float | None = None,
) -> list[float]:
    values = [float(value) for value in observation]
    if len(values) != 32:
        raise ValueError("VQ2 policy observation must contain 32 values")
    if runtime_duration_s > 0.0 or gate2_time_limit_s > 0.0:
        if runtime_duration_s <= 0.0 or gate2_time_limit_s <= 0.0:
            raise ValueError("both VQ2 clock-normalization durations are required")
        values[18] = max(
            0.0,
            min(1.0, values[18] * runtime_duration_s / gate2_time_limit_s),
        )
    if world_frame_gate_features and values[10] > 0.5:
        forward_rate = math.atanh(max(-0.999999, min(0.999999, values[0]))) * 5.0
        right_rate = math.atanh(max(-0.999999, min(0.999999, values[1]))) * 3.0
        down_rate = math.atanh(max(-0.999999, min(0.999999, values[2]))) * 3.0
        forward = math.atanh(max(-0.999999, min(0.999999, values[11]))) * 10.0
        right = math.atanh(max(-0.999999, min(0.999999, values[12]))) * 5.0
        down = math.atanh(max(-0.999999, min(0.999999, values[13]))) * 5.0
        qw, qx, qy, qz = values[6:10]

        def rotate(vector: tuple[float, float, float]) -> tuple[float, float, float]:
            vx, vy, vz = vector
            tx = 2.0 * (qy * vz - qz * vy)
            ty = 2.0 * (qz * vx - qx * vz)
            tz = 2.0 * (qx * vy - qy * vx)
            return (
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx),
            )

        world_rate = rotate((forward_rate, right_rate, -down_rate))
        world_position = rotate((forward, right, -down))
        values[0] = math.tanh(world_rate[0] / 5.0)
        values[2] = math.tanh(world_rate[2] / 3.0)
        values[13] = math.tanh(world_position[2] / 5.0)
    if forward_gate_rate_world_x is not None:
        if not world_frame_gate_features:
            raise ValueError("forward gate-rate prediction requires world-frame features")
        values[0] = math.tanh(float(forward_gate_rate_world_x) / 5.0)
    if linear_gate_features and values[10] > 0.5:
        for index in (0, 1, 2, 11, 12, 13):
            values[index] = max(
                -1.0,
                min(1.0, math.atanh(max(-0.999999, min(0.999999, values[index])))),
            )
    return values


def _advance_forward_gate_rate_world_x(
    gate_rate_world_x: float,
    observation: Sequence[float],
    dt_s: float,
) -> float:
    """One legal calibrated-plant prediction from attitude and prior thrust."""
    values = [float(value) for value in observation]
    thrust_action = max(-1.0, min(1.0, values[21]))
    command_thrust = 0.27 + thrust_action * (
        0.42 - 0.27 if thrust_action >= 0.0 else 0.27 - 0.18)
    qw, qx, qy, qz = values[6:10]
    thrust_accel = 32.81 * command_thrust
    thrust_world_x = -2.0 * thrust_accel * (qw * qy + qx * qz) * 2.60794
    velocity_world_x = -float(gate_rate_world_x)
    accel_world_x = thrust_world_x - 0.14 * velocity_world_x
    return -(velocity_world_x + accel_world_x * dt_s)


def _rotate_vector_by_quaternion(
    vector: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, float, float]:
    vx, vy, vz = (float(value) for value in vector)
    qw, qx, qy, qz = (float(value) for value in quaternion)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _apply_gate_kinematic_predictor(
    state: _CompositeState, observation: Sequence[float]
) -> list[float]:
    values = [float(value) for value in observation]
    quaternion = values[6:10]
    if state.gate_position_world is None:
        if state.initial_gate_position_world is not None:
            state.gate_position_world = state.initial_gate_position_world
        else:
            body_up = (
                math.atanh(max(-0.999999, min(0.999999, values[11]))) * 10.0,
                math.atanh(max(-0.999999, min(0.999999, values[12]))) * 5.0,
                -math.atanh(max(-0.999999, min(0.999999, values[13]))) * 5.0,
            )
            state.gate_position_world = _rotate_vector_by_quaternion(
                body_up, quaternion)
            state.gate_position_world = tuple(
                state.gate_position_world[index]
                * state.initial_gate_position_world_scale[index]
                for index in range(3)
            )
        state.gate_rate_world = state.initial_gate_rate_world
    else:
        thrust_action = max(-1.0, min(1.0, values[21]))
        command_thrust = 0.27 + thrust_action * (
            0.42 - 0.27 if thrust_action >= 0.0 else 0.27 - 0.18)
        thrust_world = list(_rotate_vector_by_quaternion(
            (0.0, 0.0, 32.81 * command_thrust), quaternion))
        thrust_world[0] *= -2.60794
        thrust_world[1] *= 0.107491
        thrust_world[2] -= 8.69
        previous_rate = state.gate_rate_world
        accel_world = tuple(
            thrust_world[index] + 0.14 * previous_rate[index]
            for index in range(3)
        )
        next_rate = tuple(
            previous_rate[index]
            - accel_world[index] * state.forward_predictor_dt_s
            for index in range(3)
        )
        state.gate_position_world = tuple(
            state.gate_position_world[index]
            + 0.5 * (previous_rate[index] + next_rate[index])
            * state.forward_predictor_dt_s
            for index in range(3)
        )
        state.gate_rate_world = next_rate

    if state.association_reseed_pending:
        raw_body_ned = _visible_gate_pose_body_ned(values)
        if raw_body_ned is not None:
            qw, qx, qy, qz = quaternion
            conjugate = (qw, -qx, -qy, -qz)
            predicted_body_up = _rotate_vector_by_quaternion(
                state.gate_position_world, conjugate)
            raw_body_up = (
                raw_body_ned[0], raw_body_ned[1], -raw_body_ned[2])
            if raw_body_up[0] > 1e-3 and predicted_body_up[0] > 1e-3:
                scale = predicted_body_up[0] / raw_body_up[0]
                corrected_body_up = tuple(value * scale for value in raw_body_up)
                state.gate_position_world = _rotate_vector_by_quaternion(
                    corrected_body_up, quaternion)
                state.association_reseed_pending = False

    qw, qx, qy, qz = quaternion
    conjugate = (qw, -qx, -qy, -qz)
    body_position_up = _rotate_vector_by_quaternion(
        state.gate_position_world, conjugate)
    body_rate_up = _rotate_vector_by_quaternion(state.gate_rate_world, conjugate)
    values[0] = math.tanh(body_rate_up[0] / 5.0)
    values[1] = math.tanh(body_rate_up[1] / 3.0)
    values[2] = math.tanh(-body_rate_up[2] / 3.0)
    values[11] = math.tanh(body_position_up[0] / 10.0)
    values[12] = math.tanh(body_position_up[1] / 5.0)
    values[13] = math.tanh(-body_position_up[2] / 5.0)
    return values


def _visible_gate_pose_body_ned(
    observation: Sequence[float],
) -> tuple[float, float, float] | None:
    values = [float(value) for value in observation]
    if len(values) != 32 or values[10] <= 0.5:
        return None
    return (
        math.atanh(max(-0.999999, min(0.999999, values[11]))) * 10.0,
        math.atanh(max(-0.999999, min(0.999999, values[12]))) * 5.0,
        math.atanh(max(-0.999999, min(0.999999, values[13]))) * 5.0,
    )


def _update_gate2_association(
    state: _CompositeState,
    observation: Sequence[float],
    gate: int,
) -> bool:
    if state.association_jump_threshold_m <= 0.0:
        state.association_ready = True
        return True
    pose = _visible_gate_pose_body_ned(observation)
    if gate < 1:
        if pose is not None:
            state.last_gate_pose_body_ned = pose
        state.association_ready = False
        return False
    if state.association_ready:
        return True
    if pose is None:
        return False
    previous = state.last_gate_pose_body_ned
    state.last_gate_pose_body_ned = pose
    if previous is None:
        return False
    jump = math.sqrt(sum(
        (pose[index] - previous[index]) ** 2 for index in range(3)
    ))
    if jump >= state.association_jump_threshold_m:
        state.association_ready = True
        state.association_reseed_pending = state.reseed_bearing_on_association
    return state.association_ready


def _resolve() -> _CompositeState:
    global _KEY, _STATE
    dimensions = (
        int(os.getenv("PUFFER_POLICY_INPUT_DIM", "32")),
        int(os.getenv("PUFFER_POLICY_HIDDEN_DIM", "128")),
        int(os.getenv("PUFFER_POLICY_NUM_LAYERS", "3")),
        int(os.getenv("PUFFER_POLICY_NUM_ACTIONS", "4")),
        int(os.getenv("PUFFER_POLICY_LAYOUT_PRECISION_BYTES", "4")),
        _env_bool("PUFFER_POLICY_NATIVE_BF16", False),
    )
    fixed_prefix_report = _optional_path(
        "PUFFER_POLICY_GATE2_FIXED_PREFIX_REPORT_PATH")
    runtime_duration_s = float(
        os.getenv("PUFFER_POLICY_RUNTIME_DURATION_SECONDS", "0"))
    gate2_time_limit_s = float(
        os.getenv("PUFFER_POLICY_GATE2_TIME_LIMIT_SECONDS", "0"))
    world_frame_gate_features = _env_bool(
        "PUFFER_POLICY_GATE2_WORLD_FRAME_FEATURES", False)
    linear_gate_features = _env_bool(
        "PUFFER_POLICY_GATE2_LINEAR_GATE_FEATURES", False)
    predict_forward_gate_rate = _env_bool(
        "PUFFER_POLICY_GATE2_PREDICT_FORWARD_GATE_RATE", False)
    predict_gate_kinematics = _env_bool(
        "PUFFER_POLICY_GATE2_PREDICT_GATE_KINEMATICS", False)
    initial_forward_gate_rate_world_x = float(os.getenv(
        "PUFFER_POLICY_GATE2_INITIAL_FORWARD_GATE_RATE_M_S", "0"))
    forward_predictor_dt_s = float(os.getenv(
        "PUFFER_POLICY_GATE2_FORWARD_PREDICTOR_DT_SECONDS", str(1.0 / 60.0)))
    initial_gate_rate_world = (
        initial_forward_gate_rate_world_x,
        float(os.getenv("PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Y_M_S", "0")),
        float(os.getenv("PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Z_M_S", "0")),
    )
    initial_gate_position_world_scale = (
        float(os.getenv(
            "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_X", "1")),
        float(os.getenv(
            "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Y", "1")),
        float(os.getenv(
            "PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Z", "1")),
    )
    initial_gate_position_values = tuple(
        os.getenv(
            f"PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_{axis}", ""
        ).strip()
        for axis in ("X", "Y", "Z")
    )
    if any(initial_gate_position_values) and not all(initial_gate_position_values):
        raise ValueError("all three initial Gate-2 world-position values are required")
    initial_gate_position_world = (
        tuple(float(value) for value in initial_gate_position_values)
        if all(initial_gate_position_values)
        else None
    )
    association_jump_threshold_m = float(os.getenv(
        "PUFFER_POLICY_GATE2_ASSOCIATION_JUMP_THRESHOLD_M", "0"))
    reseed_bearing_on_association = _env_bool(
        "PUFFER_POLICY_GATE2_RESEED_BEARING_ON_ASSOCIATION", False)
    if (
        not math.isfinite(runtime_duration_s)
        or not math.isfinite(gate2_time_limit_s)
        or runtime_duration_s < 0.0
        or gate2_time_limit_s < 0.0
        or (runtime_duration_s > 0.0) != (gate2_time_limit_s > 0.0)
        or not math.isfinite(initial_forward_gate_rate_world_x)
        or not math.isfinite(forward_predictor_dt_s)
        or forward_predictor_dt_s <= 0.0
        or not all(math.isfinite(value) for value in initial_gate_rate_world)
        or not all(
            math.isfinite(value) and value > 0.0
            for value in initial_gate_position_world_scale
        )
        or (
            initial_gate_position_world is not None
            and not all(math.isfinite(value) for value in initial_gate_position_world)
        )
        or not math.isfinite(association_jump_threshold_m)
        or association_jump_threshold_m < 0.0
        or (reseed_bearing_on_association and association_jump_threshold_m <= 0.0)
    ):
        raise ValueError(
            "VQ2 clock normalization needs two finite positive durations")
    key = (
        _required_path("PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"),
        _required_path("PUFFER_POLICY_GATE2_CHECKPOINT_PATH"),
        *dimensions,
        fixed_prefix_report,
        runtime_duration_s,
        gate2_time_limit_s,
        world_frame_gate_features,
        linear_gate_features,
        predict_forward_gate_rate,
        initial_forward_gate_rate_world_x,
        forward_predictor_dt_s,
        predict_gate_kinematics,
        initial_gate_rate_world,
        initial_gate_position_world_scale,
        initial_gate_position_world,
        association_jump_threshold_m,
        reseed_bearing_on_association,
    )
    if _STATE is None or _KEY != key:
        load = dict(
            input_dim=dimensions[0],
            hidden_dim=dimensions[1],
            num_layers=dimensions[2],
            num_actions=dimensions[3],
            layout_precision_bytes=dimensions[4],
            native_bf16=dimensions[5],
        )
        prefix = CheckpointPolicy.load(key[0], **load)
        gate2 = CheckpointPolicy.load(key[1], **load)
        gate2_warm_state = None
        if fixed_prefix_report is not None:
            for observation in _gate0_prefix_observations(fixed_prefix_report):
                gate2.infer(observation)
            gate2_warm_state = gate2.state.copy()
        _STATE = _CompositeState(
            prefix=prefix,
            gate2=gate2,
            gate2_warm_state=gate2_warm_state,
            runtime_duration_s=runtime_duration_s,
            gate2_time_limit_s=gate2_time_limit_s,
            world_frame_gate_features=world_frame_gate_features,
            linear_gate_features=linear_gate_features,
            predict_forward_gate_rate=predict_forward_gate_rate,
            initial_forward_gate_rate_world_x=initial_forward_gate_rate_world_x,
            forward_gate_rate_world_x=initial_forward_gate_rate_world_x,
            forward_predictor_dt_s=forward_predictor_dt_s,
            predict_gate_kinematics=predict_gate_kinematics,
            initial_gate_rate_world=initial_gate_rate_world,
            initial_gate_position_world_scale=initial_gate_position_world_scale,
            initial_gate_position_world=initial_gate_position_world,
            gate_rate_world=initial_gate_rate_world,
            association_jump_threshold_m=association_jump_threshold_m,
            reseed_bearing_on_association=reseed_bearing_on_association,
            association_ready=association_jump_threshold_m <= 0.0,
        )
        _KEY = key
    return _STATE


def _gate_index(observation: Sequence[float]) -> int:
    progress_index = int(os.getenv("PUFFER_POLICY_PROGRESS_INDEX", "23"))
    denominator = int(os.getenv("PUFFER_POLICY_RACE_PHASE_DENOMINATOR", "6"))
    if denominator <= 0:
        raise ValueError("PUFFER_POLICY_RACE_PHASE_DENOMINATOR must be positive")
    if progress_index < 0 or progress_index >= len(observation):
        raise ValueError("progress observation index is outside the policy input")
    scaled = float(observation[progress_index]) * denominator
    gate = int(round(scaled))
    if abs(scaled - gate) > 1e-3 or gate < 0 or gate > denominator:
        raise ValueError(f"invalid quantized race progress observation: {scaled}")
    return gate


def reset() -> None:
    state = _resolve()
    state.prefix.reset_state()
    if state.gate2_warm_state is None:
        state.gate2.reset_state()
    else:
        state.gate2.state[...] = state.gate2_warm_state
    state.forward_gate_rate_world_x = state.initial_forward_gate_rate_world_x
    state.forward_predictor_started = False
    state.gate_position_world = None
    state.gate_rate_world = state.initial_gate_rate_world
    state.association_ready = state.association_jump_threshold_m <= 0.0
    state.association_reseed_pending = False
    state.last_gate_pose_body_ned = None


def infer(observation: Sequence[float]) -> list[float]:
    state = _resolve()
    gate = _gate_index(observation)
    prefix_actions = state.prefix.infer(observation)
    association_ready = _update_gate2_association(state, observation, gate)
    if gate < 1 and state.gate2_warm_state is not None:
        gate2_actions = None
    else:
        gate2_observation = [float(value) for value in observation]
        if gate >= 1 and state.predict_gate_kinematics:
            gate2_observation = _apply_gate_kinematic_predictor(
                state, gate2_observation)
        predicted_forward_rate = None
        if gate >= 1 and state.predict_forward_gate_rate:
            if state.forward_predictor_started:
                state.forward_gate_rate_world_x = _advance_forward_gate_rate_world_x(
                state.forward_gate_rate_world_x,
                    gate2_observation,
                    state.forward_predictor_dt_s,
                )
            else:
                state.forward_predictor_started = True
            predicted_forward_rate = state.forward_gate_rate_world_x
        gate2_actions = state.gate2.infer(
            _gate2_observation(
                gate2_observation,
                runtime_duration_s=state.runtime_duration_s,
                gate2_time_limit_s=state.gate2_time_limit_s,
                world_frame_gate_features=state.world_frame_gate_features,
                linear_gate_features=state.linear_gate_features,
                forward_gate_rate_world_x=predicted_forward_rate,
            )
        )
    assert gate2_actions is not None or gate < 1
    return gate2_actions if gate >= 1 and association_ready else prefix_actions
