#!/usr/bin/env python3
"""Competition-facing drone policy observation/action contract.

The native v4 env and the SITL controller both use this contract when training
or running policies intended to transfer through the TS-002 observable boundary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections.abc import Sequence


OBSERVATION_FIELDS = (
    "body_vel_forward_norm",
    "body_vel_right_norm",
    "body_vel_down_norm",
    "gyro_roll_rate_norm",
    "gyro_pitch_rate_norm",
    "gyro_yaw_rate_norm",
    "attitude_quat_w",
    "attitude_quat_x",
    "attitude_quat_y",
    "attitude_quat_z",
    "gate_visible",
    "gate_forward_norm",
    "gate_right_norm",
    "gate_down_norm",
    "gate_yaw_error_norm",
    "gate_pitch_error_norm",
    "gate_apparent_size_norm",
    "gate_pose_confidence",
    "elapsed_time_fraction",
    "last_cmd_forward_norm",
    "last_cmd_right_norm",
    "last_cmd_down_norm",
    "last_cmd_yaw_rate_norm",
)

ACTION_FIELDS = (
    "cmd_forward_norm",
    "cmd_right_norm",
    "cmd_down_norm",
    "cmd_yaw_rate_norm",
)

OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)


@dataclasses.dataclass(frozen=True)
class PolicyActionScales:
    max_forward_m_s: float = 2.0
    max_right_m_s: float = 1.0
    max_down_m_s: float = 0.8
    max_yaw_rate_rad_s: float = 1.0


@dataclasses.dataclass(frozen=True)
class PolicyAction:
    normalized: tuple[float, float, float, float]
    forward_m_s: float
    right_m_s: float
    down_m_s: float
    yaw_rate_rad_s: float


@dataclasses.dataclass(frozen=True)
class MavlinkVelocityYawSetpoint:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yaw_rate_rad_s: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite_float(value, *, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def normalize_action(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != ACTION_SIZE:
        raise ValueError(f"expected {ACTION_SIZE} action values")
    return tuple(clamp(_finite_float(value, field_name=ACTION_FIELDS[i]), -1.0, 1.0)
        for i, value in enumerate(values))


def decode_policy_action(
    values: Sequence[float],
    *,
    yaw_rad: float,
    scales: PolicyActionScales | None = None,
) -> tuple[PolicyAction, MavlinkVelocityYawSetpoint]:
    scales = scales or PolicyActionScales()
    action = normalize_action(values)
    forward = action[0] * scales.max_forward_m_s
    right = action[1] * scales.max_right_m_s
    down = action[2] * scales.max_down_m_s
    yaw_rate = action[3] * scales.max_yaw_rate_rad_s

    yaw = _finite_float(yaw_rad, field_name="yaw_rad")
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx = c * forward - s * right
    vy = s * forward + c * right

    return (
        PolicyAction(
            normalized=action,
            forward_m_s=forward,
            right_m_s=right,
            down_m_s=down,
            yaw_rate_rad_s=yaw_rate,
        ),
        MavlinkVelocityYawSetpoint(
            vx_m_s=vx,
            vy_m_s=vy,
            vz_m_s=down,
            yaw_rate_rad_s=yaw_rate,
        ),
    )


def validate_observation(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != OBSERVATION_SIZE:
        raise ValueError(f"expected {OBSERVATION_SIZE} observation values")
    out = tuple(_finite_float(value, field_name=OBSERVATION_FIELDS[i])
        for i, value in enumerate(values))
    bad = [
        OBSERVATION_FIELDS[i]
        for i, value in enumerate(out)
        if value < -1.000001 or value > 1.000001
    ]
    if bad:
        raise ValueError(f"observation fields outside [-1, 1]: {', '.join(bad)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a normalized drone policy action")
    parser.add_argument("--action-json", required=True, help="JSON list of four normalized action values")
    parser.add_argument("--yaw-rad", type=float, default=0.0)
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    action, setpoint = decode_policy_action(json.loads(args.action_json), yaw_rad=args.yaw_rad)
    report = {
        "action": dataclasses.asdict(action),
        "setpoint": dataclasses.asdict(setpoint),
        "observation_fields": OBSERVATION_FIELDS,
        "action_fields": ACTION_FIELDS,
    }
    if args.json_path:
        directory = os.path.dirname(args.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
