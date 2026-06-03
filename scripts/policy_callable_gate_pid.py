#!/usr/bin/env python3
"""Deterministic competition-shaped policy callable baseline.

Usage with the SITL smoke runner:
  --control-mode policy
  --policy-callable scripts/policy_callable_gate_pid.py:infer
"""

from __future__ import annotations

from collections.abc import Sequence


OBS_GATE_VISIBLE = 10
OBS_GATE_FORWARD = 11
OBS_GATE_RIGHT = 12
OBS_GATE_DOWN = 13
OBS_GATE_YAW_ERROR = 14
OBS_GATE_PITCH_ERROR = 15
OBS_GATE_APPARENT_SIZE = 16
OBS_GATE_CONFIDENCE = 17
OBSERVATION_SIZE = 23
ACTION_SIZE = 4


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def infer(observation: Sequence[float]) -> list[float]:
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"expected observation size {OBSERVATION_SIZE}, got {len(observation)}")

    gate_visible = float(observation[OBS_GATE_VISIBLE])
    gate_forward = float(observation[OBS_GATE_FORWARD])
    gate_right = float(observation[OBS_GATE_RIGHT])
    gate_down = float(observation[OBS_GATE_DOWN])
    gate_yaw_error = float(observation[OBS_GATE_YAW_ERROR])
    gate_pitch_error = float(observation[OBS_GATE_PITCH_ERROR])
    gate_apparent_size = float(observation[OBS_GATE_APPARENT_SIZE])
    gate_confidence = max(0.0, float(observation[OBS_GATE_CONFIDENCE]))

    if gate_visible < 0.0:
        return [0.0, 0.0, 0.0, 0.0]

    # Confidence-based blending between cautious and nominal gains.
    conf = clamp(gate_confidence, 0.0, 1.0)
    lateral_gain = 0.35 + 0.65 * conf
    vertical_gain = 0.35 + 0.65 * conf
    yaw_gain = 0.4 + 0.6 * conf

    # Forward command keeps progress while reducing speed when the gate is very large.
    forward_bias = 0.25 + 0.55 * conf
    forward = forward_bias + 0.8 * max(0.0, gate_forward) - 0.55 * max(0.0, gate_apparent_size)
    forward = clamp(forward, 0.0, 1.0)

    right = clamp(lateral_gain * gate_right + 0.6 * gate_yaw_error, -1.0, 1.0)
    down = clamp(vertical_gain * gate_down + 0.4 * gate_pitch_error, -1.0, 1.0)
    yaw_rate = clamp(yaw_gain * gate_yaw_error, -1.0, 1.0)

    action = [forward, right, down, yaw_rate]
    if len(action) != ACTION_SIZE:
        raise RuntimeError("internal action size mismatch")
    return action
