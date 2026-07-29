#!/usr/bin/env python3
"""N221: one observable fixed-course controller for all six VQ1 gates.

N219 remains the yaw source.  A single controller owns pitch, roll, and thrust
from the official start through Gate 6, eliminating the failed live prefix/tail
handoff.  Runtime inputs remain the deployed 32-value observation and fixed
course prior; no simulator position or velocity is read.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import policy_callable_n220_observable_tail as n220


TARGET_FORWARD_SPEED_M_S = 4.00


def reset() -> None:
    n220.n219.reset()
    n220._CONTROLLER.reset()


def _estimated_forward_speed(observation: np.ndarray) -> float:
    q = np.asarray(observation[6:10], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 1e-6:
        q /= norm
    else:
        q[:] = (1.0, 0.0, 0.0, 0.0)
    gate_rate_body = np.asarray(
        (
            n220._unscale_tanh(observation[0], 0.2),
            n220._unscale_tanh(observation[1], 1.0 / 3.0),
            -n220._unscale_tanh(observation[2], 1.0 / 3.0),
        ),
        dtype=np.float64,
    )
    motion_velocity = -n220._quat_rotate(q, gate_rate_body)
    blended_velocity = (
        0.5 * n220._CONTROLLER.velocity_world + 0.5 * motion_velocity
    )
    return float(blended_velocity[0])


def infer(observation: Sequence[float]) -> list[float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")

    action = n220._CONTROLLER.act(values, n220.n219.infer(values))
    speed_error = TARGET_FORWARD_SPEED_M_S - _estimated_forward_speed(values)
    pitch_error = n220._clamp(-speed_error * 0.35, -0.60, 0.60)
    action[0] = 0.24 * pitch_error
    return action


def controller_snapshot() -> dict:
    return {
        "n221_observable_course": {
            "previous_gate": n220._CONTROLLER.previous_gate,
            "parameters": {
                "observable_controller_gate_indices": [0, 1, 2, 3, 4, 5],
                "policy_owned_actions": ["yaw"],
                "controller_owned_actions": ["pitch", "roll", "thrust"],
                "target_forward_speed_m_s": TARGET_FORWARD_SPEED_M_S,
                "runtime_privileged_state": False,
                "motion_rate_blend": 0.5,
                "gate_roll_bias": n220.ROLL_BIAS.tolist(),
                "control_aware_gate_predictor_start_index": 0,
            },
        },
        "n219_yaw_source": n220.n219.controller_snapshot(),
    }
