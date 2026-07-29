#!/usr/bin/env python3
"""Preserve a recurrent prefix and hand later gates to a tiny visual servo.

The official runner replaces observation slot 22 with normalized
``active_gate_index`` when ``--policy-race-phase-observation`` is enabled.  For
the retained recurrent checkpoint, this callable restores the previous yaw
action before advancing the network, exactly like the existing phase-head
adapter.  Gates before ``PUFFER_FINAL_GATE_HANDOFF_INDEX`` therefore keep the
unchanged checkpoint action.

At and after the handoff index, the output uses only TS-002 observables already present
in the 23-float observation: camera-relative gate pose and IMU-derived
attitude.  Each official gate-index transition starts a fresh association,
delay, and bounded-pulse epoch.  It levels roll, brakes while pointing yaw at
the gate, advances only after alignment, and regulates vertical offset with
hover-centered thrust.  No simulator coordinates or privileged state are used.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Sequence

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy


@dataclasses.dataclass(frozen=True)
class FinalGateHandoffConfig:
    handoff_gate_index: int = 3
    phase_denominator: int = 3
    gate3_correction_index: int = -1
    gate3_close_forward_m: float = 8.0
    gate3_early_bank_forward_m: float = 0.0
    gate3_early_bank_min_closing_rate_m_s: float = 0.0
    gate3_severe_bank_forward_m: float = 0.0
    gate3_severe_bank_min_closing_rate_m_s: float = 0.0
    gate3_severe_intercept_target_right_m: float = 0.0
    gate3_severe_roll_floor_rad: float = 0.0
    gate3_left_trigger_m: float = -0.8
    gate3_roll_floor_rad: float = 0.22
    gate3_brake_pitch_rad: float = 0.0
    gate3_brake_min_closing_rate_m_s: float = 0.0
    gate3_brake_release_closing_rate_m_s: float = 0.0
    gate3_search_brake_pitch_rad: float = 0.0
    gate3_level_roll_during_search: bool = False
    gate3_centered_approach_pitch_rad: float = 0.0
    gate3_lateral_kp: float = 0.0
    gate3_lateral_kd: float = 0.0
    gate3_intercept_plane_forward_m: float = 0.0
    gate3_intercept_target_right_m: float = 0.0
    gate3_counter_roll_rad: float = 0.0
    gate3_min_closing_rate_m_s: float = 1.0
    gate3_counter_latch_forward_m: float = 0.0
    gate3_latched_level_roll_threshold_rad: float = 0.10
    gate3_tilt_compensated_hover_thrust: float = 0.0
    max_pitch_rad: float = 0.12
    aligned_forward_pitch_rad: float = 0.06
    center_before_accel_yaw_error_rad: float = 0.12
    center_before_accel_down_error_m: float = 1.5
    hover_thrust: float = 0.27
    min_thrust: float = 0.18
    max_thrust: float = 0.42
    vertical_thrust_kp: float = 0.02
    final_gate_max_pose_jump_m: float = 0.0
    final_gate_max_anchor_drift_m: float = 0.0
    final_gate_association_grace_s: float = 0.0
    final_gate_freeze_yaw_on_dropout: bool = False
    final_gate_reanchor_persistence_s: float = 0.0
    final_gate_reanchor_max_forward_m: float = 0.0
    final_gate_reanchor_max_count: int = 1
    final_gate_close_reanchor_persistence_s: float = 0.0
    final_gate_close_reanchor_max_right_m: float = 0.0
    final_gate_close_reanchor_max_count: int = 1
    final_gate_crossing_reanchor_persistence_s: float = 0.0
    final_gate_crossing_reanchor_max_forward_m: float = 0.0
    final_gate_crossing_reanchor_max_right_m: float = 0.0
    final_gate_crossing_reanchor_max_abs_down_m: float = 0.0
    final_gate_crossing_reanchor_max_count: int = 1
    final_gate_lateral_roll_kp: float = 0.0
    final_gate_max_lateral_roll_rad: float = 0.0
    final_gate_hold_pulse_lateral: bool = False
    final_gate_hover_approach_pitch_rad: float = 0.0
    final_gate_hover_approach_max_forward_m: float = 0.0
    final_gate_descent_delay_s: float = 0.0
    final_gate_descent_max_forward_m: float = 0.0
    final_gate_descent_later_max_forward_m: float = 0.0
    final_gate_descent_max_abs_right_m: float = 0.0
    final_gate_descent_pulse_s: float = 0.0
    final_gate_descent_later_pulse_s: float = 0.0
    final_gate_descent_pulse_max_count: int = 1
    final_gate_descent_pulse_cooldown_s: float = 0.0
    final_gate_descent_thrust: float = 0.27
    final_gate_level_pitch_during_descent_pulse: bool = False
    final_gate_recovery_pitch_rad: float = 0.0


_MODEL: CheckpointPolicy | None = None
_MODEL_KEY: tuple[str, int, int, int, int] | None = None
_LAST_YAW_ACTION = 0.0
_LAST_FINAL_ACTION: list[float] | None = None
_LAST_FINAL_GATE_VECTOR: tuple[float, float, float] | None = None
_FINAL_GATE_ANCHOR_VECTOR: tuple[float, float, float] | None = None
_GATE3_COUNTERBANK_LATCHED = False
_GATE3_SPEED_BRAKE_LATCHED = False
_FINAL_GATE_PHASE_STARTED_S: float | None = None
_FINAL_GATE_DESCENT_PULSE_STARTED_S: float | None = None
_FINAL_GATE_DESCENT_PULSE_VECTOR: tuple[float, float, float] | None = None
_FINAL_GATE_DESCENT_PULSE_COUNT = 0
_FINAL_GATE_DESCENT_PULSE_LAST_END_S: float | None = None
_FINAL_GATE_REJECTED_CLUSTER_STARTED_S: float | None = None
_FINAL_GATE_REJECTED_CLUSTER_VECTOR: tuple[float, float, float] | None = None
_FINAL_GATE_REANCHOR_COUNT = 0
_FINAL_GATE_CLOSE_REANCHOR_COUNT = 0
_FINAL_GATE_CROSSING_REANCHOR_COUNT = 0
_POST_PREFIX_GATE_INDEX: int | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return default if not value else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return default if not value else float(value)


def load_config() -> FinalGateHandoffConfig:
    config = FinalGateHandoffConfig(
        handoff_gate_index=_env_int("PUFFER_FINAL_GATE_HANDOFF_INDEX", 3),
        phase_denominator=_env_int("PUFFER_POLICY_RACE_PHASE_DENOMINATOR", 3),
        gate3_correction_index=_env_int(
            "PUFFER_GATE3_CLOSE_CORRECTION_INDEX", -1
        ),
        gate3_close_forward_m=_env_float("PUFFER_GATE3_CLOSE_FORWARD_M", 8.0),
        gate3_early_bank_forward_m=_env_float(
            "PUFFER_GATE3_EARLY_BANK_FORWARD_M", 0.0
        ),
        gate3_early_bank_min_closing_rate_m_s=_env_float(
            "PUFFER_GATE3_EARLY_BANK_MIN_CLOSING_RATE_M_S", 0.0
        ),
        gate3_severe_bank_forward_m=_env_float(
            "PUFFER_GATE3_SEVERE_BANK_FORWARD_M", 0.0
        ),
        gate3_severe_bank_min_closing_rate_m_s=_env_float(
            "PUFFER_GATE3_SEVERE_BANK_MIN_CLOSING_RATE_M_S", 0.0
        ),
        gate3_severe_intercept_target_right_m=_env_float(
            "PUFFER_GATE3_SEVERE_INTERCEPT_TARGET_RIGHT_M", 0.0
        ),
        gate3_severe_roll_floor_rad=_env_float(
            "PUFFER_GATE3_SEVERE_ROLL_FLOOR_RAD", 0.0
        ),
        gate3_left_trigger_m=_env_float("PUFFER_GATE3_LEFT_TRIGGER_M", -0.8),
        gate3_roll_floor_rad=_env_float("PUFFER_GATE3_ROLL_FLOOR_RAD", 0.22),
        gate3_brake_pitch_rad=_env_float("PUFFER_GATE3_BRAKE_PITCH_RAD", 0.0),
        gate3_brake_min_closing_rate_m_s=_env_float(
            "PUFFER_GATE3_BRAKE_MIN_CLOSING_RATE_M_S", 0.0
        ),
        gate3_brake_release_closing_rate_m_s=_env_float(
            "PUFFER_GATE3_BRAKE_RELEASE_CLOSING_RATE_M_S", 0.0
        ),
        gate3_search_brake_pitch_rad=_env_float(
            "PUFFER_GATE3_SEARCH_BRAKE_PITCH_RAD", 0.0
        ),
        gate3_level_roll_during_search=bool(
            _env_int("PUFFER_GATE3_LEVEL_ROLL_DURING_SEARCH", 0)
        ),
        gate3_centered_approach_pitch_rad=_env_float(
            "PUFFER_GATE3_CENTERED_APPROACH_PITCH_RAD", 0.0
        ),
        gate3_lateral_kp=_env_float("PUFFER_GATE3_LATERAL_KP", 0.0),
        gate3_lateral_kd=_env_float("PUFFER_GATE3_LATERAL_KD", 0.0),
        gate3_intercept_plane_forward_m=_env_float(
            "PUFFER_GATE3_INTERCEPT_PLANE_FORWARD_M", 0.0
        ),
        gate3_intercept_target_right_m=_env_float(
            "PUFFER_GATE3_INTERCEPT_TARGET_RIGHT_M", 0.0
        ),
        gate3_counter_roll_rad=_env_float(
            "PUFFER_GATE3_COUNTER_ROLL_RAD", 0.0
        ),
        gate3_min_closing_rate_m_s=_env_float(
            "PUFFER_GATE3_MIN_CLOSING_RATE_M_S", 1.0
        ),
        gate3_counter_latch_forward_m=_env_float(
            "PUFFER_GATE3_COUNTER_LATCH_FORWARD_M", 0.0
        ),
        gate3_latched_level_roll_threshold_rad=_env_float(
            "PUFFER_GATE3_LATCHED_LEVEL_ROLL_THRESHOLD_RAD", 0.10
        ),
        gate3_tilt_compensated_hover_thrust=_env_float(
            "PUFFER_GATE3_TILT_COMPENSATED_HOVER_THRUST", 0.0
        ),
        max_pitch_rad=_env_float("PUFFER_FINAL_GATE_MAX_PITCH_RAD", 0.12),
        aligned_forward_pitch_rad=_env_float(
            "PUFFER_FINAL_GATE_ALIGNED_FORWARD_PITCH_RAD", 0.06
        ),
        center_before_accel_yaw_error_rad=_env_float(
            "PUFFER_FINAL_GATE_CENTER_BEFORE_ACCEL_YAW_ERROR_RAD", 0.12
        ),
        center_before_accel_down_error_m=_env_float(
            "PUFFER_FINAL_GATE_CENTER_BEFORE_ACCEL_DOWN_ERROR_M", 1.5
        ),
        hover_thrust=_env_float("PUFFER_FINAL_GATE_HOVER_THRUST", 0.27),
        min_thrust=_env_float("PUFFER_FINAL_GATE_MIN_THRUST", 0.18),
        max_thrust=_env_float("PUFFER_FINAL_GATE_MAX_THRUST", 0.42),
        vertical_thrust_kp=_env_float(
            "PUFFER_FINAL_GATE_VERTICAL_THRUST_KP", 0.02
        ),
        final_gate_max_pose_jump_m=_env_float(
            "PUFFER_FINAL_GATE_MAX_POSE_JUMP_M", 0.0
        ),
        final_gate_max_anchor_drift_m=_env_float(
            "PUFFER_FINAL_GATE_MAX_ANCHOR_DRIFT_M", 0.0
        ),
        final_gate_association_grace_s=_env_float(
            "PUFFER_FINAL_GATE_ASSOCIATION_GRACE_S", 0.0
        ),
        final_gate_freeze_yaw_on_dropout=bool(
            _env_int("PUFFER_FINAL_GATE_FREEZE_YAW_ON_DROPOUT", 0)
        ),
        final_gate_reanchor_persistence_s=_env_float(
            "PUFFER_FINAL_GATE_REANCHOR_PERSISTENCE_S", 0.0
        ),
        final_gate_reanchor_max_forward_m=_env_float(
            "PUFFER_FINAL_GATE_REANCHOR_MAX_FORWARD_M", 0.0
        ),
        final_gate_reanchor_max_count=_env_int(
            "PUFFER_FINAL_GATE_REANCHOR_MAX_COUNT", 1
        ),
        final_gate_close_reanchor_persistence_s=_env_float(
            "PUFFER_FINAL_GATE_CLOSE_REANCHOR_PERSISTENCE_S", 0.0
        ),
        final_gate_close_reanchor_max_right_m=_env_float(
            "PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_RIGHT_M", 0.0
        ),
        final_gate_close_reanchor_max_count=_env_int(
            "PUFFER_FINAL_GATE_CLOSE_REANCHOR_MAX_COUNT", 1
        ),
        final_gate_crossing_reanchor_persistence_s=_env_float(
            "PUFFER_FINAL_GATE_CROSSING_REANCHOR_PERSISTENCE_S", 0.0
        ),
        final_gate_crossing_reanchor_max_forward_m=_env_float(
            "PUFFER_FINAL_GATE_CROSSING_REANCHOR_MAX_FORWARD_M", 0.0
        ),
        final_gate_crossing_reanchor_max_right_m=_env_float(
            "PUFFER_FINAL_GATE_CROSSING_REANCHOR_MAX_RIGHT_M", 0.0
        ),
        final_gate_crossing_reanchor_max_abs_down_m=_env_float(
            "PUFFER_FINAL_GATE_CROSSING_REANCHOR_MAX_ABS_DOWN_M", 0.0
        ),
        final_gate_crossing_reanchor_max_count=_env_int(
            "PUFFER_FINAL_GATE_CROSSING_REANCHOR_MAX_COUNT", 1
        ),
        final_gate_lateral_roll_kp=_env_float(
            "PUFFER_FINAL_GATE_LATERAL_ROLL_KP", 0.0
        ),
        final_gate_max_lateral_roll_rad=_env_float(
            "PUFFER_FINAL_GATE_MAX_LATERAL_ROLL_RAD", 0.0
        ),
        final_gate_hold_pulse_lateral=bool(
            _env_int("PUFFER_FINAL_GATE_HOLD_PULSE_LATERAL", 0)
        ),
        final_gate_hover_approach_pitch_rad=_env_float(
            "PUFFER_FINAL_GATE_HOVER_APPROACH_PITCH_RAD", 0.0
        ),
        final_gate_hover_approach_max_forward_m=_env_float(
            "PUFFER_FINAL_GATE_HOVER_APPROACH_MAX_FORWARD_M", 0.0
        ),
        final_gate_descent_delay_s=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_DELAY_S", 0.0
        ),
        final_gate_descent_max_forward_m=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_MAX_FORWARD_M", 0.0
        ),
        final_gate_descent_later_max_forward_m=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_LATER_MAX_FORWARD_M", 0.0
        ),
        final_gate_descent_max_abs_right_m=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_MAX_ABS_RIGHT_M", 0.0
        ),
        final_gate_descent_pulse_s=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_PULSE_S", 0.0
        ),
        final_gate_descent_later_pulse_s=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_LATER_PULSE_S", 0.0
        ),
        final_gate_descent_pulse_max_count=_env_int(
            "PUFFER_FINAL_GATE_DESCENT_PULSE_MAX_COUNT", 1
        ),
        final_gate_descent_pulse_cooldown_s=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_PULSE_COOLDOWN_S", 0.0
        ),
        final_gate_descent_thrust=_env_float(
            "PUFFER_FINAL_GATE_DESCENT_THRUST", 0.27
        ),
        final_gate_level_pitch_during_descent_pulse=bool(
            _env_int("PUFFER_FINAL_GATE_LEVEL_PITCH_DURING_DESCENT_PULSE", 0)
        ),
        final_gate_recovery_pitch_rad=_env_float(
            "PUFFER_FINAL_GATE_RECOVERY_PITCH_RAD", 0.0
        ),
    )
    if config.handoff_gate_index < 0:
        raise ValueError("PUFFER_FINAL_GATE_HANDOFF_INDEX must be nonnegative")
    if config.gate3_correction_index < -1:
        raise ValueError("Gate-3 correction index must be -1 or nonnegative")
    if config.gate3_close_forward_m <= 0.0:
        raise ValueError("Gate-3 correction range must be positive")
    if config.gate3_early_bank_forward_m < 0.0:
        raise ValueError("Gate-3 early-bank range must be nonnegative")
    if (
        config.gate3_early_bank_forward_m > 0.0
        and config.gate3_early_bank_forward_m <= config.gate3_close_forward_m
    ):
        raise ValueError("Gate-3 early-bank range must exceed the close range")
    if config.gate3_early_bank_min_closing_rate_m_s < 0.0:
        raise ValueError("Gate-3 early-bank speed threshold must be nonnegative")
    if (
        config.gate3_early_bank_forward_m > 0.0
        and config.gate3_early_bank_min_closing_rate_m_s <= 0.0
    ):
        raise ValueError("Gate-3 early-bank speed threshold must be positive")
    severe_bank_values = (
        config.gate3_severe_bank_forward_m,
        config.gate3_severe_bank_min_closing_rate_m_s,
        config.gate3_severe_roll_floor_rad,
    )
    if any(value > 0.0 for value in severe_bank_values):
        if not all(value > 0.0 for value in severe_bank_values):
            raise ValueError("Gate-3 severe bank requires range, speed, and roll")
        if config.gate3_intercept_plane_forward_m <= 0.0:
            raise ValueError("Gate-3 severe bank requires an intercept plane")
        if (
            config.gate3_severe_bank_forward_m
            <= config.gate3_intercept_plane_forward_m
        ):
            raise ValueError("Gate-3 severe-bank range must exceed the intercept plane")
        if (
            config.gate3_severe_intercept_target_right_m
            >= config.gate3_intercept_target_right_m
        ):
            raise ValueError(
                "Gate-3 severe intercept target must be left of the normal target"
            )
        if config.gate3_severe_roll_floor_rad < config.gate3_roll_floor_rad:
            raise ValueError(
                "Gate-3 severe roll floor must not be below the normal floor"
            )
        if config.gate3_severe_roll_floor_rad > 0.5:
            raise ValueError("Gate-3 severe roll floor must not exceed 0.5")
    elif config.gate3_severe_intercept_target_right_m != 0.0:
        raise ValueError("Gate-3 severe intercept target requires severe bank")
    if config.gate3_left_trigger_m >= 0.0:
        raise ValueError("Gate-3 left trigger must be negative")
    if not 0.0 < config.gate3_roll_floor_rad <= 0.5:
        raise ValueError("Gate-3 roll floor must be in (0, 0.5]")
    if not -0.5 <= config.gate3_brake_pitch_rad <= 0.0:
        raise ValueError("Gate-3 brake pitch must be in [-0.5, 0]")
    if config.gate3_brake_min_closing_rate_m_s < 0.0:
        raise ValueError("Gate-3 brake closing-rate threshold must be nonnegative")
    if config.gate3_brake_release_closing_rate_m_s < 0.0:
        raise ValueError("Gate-3 brake release threshold must be nonnegative")
    if (
        config.gate3_brake_release_closing_rate_m_s > 0.0
        and config.gate3_brake_release_closing_rate_m_s
        >= config.gate3_brake_min_closing_rate_m_s
    ):
        raise ValueError("Gate-3 brake release threshold must be below activation")
    if not -0.5 <= config.gate3_search_brake_pitch_rad <= 0.0:
        raise ValueError("Gate-3 search brake pitch must be in [-0.5, 0]")
    if not 0.0 <= config.gate3_centered_approach_pitch_rad <= 0.5:
        raise ValueError("Gate-3 centered approach pitch must be in [0, 0.5]")
    if config.gate3_lateral_kp < 0.0 or config.gate3_lateral_kd < 0.0:
        raise ValueError("Gate-3 lateral gains must be nonnegative")
    if config.gate3_intercept_plane_forward_m < 0.0:
        raise ValueError("Gate-3 intercept plane must be nonnegative")
    if config.gate3_intercept_plane_forward_m >= config.gate3_close_forward_m:
        raise ValueError("Gate-3 intercept plane must be below the close range")
    if not -0.5 <= config.gate3_counter_roll_rad <= 0.0:
        raise ValueError("Gate-3 counter roll must be in [-0.5, 0]")
    if config.gate3_min_closing_rate_m_s <= 0.0:
        raise ValueError("Gate-3 minimum closing rate must be positive")
    if config.gate3_counter_latch_forward_m < 0.0:
        raise ValueError("Gate-3 counter latch range must be nonnegative")
    if config.gate3_counter_latch_forward_m >= config.gate3_close_forward_m:
        raise ValueError("Gate-3 counter latch range must be below close range")
    if not 0.0 <= config.gate3_latched_level_roll_threshold_rad <= 0.5:
        raise ValueError("Gate-3 latched level-roll threshold must be in [0, 0.5]")
    if config.gate3_tilt_compensated_hover_thrust < 0.0:
        raise ValueError("Gate-3 tilt-compensated hover thrust must be nonnegative")
    if (
        config.gate3_tilt_compensated_hover_thrust > 0.0
        and not config.min_thrust
        <= config.gate3_tilt_compensated_hover_thrust
        <= config.max_thrust
    ):
        raise ValueError("Gate-3 hover thrust must be inside final thrust bounds")
    if config.phase_denominator <= 0:
        raise ValueError("PUFFER_POLICY_RACE_PHASE_DENOMINATOR must be positive")
    if config.max_pitch_rad <= 0.0 or config.max_pitch_rad > 0.5:
        raise ValueError("final-gate max pitch must be in (0, 0.5]")
    if not 0.0 < config.aligned_forward_pitch_rad <= config.max_pitch_rad:
        raise ValueError("aligned forward pitch must be in (0, max_pitch]")
    if not config.min_thrust < config.hover_thrust < config.max_thrust:
        raise ValueError("final-gate thrust limits must bracket hover thrust")
    if config.center_before_accel_down_error_m < 0.0:
        raise ValueError("final-gate vertical alignment tolerance must be nonnegative")
    if config.final_gate_max_pose_jump_m < 0.0:
        raise ValueError("final-gate maximum pose jump must be nonnegative")
    if config.final_gate_max_anchor_drift_m < 0.0:
        raise ValueError("final-gate maximum anchor drift must be nonnegative")
    if config.final_gate_association_grace_s < 0.0:
        raise ValueError("final-gate association grace must be nonnegative")
    if config.final_gate_reanchor_persistence_s < 0.0:
        raise ValueError("final-gate re-anchor persistence must be nonnegative")
    if config.final_gate_reanchor_max_forward_m < 0.0:
        raise ValueError("final-gate re-anchor range must be nonnegative")
    if config.final_gate_reanchor_max_count < 1:
        raise ValueError("final-gate re-anchor count must be positive")
    if config.final_gate_close_reanchor_persistence_s < 0.0:
        raise ValueError("close final-gate re-anchor persistence must be nonnegative")
    if config.final_gate_close_reanchor_max_right_m < 0.0:
        raise ValueError("close final-gate re-anchor lateral bound must be nonnegative")
    if config.final_gate_close_reanchor_max_count < 1:
        raise ValueError("close final-gate re-anchor count must be positive")
    crossing_reanchor_values = (
        config.final_gate_crossing_reanchor_persistence_s,
        config.final_gate_crossing_reanchor_max_forward_m,
        config.final_gate_crossing_reanchor_max_right_m,
        config.final_gate_crossing_reanchor_max_abs_down_m,
    )
    if any(value < 0.0 for value in crossing_reanchor_values):
        raise ValueError("crossing final-gate re-anchor bounds must be nonnegative")
    if config.final_gate_crossing_reanchor_max_count < 1:
        raise ValueError("crossing final-gate re-anchor count must be positive")
    if any(value > 0.0 for value in crossing_reanchor_values) and not all(
        value > 0.0 for value in crossing_reanchor_values
    ):
        raise ValueError("crossing final-gate re-anchor requires all bounds")
    if config.final_gate_lateral_roll_kp < 0.0:
        raise ValueError("final-gate lateral roll gain must be nonnegative")
    if not 0.0 <= config.final_gate_max_lateral_roll_rad <= 0.5:
        raise ValueError("final-gate lateral roll limit must be in [0, 0.5]")
    if (
        config.final_gate_lateral_roll_kp > 0.0
        and config.final_gate_max_lateral_roll_rad <= 0.0
    ):
        raise ValueError("final-gate lateral roll gain requires a positive limit")
    if not 0.0 <= config.final_gate_hover_approach_pitch_rad <= config.max_pitch_rad:
        raise ValueError("final-gate hover approach pitch must be in [0, max_pitch]")
    if config.final_gate_hover_approach_max_forward_m < 0.0:
        raise ValueError("final-gate hover approach range must be nonnegative")
    if config.final_gate_descent_delay_s < 0.0:
        raise ValueError("final-gate descent delay must be nonnegative")
    if config.final_gate_descent_max_forward_m < 0.0:
        raise ValueError("final-gate descent range must be nonnegative")
    if config.final_gate_descent_later_max_forward_m < 0.0:
        raise ValueError("later final-gate descent range must be nonnegative")
    if config.final_gate_descent_max_abs_right_m < 0.0:
        raise ValueError("final-gate descent lateral bound must be nonnegative")
    if config.final_gate_descent_pulse_s < 0.0:
        raise ValueError("final-gate descent pulse must be nonnegative")
    if config.final_gate_descent_later_pulse_s < 0.0:
        raise ValueError("later final-gate descent pulse must be nonnegative")
    if config.final_gate_descent_pulse_max_count < 1:
        raise ValueError("final-gate descent pulse count must be positive")
    if config.final_gate_descent_pulse_cooldown_s < 0.0:
        raise ValueError("final-gate descent pulse cooldown must be nonnegative")
    if not config.min_thrust <= config.final_gate_descent_thrust <= config.hover_thrust:
        raise ValueError("final-gate descent thrust must be in [min, hover]")
    if config.final_gate_descent_pulse_s > 0.0:
        if config.final_gate_descent_delay_s <= 0.0:
            raise ValueError("final-gate descent pulse requires a positive delay")
        if config.final_gate_descent_max_forward_m <= 0.0:
            raise ValueError("final-gate descent pulse requires a positive range")
        if config.final_gate_descent_thrust >= config.hover_thrust:
            raise ValueError("final-gate descent pulse thrust must be below hover")
        if (
            config.final_gate_descent_pulse_max_count > 1
            and config.final_gate_descent_pulse_cooldown_s <= 0.0
        ):
            raise ValueError("multiple descent pulses require a positive cooldown")
        if (
            config.final_gate_descent_later_pulse_s > 0.0
            and config.final_gate_descent_later_pulse_s
            > config.final_gate_descent_pulse_s
        ):
            raise ValueError("later descent pulse must not exceed the first pulse")
        if (
            config.final_gate_descent_later_max_forward_m > 0.0
            and config.final_gate_descent_later_max_forward_m
            > config.final_gate_descent_max_forward_m
        ):
            raise ValueError("later descent range must not exceed the first range")
    if config.final_gate_reanchor_persistence_s > 0.0:
        if config.final_gate_max_pose_jump_m <= 0.0:
            raise ValueError("final-gate re-anchor requires a positive step bound")
        if config.final_gate_max_anchor_drift_m <= 0.0:
            raise ValueError("final-gate re-anchor requires a positive anchor bound")
        if (
            config.final_gate_reanchor_max_forward_m <= 0.0
            and config.final_gate_descent_max_forward_m <= 0.0
        ):
            raise ValueError("final-gate re-anchor requires a positive range")
    if config.final_gate_close_reanchor_persistence_s > 0.0:
        if config.final_gate_max_pose_jump_m <= 0.0:
            raise ValueError("close final-gate re-anchor requires a positive step bound")
        if config.final_gate_descent_delay_s <= 0.0:
            raise ValueError("close final-gate re-anchor requires a positive delay")
        if config.final_gate_descent_max_forward_m <= 0.0:
            raise ValueError("close final-gate re-anchor requires a positive range")
        if config.final_gate_close_reanchor_max_right_m <= 0.0:
            raise ValueError("close final-gate re-anchor requires a lateral bound")
    if config.final_gate_crossing_reanchor_persistence_s > 0.0:
        if config.final_gate_max_pose_jump_m <= 0.0:
            raise ValueError("crossing final-gate re-anchor requires a step bound")
        if config.final_gate_descent_delay_s <= 0.0:
            raise ValueError("crossing final-gate re-anchor requires a positive delay")
    if config.final_gate_hover_approach_pitch_rad > 0.0:
        if config.final_gate_descent_delay_s <= 0.0:
            raise ValueError("final-gate hover approach requires a positive delay")
        if (
            config.final_gate_hover_approach_max_forward_m
            <= config.final_gate_descent_max_forward_m
        ):
            raise ValueError("final-gate hover approach range must exceed descent range")
    if not 0.0 <= config.final_gate_recovery_pitch_rad <= config.max_pitch_rad:
        raise ValueError("final-gate recovery pitch must be in [0, max_pitch]")
    return config


def _resolve_model() -> CheckpointPolicy:
    global _MODEL, _MODEL_KEY
    checkpoint = os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "").strip()
    if not checkpoint:
        raise RuntimeError("PUFFER_POLICY_CHECKPOINT_PATH is required")
    key = (
        os.path.abspath(checkpoint),
        _env_int("PUFFER_POLICY_INPUT_DIM", 23),
        _env_int("PUFFER_POLICY_HIDDEN_DIM", 128),
        _env_int("PUFFER_POLICY_NUM_LAYERS", 3),
        _env_int("PUFFER_POLICY_NUM_ACTIONS", 4),
    )
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = CheckpointPolicy.load(
            key[0],
            input_dim=key[1],
            hidden_dim=key[2],
            num_layers=key[3],
            num_actions=key[4],
        )
        _MODEL_KEY = key
    return _MODEL


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _inverse_tanh_norm(value: float, scale: float) -> float:
    return math.atanh(_clamp(float(value), -0.999, 0.999)) * scale


def _yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def _pitch_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    return math.asin(_clamp(2.0 * (w * y - z * x), -1.0, 1.0))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _encode_thrust(thrust: float, config: FinalGateHandoffConfig) -> float:
    span = (
        config.max_thrust - config.hover_thrust
        if thrust >= config.hover_thrust
        else config.hover_thrust - config.min_thrust
    )
    return _clamp((thrust - config.hover_thrust) / span, -1.0, 1.0)


def final_gate_vector_from_observation(
    observation: Sequence[float],
) -> tuple[float, float, float] | None:
    if float(observation[10]) < 0.5:
        return None
    return (
        _inverse_tanh_norm(observation[11], 10.0),
        _inverse_tanh_norm(observation[12], 5.0),
        _inverse_tanh_norm(observation[13], 5.0),
    )


def final_gate_pose_is_associated(
    gate_vector: Sequence[float] | None,
    previous_vector: Sequence[float] | None,
    max_jump_m: float,
    *,
    anchor_vector: Sequence[float] | None = None,
    max_anchor_drift_m: float = 0.0,
    allow_reseed: bool = False,
) -> bool:
    if allow_reseed:
        return True
    if gate_vector is None:
        return True
    if previous_vector is not None and max_jump_m > 0.0:
        jump_m = math.sqrt(
            sum(
                (float(gate_vector[index]) - float(previous_vector[index])) ** 2
                for index in range(3)
            )
        )
        if jump_m > max_jump_m:
            return False
    if anchor_vector is not None and max_anchor_drift_m > 0.0:
        anchor_drift_m = math.sqrt(
            sum(
                (float(gate_vector[index]) - float(anchor_vector[index])) ** 2
                for index in range(3)
            )
        )
        if anchor_drift_m > max_anchor_drift_m:
            return False
    return True


def final_gate_rejected_cluster_state(
    gate_vector: Sequence[float] | None,
    now_s: float,
    cluster_started_s: float | None,
    cluster_vector: Sequence[float] | None,
    config: FinalGateHandoffConfig,
    *,
    persistence_s: float | None = None,
    max_forward_m: float | None = None,
) -> tuple[
    bool,
    float | None,
    tuple[float, float, float] | None,
]:
    """Admit a close rejected pose only after one coherent persistence window."""

    required_persistence_s = (
        config.final_gate_reanchor_persistence_s
        if persistence_s is None
        else float(persistence_s)
    )
    if required_persistence_s <= 0.0 or gate_vector is None:
        return False, None, None
    current = tuple(float(value) for value in gate_vector)
    if cluster_vector is None or cluster_started_s is None:
        return False, now_s, current
    cluster_step_m = math.sqrt(
        sum(
            (current[index] - float(cluster_vector[index])) ** 2
            for index in range(3)
        )
    )
    if cluster_step_m > config.final_gate_max_pose_jump_m:
        return False, now_s, current
    reanchor = (
        now_s - cluster_started_s >= required_persistence_s
        and 0.0
        < current[0]
        <= (
            float(max_forward_m)
            if max_forward_m is not None
            else (
                config.final_gate_reanchor_max_forward_m
                if config.final_gate_reanchor_max_forward_m > 0.0
                else config.final_gate_descent_max_forward_m
            )
        )
    )
    return reanchor, cluster_started_s, current


def final_gate_close_reanchor_candidate(
    gate_vector: Sequence[float] | None,
    phase_elapsed_s: float,
    config: FinalGateHandoffConfig,
) -> bool:
    """Identify a brief centered close-gate cluster eligible for one fast re-anchor."""

    return bool(
        config.final_gate_close_reanchor_persistence_s > 0.0
        and phase_elapsed_s >= config.final_gate_descent_delay_s
        and gate_vector is not None
        and 0.0
        < float(gate_vector[0])
        <= config.final_gate_descent_max_forward_m
        and abs(float(gate_vector[1]))
        <= config.final_gate_close_reanchor_max_right_m
        and float(gate_vector[2]) > config.center_before_accel_down_error_m
    )


def final_gate_crossing_reanchor_candidate(
    gate_vector: Sequence[float] | None,
    phase_elapsed_s: float,
    config: FinalGateHandoffConfig,
) -> bool:
    """Admit only a tightly centered, vertically aligned final crossing pose."""

    return bool(
        config.final_gate_crossing_reanchor_persistence_s > 0.0
        and phase_elapsed_s >= config.final_gate_descent_delay_s
        and gate_vector is not None
        and 0.0
        < float(gate_vector[0])
        <= config.final_gate_crossing_reanchor_max_forward_m
        and abs(float(gate_vector[1]))
        <= config.final_gate_crossing_reanchor_max_right_m
        and abs(float(gate_vector[2]))
        <= config.final_gate_crossing_reanchor_max_abs_down_m
    )


def gate_index_from_observation(
    observation: Sequence[float], config: FinalGateHandoffConfig
) -> int:
    return int(round(_clamp(float(observation[22]), 0.0, 1.0) * config.phase_denominator))


def gate3_close_correction_action(
    observation: Sequence[float],
    base_action: Sequence[float],
    config: FinalGateHandoffConfig,
    *,
    force_counter_bank: bool = False,
    force_speed_brake: bool = False,
) -> list[float]:
    """Keep leftward bank authority through a measured close Gate-3 miss."""

    action = [_clamp(float(value), -1.0, 1.0) for value in base_action]
    if float(observation[10]) < 0.5:
        if config.gate3_search_brake_pitch_rad < 0.0:
            action[0] = min(
                action[0], config.gate3_search_brake_pitch_rad / 0.5
            )
        if config.gate3_level_roll_during_search:
            action[1] = 0.0
        return action
    gate_forward_m = _inverse_tanh_norm(observation[11], 10.0)
    gate_right_m = _inverse_tanh_norm(observation[12], 5.0)
    in_close_window = 0.0 < gate_forward_m <= config.gate3_close_forward_m
    gate_forward_rate_m_s = _inverse_tanh_norm(observation[0], 5.0)
    gate_right_rate_m_s = _inverse_tanh_norm(observation[1], 3.0)
    closing_rate_m_s = -gate_forward_rate_m_s
    predicted_right_m = gate_right_m
    if (
        config.gate3_intercept_plane_forward_m > 0.0
        and closing_rate_m_s >= config.gate3_min_closing_rate_m_s
    ):
        predicted_right_m += gate_right_rate_m_s * max(
            gate_forward_m - config.gate3_intercept_plane_forward_m,
            0.0,
        ) / closing_rate_m_s
    adaptive_bank_active = (
        config.gate3_severe_bank_forward_m > 0.0
        and 0.0 < gate_forward_m <= config.gate3_severe_bank_forward_m
        and closing_rate_m_s >= config.gate3_severe_bank_min_closing_rate_m_s
        and gate_right_m < config.gate3_left_trigger_m
        and predicted_right_m < config.gate3_intercept_target_right_m
    )
    early_bank_active = (
        config.gate3_early_bank_forward_m > 0.0
        and config.gate3_close_forward_m
        < gate_forward_m
        <= config.gate3_early_bank_forward_m
        and closing_rate_m_s >= config.gate3_early_bank_min_closing_rate_m_s
        and gate_right_m < config.gate3_left_trigger_m
    )
    brake_active = config.gate3_brake_pitch_rad < 0.0 and (
        force_speed_brake
        or config.gate3_brake_min_closing_rate_m_s <= 0.0
        or closing_rate_m_s >= config.gate3_brake_min_closing_rate_m_s
    )
    if not in_close_window and config.gate3_search_brake_pitch_rad < 0.0:
        action[0] = min(action[0], config.gate3_search_brake_pitch_rad / 0.5)
    if not in_close_window and config.gate3_level_roll_during_search:
        action[1] = 0.0
    if adaptive_bank_active:
        severity = _clamp(
            (
                config.gate3_intercept_target_right_m - predicted_right_m
            )
            / (
                config.gate3_intercept_target_right_m
                - config.gate3_severe_intercept_target_right_m
            ),
            0.0,
            1.0,
        )
        roll_floor_rad = config.gate3_roll_floor_rad + severity * (
            config.gate3_severe_roll_floor_rad - config.gate3_roll_floor_rad
        )
        action[1] = max(action[1], roll_floor_rad / 0.5)
        return action
    if early_bank_active:
        action[1] = max(action[1], config.gate3_roll_floor_rad / 0.5)
        return action
    if in_close_window and config.gate3_tilt_compensated_hover_thrust > 0.0:
        qw, qx, qy, qz = (float(value) for value in observation[6:10])
        current_roll_rad = _roll_from_quaternion(qw, qx, qy, qz)
        current_pitch_rad = _pitch_from_quaternion(qw, qx, qy, qz)
        vertical_scale = max(
            math.cos(current_roll_rad) * math.cos(current_pitch_rad), 0.5
        )
        compensated_thrust = _clamp(
            config.gate3_tilt_compensated_hover_thrust / vertical_scale,
            config.min_thrust,
            config.max_thrust,
        )
        action[2] = _encode_thrust(compensated_thrust, config)
    if in_close_window and config.gate3_intercept_plane_forward_m > 0.0:
        if brake_active:
            action[0] = min(action[0], config.gate3_brake_pitch_rad / 0.5)
        latch_mode = config.gate3_counter_latch_forward_m > 0.0
        correction_needed = gate_right_m < config.gate3_left_trigger_m and (
            latch_mode or predicted_right_m < config.gate3_intercept_target_right_m
        )
        if force_counter_bank and latch_mode:
            if config.gate3_centered_approach_pitch_rad > 0.0:
                action[0] = config.gate3_centered_approach_pitch_rad / 0.5
            current_roll_rad = _roll_from_quaternion(
                *(float(value) for value in observation[6:10])
            )
            action[1] = (
                config.gate3_counter_roll_rad / 0.5
                if current_roll_rad
                > config.gate3_latched_level_roll_threshold_rad
                else 0.0
            )
        elif correction_needed:
            action[1] = max(action[1], config.gate3_roll_floor_rad / 0.5)
        else:
            action[1] = config.gate3_counter_roll_rad / 0.5
    elif in_close_window and config.gate3_lateral_kp > 0.0:
        gate_right_rate_m_s = _inverse_tanh_norm(observation[1], 3.0)
        if brake_active:
            action[0] = min(action[0], config.gate3_brake_pitch_rad / 0.5)
        desired_roll_rad = _clamp(
            -config.gate3_lateral_kp * gate_right_m
            - config.gate3_lateral_kd * gate_right_rate_m_s,
            -config.gate3_roll_floor_rad,
            config.gate3_roll_floor_rad,
        )
        action[1] = desired_roll_rad / 0.5
    elif in_close_window and gate_right_m < config.gate3_left_trigger_m:
        if brake_active:
            action[0] = min(action[0], config.gate3_brake_pitch_rad / 0.5)
        action[1] = max(action[1], config.gate3_roll_floor_rad / 0.5)
    elif in_close_window and config.gate3_centered_approach_pitch_rad > 0.0:
        action[0] = config.gate3_centered_approach_pitch_rad / 0.5
        action[1] = 0.0
    return action


def gate3_counter_latch_ready(
    observation: Sequence[float], config: FinalGateHandoffConfig
) -> bool:
    if (
        config.gate3_counter_latch_forward_m <= 0.0
        or config.gate3_intercept_plane_forward_m <= 0.0
        or float(observation[10]) < 0.5
    ):
        return False
    gate_forward_m = _inverse_tanh_norm(observation[11], 10.0)
    if not 0.0 < gate_forward_m <= config.gate3_counter_latch_forward_m:
        return False
    gate_right_m = _inverse_tanh_norm(observation[12], 5.0)
    gate_forward_rate_m_s = _inverse_tanh_norm(observation[0], 5.0)
    gate_right_rate_m_s = _inverse_tanh_norm(observation[1], 3.0)
    closing_rate_m_s = -gate_forward_rate_m_s
    predicted_right_m = gate_right_m
    if closing_rate_m_s >= config.gate3_min_closing_rate_m_s:
        predicted_right_m += gate_right_rate_m_s * max(
            gate_forward_m - config.gate3_intercept_plane_forward_m,
            0.0,
        ) / closing_rate_m_s
    return not (
        gate_right_m < config.gate3_left_trigger_m
        and predicted_right_m < config.gate3_intercept_target_right_m
    )


def gate3_speed_brake_latched(
    observation: Sequence[float],
    config: FinalGateHandoffConfig,
    currently_latched: bool,
) -> bool:
    if (
        config.gate3_brake_pitch_rad >= 0.0
        or config.gate3_brake_min_closing_rate_m_s <= 0.0
        or config.gate3_brake_release_closing_rate_m_s <= 0.0
        or float(observation[10]) < 0.5
    ):
        return False
    gate_forward_m = _inverse_tanh_norm(observation[11], 10.0)
    if not 0.0 < gate_forward_m <= config.gate3_close_forward_m:
        return False
    closing_rate_m_s = -_inverse_tanh_norm(observation[0], 5.0)
    if currently_latched:
        return closing_rate_m_s > config.gate3_brake_release_closing_rate_m_s
    return closing_rate_m_s >= config.gate3_brake_min_closing_rate_m_s


def final_gate_action(
    observation: Sequence[float],
    base_action: Sequence[float],
    config: FinalGateHandoffConfig,
    *,
    fallback_action: Sequence[float] | None = None,
) -> list[float]:
    """Return the observable final-gate command or a bounded dropout fallback."""

    if len(observation) != 23 or len(base_action) != 4:
        raise ValueError("expected 23 observation values and four base actions")
    gate_visible = float(observation[10]) >= 0.5
    if not gate_visible:
        if fallback_action is not None:
            result = [_clamp(float(value), -1.0, 1.0) for value in fallback_action]
            result[1] = 0.0
            return result
        # A level, hover-centered command is safer than advancing an untrained
        # final-gate recurrent head through an unobserved target.
        return [0.0, 0.0, 0.0, float(base_action[3])]

    gate_right_m = _inverse_tanh_norm(observation[12], 5.0)
    gate_down_m = _inverse_tanh_norm(observation[13], 5.0)
    yaw_error_rad = _clamp(float(observation[14]), -1.0, 1.0) * (math.pi / 4.0)
    qw, qx, qy, qz = (float(value) for value in observation[6:10])
    current_yaw_rad = _yaw_from_quaternion(qw, qx, qy, qz)

    aligned = (
        abs(yaw_error_rad) <= config.center_before_accel_yaw_error_rad
        and abs(gate_down_m) <= config.center_before_accel_down_error_m
    )
    if not aligned:
        # Shed the inherited prefix speed while rotating. The v3385 measured
        # attitude plant uses negative pitch as braking and positive pitch as
        # forward acceleration for this policy-action convention.
        pitch_rad = -config.max_pitch_rad
    else:
        pitch_rad = config.aligned_forward_pitch_rad

    # v3385's commanded yaw sign is opposite the camera-bearing convention:
    # positive image-right error requires a negative absolute-yaw correction.
    desired_yaw_rad = _wrap_angle(current_yaw_rad - yaw_error_rad)
    thrust = _clamp(
        config.hover_thrust - config.vertical_thrust_kp * gate_down_m,
        config.min_thrust,
        config.max_thrust,
    )
    roll_rad = _clamp(
        -config.final_gate_lateral_roll_kp * gate_right_m,
        -config.final_gate_max_lateral_roll_rad,
        config.final_gate_max_lateral_roll_rad,
    )
    return [
        _clamp(pitch_rad / 0.5, -1.0, 1.0),
        _clamp(roll_rad / 0.5, -1.0, 1.0),
        _encode_thrust(thrust, config),
        _clamp(desired_yaw_rad / math.pi, -1.0, 1.0),
    ]


def final_gate_dropout_yaw_action(
    action: Sequence[float],
    observation: Sequence[float],
    current_associated_gate_vector: Sequence[float] | None,
    config: FinalGateHandoffConfig,
) -> list[float]:
    """Arrest a stale absolute-yaw sweep while final association is absent."""

    if len(action) != 4 or len(observation) != 23:
        raise ValueError("expected four actions and 23 observation values")
    result = [_clamp(float(value), -1.0, 1.0) for value in action]
    if (
        not config.final_gate_freeze_yaw_on_dropout
        or current_associated_gate_vector is not None
    ):
        return result
    current_yaw_rad = _yaw_from_quaternion(
        *(float(value) for value in observation[6:10])
    )
    result[3] = _clamp(current_yaw_rad / math.pi, -1.0, 1.0)
    return result


def delayed_final_gate_recovery_action(
    action: Sequence[float],
    associated_gate_vector: Sequence[float] | None,
    phase_elapsed_s: float,
    config: FinalGateHandoffConfig,
) -> list[float]:
    """Descend and optionally advance only after the wall-clearance delay."""

    result = [_clamp(float(value), -1.0, 1.0) for value in action]
    if config.final_gate_descent_max_forward_m > 0.0 and (
        associated_gate_vector is None
        or not 0.0
        < float(associated_gate_vector[0])
        <= config.final_gate_descent_max_forward_m
    ):
        # A stale last action may already contain descent thrust. Range gating
        # is fail-safe: a dropout, association rejection, or far pose restores
        # hover until a current accepted active-gate pose authorizes descent.
        result[2] = _encode_thrust(config.hover_thrust, config)
        return result
    if (
        config.final_gate_descent_delay_s <= 0.0
        or phase_elapsed_s < config.final_gate_descent_delay_s
        or associated_gate_vector is None
        or (
            config.final_gate_descent_max_forward_m > 0.0
            and not 0.0
            < float(associated_gate_vector[0])
            <= config.final_gate_descent_max_forward_m
        )
        or float(associated_gate_vector[2])
        <= config.center_before_accel_down_error_m
    ):
        return result
    if config.final_gate_recovery_pitch_rad > 0.0:
        result[0] = config.final_gate_recovery_pitch_rad / 0.5
    result[2] = _encode_thrust(config.final_gate_descent_thrust, config)
    return result


def final_gate_hover_approach_action(
    action: Sequence[float],
    associated_gate_vector: Sequence[float] | None,
    phase_elapsed_s: float,
    config: FinalGateHandoffConfig,
) -> list[float]:
    """Approach a trusted low gate at hover altitude before close descent."""

    result = [_clamp(float(value), -1.0, 1.0) for value in action]
    if config.final_gate_hover_approach_pitch_rad <= 0.0:
        return result
    if associated_gate_vector is None:
        # Never hold a positive approach command through a rejected pose or
        # detector dropout. Brake and hover until a current pose is accepted.
        result[0] = -config.max_pitch_rad / 0.5
        result[2] = _encode_thrust(config.hover_thrust, config)
        return result
    forward_m = float(associated_gate_vector[0])
    gate_down_m = float(associated_gate_vector[2])
    if (
        phase_elapsed_s >= config.final_gate_descent_delay_s
        and config.final_gate_descent_max_forward_m
        < forward_m
        <= config.final_gate_hover_approach_max_forward_m
        and gate_down_m > config.center_before_accel_down_error_m
    ):
        result[0] = config.final_gate_hover_approach_pitch_rad / 0.5
        result[2] = _encode_thrust(config.hover_thrust, config)
    return result


def final_gate_descent_pulse_vector(
    current_gate_vector: Sequence[float] | None,
    phase_elapsed_s: float,
    now_s: float,
    pulse_started_s: float | None,
    pulse_vector: Sequence[float] | None,
    pulse_count: int,
    pulse_last_end_s: float | None,
    config: FinalGateHandoffConfig,
) -> tuple[
    Sequence[float] | None,
    float | None,
    Sequence[float] | None,
    int,
    float | None,
]:
    """Bridge one brief association dropout with a bounded descent pulse."""

    if config.final_gate_descent_pulse_s <= 0.0:
        return (
            current_gate_vector,
            pulse_started_s,
            pulse_vector,
            pulse_count,
            pulse_last_end_s,
        )
    if (
        pulse_started_s is not None
        and current_gate_vector is not None
        and float(current_gate_vector[2])
        <= config.center_before_accel_down_error_m
    ):
        pulse_last_end_s = now_s
        pulse_started_s = None
        pulse_vector = None
    active_pulse_s = (
        config.final_gate_descent_later_pulse_s
        if pulse_count > 1 and config.final_gate_descent_later_pulse_s > 0.0
        else config.final_gate_descent_pulse_s
    )
    if (
        pulse_started_s is not None
        and now_s - pulse_started_s >= active_pulse_s
    ):
        pulse_last_end_s = pulse_started_s + active_pulse_s
        pulse_started_s = None
        pulse_vector = None
    ready = (
        pulse_started_s is None
        and pulse_count < config.final_gate_descent_pulse_max_count
        and (
            pulse_last_end_s is None
            or now_s - pulse_last_end_s
            >= config.final_gate_descent_pulse_cooldown_s
        )
        and current_gate_vector is not None
        and phase_elapsed_s >= config.final_gate_descent_delay_s
        and 0.0
        < float(current_gate_vector[0])
        <= (
            config.final_gate_descent_later_max_forward_m
            if pulse_count >= 1
            and config.final_gate_descent_later_max_forward_m > 0.0
            else config.final_gate_descent_max_forward_m
        )
        and (
            config.final_gate_descent_max_abs_right_m <= 0.0
            or abs(float(current_gate_vector[1]))
            <= config.final_gate_descent_max_abs_right_m
        )
        and float(current_gate_vector[2])
        > config.center_before_accel_down_error_m
    )
    if ready:
        pulse_started_s = now_s
        pulse_vector = tuple(float(value) for value in current_gate_vector)
        pulse_count += 1
    if (
        pulse_started_s is not None
        and pulse_vector is not None
        and now_s - pulse_started_s < active_pulse_s
    ):
        return (
            pulse_vector,
            pulse_started_s,
            pulse_vector,
            pulse_count,
            pulse_last_end_s,
        )
    return None, pulse_started_s, pulse_vector, pulse_count, pulse_last_end_s


def final_gate_pulse_lateral_hold_action(
    action: Sequence[float],
    pulse_gate_vector: Sequence[float] | None,
    current_associated_gate_vector: Sequence[float] | None,
    pulse_started_s: float | None,
    config: FinalGateHandoffConfig,
) -> list[float]:
    """Keep bounded saved lateral correction only during an active pulse dropout."""

    result = [_clamp(float(value), -1.0, 1.0) for value in action]
    if (
        not config.final_gate_hold_pulse_lateral
        or pulse_started_s is None
        or pulse_gate_vector is None
        or current_associated_gate_vector is not None
    ):
        return result
    roll_rad = _clamp(
        -config.final_gate_lateral_roll_kp * float(pulse_gate_vector[1]),
        -config.final_gate_max_lateral_roll_rad,
        config.final_gate_max_lateral_roll_rad,
    )
    result[1] = _clamp(roll_rad / 0.5, -1.0, 1.0)
    return result


def final_gate_pulse_pitch_action(
    action: Sequence[float],
    pulse_started_s: float | None,
    config: FinalGateHandoffConfig,
) -> list[float]:
    """Level pitch only during an already-authorized bounded descent pulse."""

    result = [_clamp(float(value), -1.0, 1.0) for value in action]
    if (
        config.final_gate_level_pitch_during_descent_pulse
        and pulse_started_s is not None
    ):
        result[0] = 0.0
    return result


def _clear_post_prefix_gate_state() -> None:
    """Clear target-specific state without resetting the recurrent prefix."""

    global _LAST_FINAL_ACTION, _LAST_FINAL_GATE_VECTOR, _FINAL_GATE_ANCHOR_VECTOR
    global _GATE3_COUNTERBANK_LATCHED, _GATE3_SPEED_BRAKE_LATCHED
    global _FINAL_GATE_PHASE_STARTED_S
    global _FINAL_GATE_DESCENT_PULSE_STARTED_S, _FINAL_GATE_DESCENT_PULSE_VECTOR
    global _FINAL_GATE_DESCENT_PULSE_COUNT, _FINAL_GATE_DESCENT_PULSE_LAST_END_S
    global _FINAL_GATE_REJECTED_CLUSTER_STARTED_S, _FINAL_GATE_REJECTED_CLUSTER_VECTOR
    global _FINAL_GATE_REANCHOR_COUNT, _FINAL_GATE_CLOSE_REANCHOR_COUNT
    global _FINAL_GATE_CROSSING_REANCHOR_COUNT
    _LAST_FINAL_ACTION = None
    _LAST_FINAL_GATE_VECTOR = None
    _FINAL_GATE_ANCHOR_VECTOR = None
    _GATE3_COUNTERBANK_LATCHED = False
    _GATE3_SPEED_BRAKE_LATCHED = False
    _FINAL_GATE_PHASE_STARTED_S = None
    _FINAL_GATE_DESCENT_PULSE_STARTED_S = None
    _FINAL_GATE_DESCENT_PULSE_VECTOR = None
    _FINAL_GATE_DESCENT_PULSE_COUNT = 0
    _FINAL_GATE_DESCENT_PULSE_LAST_END_S = None
    _FINAL_GATE_REJECTED_CLUSTER_STARTED_S = None
    _FINAL_GATE_REJECTED_CLUSTER_VECTOR = None
    _FINAL_GATE_REANCHOR_COUNT = 0
    _FINAL_GATE_CLOSE_REANCHOR_COUNT = 0
    _FINAL_GATE_CROSSING_REANCHOR_COUNT = 0


def _start_post_prefix_gate(gate_index: int) -> bool:
    """Start a fresh target epoch when the authoritative gate index advances."""

    global _POST_PREFIX_GATE_INDEX
    normalized_index = int(gate_index)
    if _POST_PREFIX_GATE_INDEX == normalized_index:
        return False
    _clear_post_prefix_gate_state()
    _POST_PREFIX_GATE_INDEX = normalized_index
    return True


def reset() -> None:
    global _LAST_YAW_ACTION, _LAST_FINAL_ACTION, _LAST_FINAL_GATE_VECTOR
    global _FINAL_GATE_ANCHOR_VECTOR
    global _GATE3_COUNTERBANK_LATCHED
    global _GATE3_SPEED_BRAKE_LATCHED
    global _FINAL_GATE_PHASE_STARTED_S
    global _FINAL_GATE_DESCENT_PULSE_STARTED_S, _FINAL_GATE_DESCENT_PULSE_VECTOR
    global _FINAL_GATE_DESCENT_PULSE_COUNT, _FINAL_GATE_DESCENT_PULSE_LAST_END_S
    global _FINAL_GATE_REJECTED_CLUSTER_STARTED_S, _FINAL_GATE_REJECTED_CLUSTER_VECTOR
    global _FINAL_GATE_REANCHOR_COUNT, _FINAL_GATE_CLOSE_REANCHOR_COUNT
    global _FINAL_GATE_CROSSING_REANCHOR_COUNT
    global _POST_PREFIX_GATE_INDEX
    model = _resolve_model()
    model.reset_state()
    _LAST_YAW_ACTION = 0.0
    _clear_post_prefix_gate_state()
    _POST_PREFIX_GATE_INDEX = None


def infer(observation: Sequence[float]) -> list[float]:
    global _LAST_YAW_ACTION, _LAST_FINAL_ACTION, _LAST_FINAL_GATE_VECTOR
    global _FINAL_GATE_ANCHOR_VECTOR
    global _GATE3_COUNTERBANK_LATCHED
    global _GATE3_SPEED_BRAKE_LATCHED
    global _FINAL_GATE_PHASE_STARTED_S
    global _FINAL_GATE_DESCENT_PULSE_STARTED_S, _FINAL_GATE_DESCENT_PULSE_VECTOR
    global _FINAL_GATE_DESCENT_PULSE_COUNT, _FINAL_GATE_DESCENT_PULSE_LAST_END_S
    global _FINAL_GATE_REJECTED_CLUSTER_STARTED_S, _FINAL_GATE_REJECTED_CLUSTER_VECTOR
    global _FINAL_GATE_REANCHOR_COUNT, _FINAL_GATE_CLOSE_REANCHOR_COUNT
    global _FINAL_GATE_CROSSING_REANCHOR_COUNT
    global _POST_PREFIX_GATE_INDEX
    model = _resolve_model()
    config = load_config()
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (model.input_dim,) or model.input_dim != 23:
        raise ValueError(f"expected 23-float phase observation, got {values.shape}")

    gate_index = gate_index_from_observation(values, config)
    if gate_index >= config.handoff_gate_index:
        _start_post_prefix_gate(gate_index)
    else:
        _POST_PREFIX_GATE_INDEX = None
    recurrent_observation = values.copy()
    recurrent_observation[22] = np.float32(_LAST_YAW_ACTION)
    base_action = model.infer(recurrent_observation)
    if gate_index == config.gate3_correction_index:
        _FINAL_GATE_PHASE_STARTED_S = None
        _FINAL_GATE_DESCENT_PULSE_STARTED_S = None
        _FINAL_GATE_DESCENT_PULSE_VECTOR = None
        _FINAL_GATE_DESCENT_PULSE_COUNT = 0
        _FINAL_GATE_DESCENT_PULSE_LAST_END_S = None
        _FINAL_GATE_REJECTED_CLUSTER_STARTED_S = None
        _FINAL_GATE_REJECTED_CLUSTER_VECTOR = None
        _FINAL_GATE_REANCHOR_COUNT = 0
        _FINAL_GATE_CLOSE_REANCHOR_COUNT = 0
        _FINAL_GATE_CROSSING_REANCHOR_COUNT = 0
        _GATE3_SPEED_BRAKE_LATCHED = gate3_speed_brake_latched(
            values,
            config,
            _GATE3_SPEED_BRAKE_LATCHED,
        )
        if gate3_counter_latch_ready(values, config):
            _GATE3_COUNTERBANK_LATCHED = True
        action = gate3_close_correction_action(
            values,
            base_action,
            config,
            force_counter_bank=_GATE3_COUNTERBANK_LATCHED,
            force_speed_brake=_GATE3_SPEED_BRAKE_LATCHED,
        )
        _LAST_FINAL_ACTION = None
        _LAST_FINAL_GATE_VECTOR = None
        _FINAL_GATE_ANCHOR_VECTOR = None
    elif gate_index >= config.handoff_gate_index:
        now_s = time.monotonic()
        if _FINAL_GATE_PHASE_STARTED_S is None:
            _FINAL_GATE_PHASE_STARTED_S = now_s
        phase_elapsed_s = now_s - _FINAL_GATE_PHASE_STARTED_S
        _GATE3_COUNTERBANK_LATCHED = False
        _GATE3_SPEED_BRAKE_LATCHED = False
        gate_vector = final_gate_vector_from_observation(values)
        allow_reseed = (
            now_s - _FINAL_GATE_PHASE_STARTED_S
            <= config.final_gate_association_grace_s
        )
        associated = final_gate_pose_is_associated(
            gate_vector,
            _LAST_FINAL_GATE_VECTOR,
            config.final_gate_max_pose_jump_m,
            anchor_vector=_FINAL_GATE_ANCHOR_VECTOR,
            max_anchor_drift_m=config.final_gate_max_anchor_drift_m,
            allow_reseed=allow_reseed,
        )
        reanchored = False
        if associated:
            _FINAL_GATE_REJECTED_CLUSTER_STARTED_S = None
            _FINAL_GATE_REJECTED_CLUSTER_VECTOR = None
        else:
            crossing_reanchor = final_gate_crossing_reanchor_candidate(
                gate_vector,
                phase_elapsed_s,
                config,
            ) and (
                _FINAL_GATE_CROSSING_REANCHOR_COUNT
                < config.final_gate_crossing_reanchor_max_count
            )
            close_reanchor = final_gate_close_reanchor_candidate(
                gate_vector,
                phase_elapsed_s,
                config,
            ) and (
                _FINAL_GATE_CLOSE_REANCHOR_COUNT
                < config.final_gate_close_reanchor_max_count
            )
            regular_reanchor = (
                _FINAL_GATE_REANCHOR_COUNT < config.final_gate_reanchor_max_count
            )
        if not associated and (
            crossing_reanchor or close_reanchor or regular_reanchor
        ):
            (
                reanchored,
                _FINAL_GATE_REJECTED_CLUSTER_STARTED_S,
                _FINAL_GATE_REJECTED_CLUSTER_VECTOR,
            ) = final_gate_rejected_cluster_state(
                gate_vector,
                now_s,
                _FINAL_GATE_REJECTED_CLUSTER_STARTED_S,
                _FINAL_GATE_REJECTED_CLUSTER_VECTOR,
                config,
                persistence_s=(
                    config.final_gate_crossing_reanchor_persistence_s
                    if crossing_reanchor
                    else (
                        config.final_gate_close_reanchor_persistence_s
                        if close_reanchor
                        else config.final_gate_reanchor_persistence_s
                    )
                ),
                max_forward_m=(
                    config.final_gate_crossing_reanchor_max_forward_m
                    if crossing_reanchor
                    else (
                        config.final_gate_descent_max_forward_m
                        if close_reanchor
                        else None
                    )
                ),
            )
            if reanchored:
                associated = True
                if crossing_reanchor:
                    _FINAL_GATE_CROSSING_REANCHOR_COUNT += 1
                elif close_reanchor:
                    _FINAL_GATE_CLOSE_REANCHOR_COUNT += 1
                else:
                    _FINAL_GATE_REANCHOR_COUNT += 1
                _FINAL_GATE_REJECTED_CLUSTER_STARTED_S = None
                _FINAL_GATE_REJECTED_CLUSTER_VECTOR = None
        if not associated and _LAST_FINAL_ACTION is not None:
            action = list(_LAST_FINAL_ACTION)
            action[1] = 0.0
        else:
            action = final_gate_action(
                values,
                base_action,
                config,
                fallback_action=_LAST_FINAL_ACTION,
            )
            if gate_vector is not None:
                _LAST_FINAL_GATE_VECTOR = gate_vector
                if reanchored or allow_reseed or _FINAL_GATE_ANCHOR_VECTOR is None:
                    _FINAL_GATE_ANCHOR_VECTOR = gate_vector
        current_associated_gate_vector = gate_vector if associated else None
        action = final_gate_hover_approach_action(
            action,
            current_associated_gate_vector,
            phase_elapsed_s,
            config,
        )
        descent_gate_vector = _LAST_FINAL_GATE_VECTOR
        if config.final_gate_descent_max_forward_m > 0.0:
            descent_gate_vector = current_associated_gate_vector
        (
            descent_gate_vector,
            _FINAL_GATE_DESCENT_PULSE_STARTED_S,
            _FINAL_GATE_DESCENT_PULSE_VECTOR,
            _FINAL_GATE_DESCENT_PULSE_COUNT,
            _FINAL_GATE_DESCENT_PULSE_LAST_END_S,
        ) = final_gate_descent_pulse_vector(
            descent_gate_vector,
            phase_elapsed_s,
            now_s,
            _FINAL_GATE_DESCENT_PULSE_STARTED_S,
            _FINAL_GATE_DESCENT_PULSE_VECTOR,
            _FINAL_GATE_DESCENT_PULSE_COUNT,
            _FINAL_GATE_DESCENT_PULSE_LAST_END_S,
            config,
        )
        action = final_gate_pulse_lateral_hold_action(
            action,
            descent_gate_vector,
            current_associated_gate_vector,
            _FINAL_GATE_DESCENT_PULSE_STARTED_S,
            config,
        )
        action = delayed_final_gate_recovery_action(
            action,
            descent_gate_vector,
            phase_elapsed_s,
            config,
        )
        action = final_gate_pulse_pitch_action(
            action,
            _FINAL_GATE_DESCENT_PULSE_STARTED_S,
            config,
        )
        action = final_gate_dropout_yaw_action(
            action,
            values,
            current_associated_gate_vector,
            config,
        )
        _LAST_FINAL_ACTION = list(action)
    else:
        _FINAL_GATE_PHASE_STARTED_S = None
        _FINAL_GATE_DESCENT_PULSE_STARTED_S = None
        _FINAL_GATE_DESCENT_PULSE_VECTOR = None
        _FINAL_GATE_DESCENT_PULSE_COUNT = 0
        _FINAL_GATE_DESCENT_PULSE_LAST_END_S = None
        _FINAL_GATE_REJECTED_CLUSTER_STARTED_S = None
        _FINAL_GATE_REJECTED_CLUSTER_VECTOR = None
        _FINAL_GATE_REANCHOR_COUNT = 0
        _FINAL_GATE_CLOSE_REANCHOR_COUNT = 0
        _FINAL_GATE_CROSSING_REANCHOR_COUNT = 0
        action = list(base_action)
        _LAST_FINAL_ACTION = None
        _LAST_FINAL_GATE_VECTOR = None
        _FINAL_GATE_ANCHOR_VECTOR = None
        _GATE3_COUNTERBANK_LATCHED = False
        _GATE3_SPEED_BRAKE_LATCHED = False
    _LAST_YAW_ACTION = float(action[3])
    return action
