#!/usr/bin/env python3
"""Competition-facing SITL smoke loop for telemetry + camera + controller wiring."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from drone_camera_receiver import (
    CAMERA_WIDTH,
    DEFAULT_CAMERA_UDP_PORT,
    LatestFrameCameraReceiver,
    UdpCameraReceiver,
)
from drone_course_controller import (
    CourseController,
    CourseControllerConfig,
    CyanGuidanceDetector,
)
from drone_gate_detector import GateDetection, SquareGateDetector, cv2, np
from drone_policy_contract import (
    ATTITUDE_ACTION_FIELDS,
    OBSERVATION_FIELDS,
    OBSERVATION_SIZE,
    ObservableGateMotionFilter,
    GateMotionFilterConfig,
    attitude_last_cmd_normalized,
    decode_attitude_policy_action,
    decode_policy_action,
    normalize_attitude_thrust,
    validate_observation,
)
from drone_sitl_adapter import (
    AttitudeSetpoint,
    LocalNedSetpoint,
    MavlinkSitlAdapter,
    SitlRunReport,
    TelemetryState,
    validate_rates,
)
from drone_state_estimator import (
    AttitudeLoopConfig,
    DeadReckoningEstimator,
    attitude_body_rate_command,
    attitude_loop_command,
    wrap_angle,
)
from drone_visual_servo import estimate_gate_pose_from_corners, visual_servo_command


PHASE_ADAPTER_OBSERVATION_FIELDS = (
    "official_gate_one_active",
    "official_gate_two_active",
    "official_gate_three_active",
    "gate_two_right_norm",
    "gate_two_right_rate_norm",
    "gate_two_down_norm",
    "gate_two_down_rate_norm",
    "gate_three_right_norm",
    "gate_three_right_rate_norm",
)

GATE_PROGRESS_ADAPTER_OBSERVATION_FIELDS = (
    "official_gate_progress_norm",
    "reserved_gate_progress_1",
    "reserved_gate_progress_2",
    "reserved_gate_progress_3",
    "reserved_gate_progress_4",
    "reserved_gate_progress_5",
    "reserved_gate_progress_6",
    "reserved_gate_progress_7",
    "reserved_gate_progress_8",
)

GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS = (
    "official_gate_progress_norm",
    "official_gate_one_active",
    "official_gate_two_active",
    "official_gate_three_active",
    "official_gate_four_active",
    "official_gate_five_active",
    "official_gate_six_active",
    "reserved_gate_phase_1",
    "reserved_gate_phase_2",
)

HYBRID_GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS = (
    *GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS[:-2],
    "legacy_gate_pose_confidence",
    "legacy_elapsed_time_fraction",
)

LEGACY_HYBRID_PREFIX_DURATION_S = 10.5


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    out = float(value)
    return out if math.isfinite(out) else float(default)


def final_phase_any_edge_priority_enabled(
    *,
    final_phase_edge_preference: bool,
    configured: bool,
    phase_elapsed_s: float,
    delay_s: float,
) -> bool:
    """Gate strong edge identity priority until the final handoff settles."""
    return bool(
        final_phase_edge_preference
        and configured
        and float(phase_elapsed_s) >= float(delay_s)
    )


def raw_gate_observation_enabled(
    *,
    official_gate_index: int | None,
    start_index: int,
    end_index: int,
) -> bool:
    """Select a bounded official-gate interval for unfiltered policy poses."""

    if official_gate_index is None or start_index < 0:
        return False
    gate_index = int(official_gate_index)
    return gate_index >= start_index and (end_index < 0 or gate_index <= end_index)


def control_aware_gate_predictor_gain(
    *,
    enabled: bool,
    official_gate_index: int | None,
    start_index: int,
) -> float:
    """Enable the roll-aware predictor only at its validated course phase."""

    if not enabled:
        return 0.0
    if official_gate_index is not None and int(official_gate_index) < start_index:
        return 0.0
    return 4.295


def gate_dropout_prediction_enabled(
    *,
    enabled: bool,
    official_gate_index: int | None,
    start_index: int,
) -> bool:
    """Enable measurement-dropout propagation only at its validated phase."""

    if not enabled:
        return False
    if official_gate_index is None:
        return start_index == 0
    return int(official_gate_index) >= start_index


def update_post_prefix_edge_phase(
    *,
    enabled: bool,
    official_gate_index: int | None,
    now_s: float,
    phase_gate_index: int | None,
    phase_started_s: float | None,
) -> tuple[int | None, float | None]:
    """Reset delayed edge preference independently for each official gate."""

    if not enabled or official_gate_index is None:
        return None, None
    gate_index = int(official_gate_index)
    if phase_gate_index != gate_index or phase_started_s is None:
        return gate_index, float(now_s)
    return phase_gate_index, phase_started_s


class FixedRateAttitudePublisher:
    """Repeat the latest controller target at the required MAVLink wire rate."""

    def __init__(self, adapter, *, command_hz: float, initial_target: AttitudeSetpoint, send_lock):
        self.adapter = adapter
        self.period_s = 1.0 / float(command_hz)
        self._target = initial_target
        self._target_lock = threading.Lock()
        self._send_lock = send_lock
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="course-setpoint-publisher",
            daemon=True,
        )
        self.commands_sent = 0
        self.command_rate_violations = 0
        self.first_command_sent_s: float | None = None
        self.last_command_sent_s: float | None = None
        self.error: Exception | None = None
        self._windows_timer_period_active = False

    def start(self) -> None:
        # The camera/inference loop contains Python-heavy sections. The default
        # 5 ms GIL interval can starve a 60 Hz publisher on Windows even when
        # its timer wakes on schedule. Use a shorter handoff interval; the live
        # zero-command cadence probe remains the promotion authority.
        if sys.getswitchinterval() > 0.001:
            sys.setswitchinterval(0.001)
        if os.name == "nt":
            import ctypes

            self._windows_timer_period_active = (
                ctypes.windll.winmm.timeBeginPeriod(1) == 0
            )
        self._thread.start()

    def update(self, target: AttitudeSetpoint) -> None:
        with self._target_lock:
            self._target = target

    def _run(self) -> None:
        if os.name == "nt":
            import ctypes

            # Prefer the short wire-send thread when it competes with camera
            # decoding/inference for a Windows scheduler quantum.
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 2
            )
        next_send_s = time.monotonic()
        try:
            while not self._stop.is_set():
                now_s = time.monotonic()
                wait_s = next_send_s - now_s
                if wait_s > 0.0:
                    # Release the GIL for most of every wire period so camera
                    # decoding and recurrent inference retain the historical
                    # 10+ Hz update cadence. On Windows, the 1 ms multimedia
                    # timer and final 1 ms hot deadline preserve wire timing
                    # without a full-period busy spin starving the main loop.
                    hot_deadline_s = 0.001 if os.name == "nt" else 0.0
                    sleep_s = min(max(0.0, wait_s - hot_deadline_s), 0.01)
                    if sleep_s > 0.0:
                        self._stop.wait(sleep_s)
                    continue
                with self._target_lock:
                    target = self._target
                with self._send_lock:
                    self.adapter.send_attitude_setpoint(target, mode="body_rates")
                sent_s = time.monotonic()
                if self.first_command_sent_s is None:
                    self.first_command_sent_s = sent_s
                if (
                    self.last_command_sent_s is not None
                    and sent_s - self.last_command_sent_s < 0.01
                ):
                    self.command_rate_violations += 1
                self.last_command_sent_s = sent_s
                self.commands_sent += 1
                next_send_s += self.period_s
                if next_send_s - sent_s < 0.0105:
                    # Recover a missed Windows scheduler quantum at a bounded
                    # 95.2 Hz cadence. This preserves the strict <100 Hz wire
                    # limit while allowing the average stream to stay above
                    # the 50 Hz engineering floor under Python thread contention.
                    next_send_s = sent_s + 0.0105
        except Exception as exc:  # pragma: no cover - live transport failure
            self.error = exc
            self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("course setpoint publisher did not stop")
        if self._windows_timer_period_active:
            import ctypes

            ctypes.windll.winmm.timeEndPeriod(1)
            self._windows_timer_period_active = False


def calculate_effective_command_rate(
    *,
    commands_sent: int,
    first_command_sent_s: float | None,
    last_command_sent_s: float | None,
    fallback_duration_s: float,
) -> tuple[float, float]:
    """Return wire rate and the interval used to measure it.

    Official-reset calibration can precede the first setpoint by several
    seconds. It belongs to attempt elapsed time but not to the command stream.
    """

    if commands_sent <= 0:
        return 0.0, 0.0
    if (
        commands_sent > 1
        and first_command_sent_s is not None
        and last_command_sent_s is not None
        and last_command_sent_s > first_command_sent_s
    ):
        interval_s = last_command_sent_s - first_command_sent_s
        return (commands_sent - 1) / interval_s, interval_s
    interval_s = max(float(fallback_duration_s), 1e-9)
    return commands_sent / interval_s, interval_s


class _NoopAttitudeAdapter:
    """Exercise publisher scheduling without writing a MAVLink setpoint."""

    def send_attitude_setpoint(self, _target, *, mode: str = "body_rates") -> None:
        if mode != "body_rates":
            raise ValueError("shadow cadence probe requires body_rates mode")


@dataclasses.dataclass(frozen=True)
class OfficialResetStartResult:
    """Evidence that an in-place reset reached the timed-race boundary."""

    race_start_monotonic_s: float
    last_imu_time_usec: int | None
    report: dict


def reset_and_wait_for_official_policy_start(
    adapter,
    estimator: DeadReckoningEstimator,
    *,
    heartbeat_hz: float,
    timeout_s: float = 12.0,
    policy_lead_s: float = 0.05,
    min_calibration_samples: int = 60,
    monotonic_fn=None,
    sleep_fn=None,
) -> OfficialResetStartResult:
    """Reset in place and stop immediately before the scheduled race start.

    v3385 publishes ``race_start_boot_time_ms`` during its countdown.  Waiting
    merely for that value to become non-negative starts the controller several
    seconds late.  This helper instead maps the simulator's scheduled boot-time
    boundary to the local monotonic clock, calibrates the observable IMU while
    the vehicle is stationary, and returns just early enough for the policy's
    first 60 Hz target to be installed before the timer begins.
    """

    if heartbeat_hz <= 0.0:
        raise ValueError("heartbeat_hz must be positive")
    if timeout_s <= 0.0:
        raise ValueError("official reset start timeout must be positive")
    if policy_lead_s < 0.0:
        raise ValueError("official policy lead must be non-negative")
    if min_calibration_samples <= 0:
        raise ValueError("minimum calibration samples must be positive")

    monotonic = time.monotonic if monotonic_fn is None else monotonic_fn
    sleep = time.sleep if sleep_fn is None else sleep_fn
    heartbeat_period_s = 1.0 / heartbeat_hz
    wait_started_s = monotonic()
    deadline_s = wait_started_s + timeout_s

    # Learn both the component IDs and the current boot counter before sending
    # the simulator-specific command.  This makes a missing/ignored reset
    # distinguishable from a slow countdown.
    while monotonic() < deadline_s:
        adapter.drain_telemetry(timeout_s=0.05)
        if (
            adapter.telemetry.metrics.heartbeats > 0
            and adapter.telemetry.state.race_status is not None
        ):
            break
    else:
        raise RuntimeError("official reset preflight did not receive heartbeat and race status")

    pre_status = adapter.telemetry.state.race_status
    pre_boot_ms = int(pre_status.sim_boot_time_ms)
    adapter.send_heartbeat()
    # v3385 ignores command 31000 while the UI-started vehicle remains armed
    # (base_mode 193). Match the live-validated reset snapshot sequence.
    adapter.send_disarm_command()
    sleep(0.1)
    adapter.send_sim_reset_command()
    reset_command_s = monotonic()

    reset_seen_s = None
    scheduled_start_s = None
    last_imu_time_usec = None
    next_heartbeat_s = 0.0
    next_arm_s = 0.0
    arm_commands_sent = 0
    calibration_report: dict = {}

    while monotonic() < deadline_s:
        now_s = monotonic()
        adapter.drain_telemetry(timeout_s=0.02)
        status = adapter.telemetry.state.race_status
        if status is None:
            sleep(0.001)
            continue

        sim_boot_ms = int(status.sim_boot_time_ms)
        if reset_seen_s is None and sim_boot_ms + 1000 < pre_boot_ms:
            reset_seen_s = now_s

        if reset_seen_s is not None:
            imu_time_usec = adapter.telemetry.state.imu_time_usec
            if imu_time_usec is not None and imu_time_usec != last_imu_time_usec:
                last_imu_time_usec = int(imu_time_usec)
                telemetry = adapter.telemetry.state
                if (
                    telemetry.xacc is not None
                    and telemetry.yacc is not None
                    and telemetry.zacc is not None
                ):
                    estimator.add_calibration_sample(
                        (telemetry.xacc, telemetry.yacc, telemetry.zacc),
                        (
                            telemetry.xgyro or 0.0,
                            telemetry.ygyro or 0.0,
                            telemetry.zgyro or 0.0,
                        ),
                    )
                    if (
                        not estimator.state.calibrated
                        and estimator.calibration_samples >= min_calibration_samples
                    ):
                        calibration_report = estimator.finish_calibration()

            race_start_boot_ms = int(status.race_start_boot_time_ms)
            if race_start_boot_ms >= 0:
                scheduled_start_s = now_s + max(
                    0.0, (race_start_boot_ms - sim_boot_ms) * 1e-3
                )

                # Arming is deterministic lifecycle work, but do it only near
                # the scheduled boundary so no stale target can move the reset
                # vehicle during IMU calibration.
                if now_s >= next_arm_s and scheduled_start_s - now_s <= 0.25:
                    adapter.send_arm_command()
                    arm_commands_sent += 1
                    next_arm_s = now_s + 0.1

                if (
                    estimator.state.calibrated
                    and scheduled_start_s - now_s <= policy_lead_s
                ):
                    return OfficialResetStartResult(
                        race_start_monotonic_s=scheduled_start_s,
                        last_imu_time_usec=last_imu_time_usec,
                        report={
                            "reset_sent": True,
                            "pre_reset_disarm_sent": True,
                            "pre_reset_sim_boot_time_ms": pre_boot_ms,
                            "post_reset_sim_boot_time_ms": sim_boot_ms,
                            "race_start_boot_time_ms": race_start_boot_ms,
                            "reset_detected": True,
                            "reset_detection_s": round(reset_seen_s - reset_command_s, 6),
                            "race_start_wait_s": round(now_s - reset_command_s, 6),
                            "policy_lead_s": round(max(0.0, scheduled_start_s - now_s), 6),
                            "calibration_samples": int(
                                calibration_report.get("samples", 0)
                            ),
                            "calibration": calibration_report,
                            "arm_commands_sent": arm_commands_sent,
                        },
                    )

        if now_s >= next_heartbeat_s:
            adapter.send_heartbeat()
            next_heartbeat_s = now_s + heartbeat_period_s
        sleep(0.001)

    if reset_seen_s is None:
        raise RuntimeError("MAVLink 31000 was not acknowledged by a simulator boot reset")
    if scheduled_start_s is None:
        raise RuntimeError("official race did not publish a scheduled start after reset")
    if not estimator.state.calibrated:
        raise RuntimeError(
            "official reset countdown ended without enough stationary IMU calibration samples"
        )
    raise RuntimeError("official reset start timed out before the policy boundary")


def maybe_parse_action_json(action_json: str) -> tuple[float, float, float, float]:
    values = json.loads(action_json)
    if not isinstance(values, list):
        raise ValueError("--policy-action-json must decode to a JSON list")
    action, _setpoint = decode_policy_action(values, yaw_rad=0.0)
    return action.normalized


def load_acceptance_config(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("acceptance config must decode to a JSON object")
    return payload


def resolve_config_value(config: dict, section: str, key: str, default):
    section_obj = config.get(section, {})
    if not isinstance(section_obj, dict):
        return default
    return section_obj.get(key, default)


def load_course_controller_config(args) -> CourseControllerConfig:
    """Load one tunable course config, with optional explicit CLI overrides."""
    path = str(
        getattr(args, "course_controller_config", "config/course_fsm_defaults.json") or ""
    )
    values: dict = {}
    if path and os.path.exists(path):
        with open(path) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("course controller config must decode to a JSON object")
        values.update(payload.get("controller", payload))

    override_fields = {
        "course_hover_thrust": "hover_thrust",
        "course_min_thrust": "min_thrust",
        "course_max_thrust": "max_thrust",
        "course_guidance_pitch_rad": "guidance_pitch_rad",
        "course_guidance_yaw_gain": "guidance_yaw_gain",
        "course_guidance_max_yaw_step_rad": "guidance_max_yaw_step_rad",
        "course_guidance_search_rate_rad_s": "guidance_lost_search_rate_rad_s",
        "course_align_yaw_tolerance_rad": "align_yaw_tolerance_rad",
        "course_align_vertical_tolerance_m": "align_vertical_tolerance_m",
        "course_align_ticks": "align_consecutive_ticks",
        "course_approach_pitch_rad": "approach_pitch_rad",
        "course_commit_range_m": "commit_range_m",
        "course_commit_pitch_rad": "commit_pitch_rad",
        "course_commit_duration_s": "commit_duration_s",
        "course_brake_pitch_rad": "brake_pitch_rad",
        "course_brake_duration_s": "brake_duration_s",
        "course_brake_settle_s": "brake_settle_s",
    }
    for arg_name, field_name in override_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            values[field_name] = value
    return CourseControllerConfig(**values)


def roll_pitch_yaw_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def rotate_vector_by_quat(
    vector: Sequence[float], quat_wxyz: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a 3-vector by a unit quaternion using observable attitude."""
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


def body_velocity_from_estimator(estimator: DeadReckoningEstimator) -> tuple[float, float, float]:
    vx, vy, vz = estimator.state.velocity_ned_m_s
    yaw = estimator.state.yaw
    c = math.cos(yaw)
    s = math.sin(yaw)
    forward = c * vx + s * vy
    right = -s * vx + c * vy
    down = -vz
    return (forward, right, down)


def build_policy_observation(
    telemetry: TelemetryState,
    *,
    gate_detection: GateDetection | None,
    gate_pose,
    elapsed_fraction: float,
    last_cmd_norm: Sequence[float],
    estimator_state=None,
    guidance: GuidanceDetection | None = None,
    observation_time_s: float | None = None,
    gate_motion_state: ObservableGateMotionFilter | None = None,
    gate_identity=None,
    race_phase_gate_index: int | None = None,
    race_phase_denominator: int = 3,
    phase_adapter_gate_index: int | None = None,
    gate_progress_adapter_gate_index: int | None = None,
    gate_progress_adapter_denominator: int = 6,
    gate_phase_onehot_adapter_gate_index: int | None = None,
    gate_phase_onehot_adapter_denominator: int = 6,
    hybrid_prefix_confidence_observation: bool = False,
    hybrid_prefix_elapsed_fraction: float | None = None,
) -> tuple[float, ...]:
    from drone_policy_contract import build_ts002_observation

    if estimator_state is not None and getattr(estimator_state, "calibrated", False):
        roll = float(estimator_state.roll)
        pitch = float(estimator_state.pitch)
        yaw = float(estimator_state.yaw)
        quat = roll_pitch_yaw_to_quat(roll, pitch, yaw)
    else:
        roll = safe_float(telemetry.roll)
        pitch = safe_float(telemetry.pitch)
        yaw = safe_float(telemetry.yaw)
        quat = roll_pitch_yaw_to_quat(roll, pitch, yaw)

    gyro_roll = safe_float(telemetry.xgyro, default=safe_float(telemetry.rollspeed))
    gyro_pitch = safe_float(telemetry.ygyro, default=safe_float(telemetry.pitchspeed))
    gyro_yaw = safe_float(telemetry.zgyro, default=safe_float(telemetry.yawspeed))

    gate_visible = gate_pose is not None
    gate_forward_m = 0.0
    gate_right_m = 0.0
    gate_down_m = 0.0
    gate_yaw_error_rad = 0.0
    gate_range_m = None
    gate_forward_rate_m_s = 0.0
    gate_right_rate_m_s = 0.0
    gate_down_rate_m_s = 0.0
    if gate_pose is not None:
        gate_forward_m, gate_right_m, gate_down_m = gate_pose.body_vector_ned_m
        gate_yaw_error_rad = float(gate_pose.yaw_error_rad)
        gate_range_m = float(gate_pose.range_camera_m)
        if gate_motion_state is not None and observation_time_s is not None:
            (
                gate_forward_rate_m_s,
                gate_right_rate_m_s,
                gate_down_rate_m_s,
            ) = gate_motion_state.update(
                (gate_forward_m, gate_right_m, gate_down_m),
                quat,
                observation_time_s,
                identity=gate_identity,
            )
            tracked_pose = gate_motion_state.tracked_body_vector_ned(quat)
            if tracked_pose is not None:
                gate_forward_m, gate_right_m, gate_down_m = tracked_pose
                gate_range_m = math.sqrt(
                    gate_forward_m * gate_forward_m
                    + gate_right_m * gate_right_m
                    + gate_down_m * gate_down_m
                )
                gate_yaw_error_rad = math.atan2(
                    gate_right_m, max(gate_forward_m, 1e-4)
                )
            else:
                gate_visible = False
                gate_forward_m = 0.0
                gate_right_m = 0.0
                gate_down_m = 0.0
                gate_yaw_error_rad = 0.0
                gate_range_m = None
    elif gate_motion_state is not None:
        (
            gate_forward_rate_m_s,
            gate_right_rate_m_s,
            gate_down_rate_m_s,
        ), tracked_pose = gate_motion_state.hold_without_measurement(
            quat,
            identity=gate_identity,
            time_s=observation_time_s,
        )
        if tracked_pose is not None:
            gate_visible = True
            gate_forward_m, gate_right_m, gate_down_m = tracked_pose
            gate_range_m = math.sqrt(
                gate_forward_m * gate_forward_m
                + gate_right_m * gate_right_m
                + gate_down_m * gate_down_m
            )
            gate_yaw_error_rad = math.atan2(
                gate_right_m, max(gate_forward_m, 1e-4)
            )

    observation = build_ts002_observation(
        guidance_visible=guidance is not None,
        guidance_center_x_norm=(0.0 if guidance is None else guidance.center_x_norm),
        guidance_heading_error_rad=(
            0.0 if guidance is None else guidance.heading_error_rad
        ),
        gate_forward_rate_m_s=gate_forward_rate_m_s,
        gate_right_rate_m_s=gate_right_rate_m_s,
        gate_down_rate_m_s=gate_down_rate_m_s,
        gyro_roll_rad_s=gyro_roll,
        gyro_pitch_rad_s=gyro_pitch,
        gyro_yaw_rad_s=gyro_yaw,
        quat_wxyz=quat,
        gate_visible=gate_visible,
        gate_forward_m=gate_forward_m,
        gate_right_m=gate_right_m,
        gate_down_m=gate_down_m,
        gate_yaw_error_rad=gate_yaw_error_rad,
        gate_range_m=gate_range_m,
        elapsed_fraction=elapsed_fraction,
        last_cmd=last_cmd_norm,
    )
    if race_phase_gate_index is not None:
        denominator = max(int(race_phase_denominator), 1)
        values = list(observation)
        values[-1] = clamp(float(race_phase_gate_index) / denominator, 0.0, 1.0)
        observation = tuple(values)
    if phase_adapter_gate_index is not None:
        gate_index = int(phase_adapter_gate_index)
        gate_one = 1.0 if gate_index == 1 else 0.0
        gate_two = 1.0 if gate_index == 2 else 0.0
        gate_three = 1.0 if gate_index == 3 else 0.0
        observation = (
            *observation,
            gate_one,
            gate_two,
            gate_three,
            gate_two * observation[12],
            gate_two * observation[1],
            gate_two * observation[13],
            gate_two * observation[2],
            gate_three * observation[12],
            gate_three * observation[1],
        )
    if gate_progress_adapter_gate_index is not None:
        denominator = max(int(gate_progress_adapter_denominator), 1)
        gate_progress = clamp(
            float(gate_progress_adapter_gate_index) / denominator, 0.0, 1.0
        )
        observation = (*observation, gate_progress, *(0.0 for _ in range(8)))
    if gate_phase_onehot_adapter_gate_index is not None:
        denominator = max(int(gate_phase_onehot_adapter_denominator), 1)
        gate_index = int(gate_phase_onehot_adapter_gate_index)
        gate_progress = clamp(float(gate_index) / denominator, 0.0, 1.0)
        gate_flags = tuple(1.0 if gate_index == index else 0.0 for index in range(6))
        legacy_confidence = 0.0
        legacy_elapsed_fraction = 0.0
        if (
            hybrid_prefix_confidence_observation
            and gate_index <= 2
        ):
            if hybrid_prefix_elapsed_fraction is None:
                raise ValueError(
                    "hybrid prefix observation requires legacy elapsed fraction"
                )
            if gate_pose is not None:
                legacy_confidence = clamp(
                    float(getattr(gate_pose, "confidence", 0.0)), 0.0, 1.0
                )
            legacy_elapsed_fraction = clamp(
                float(hybrid_prefix_elapsed_fraction), 0.0, 1.0
            )
        observation = (
            *observation,
            gate_progress,
            *gate_flags,
            legacy_confidence,
            legacy_elapsed_fraction,
        )
    elif hybrid_prefix_confidence_observation:
        raise ValueError(
            "hybrid prefix confidence requires the gate-phase one-hot adapter"
        )
    expected_size = (
        32
        if phase_adapter_gate_index is not None
        or gate_progress_adapter_gate_index is not None
        or gate_phase_onehot_adapter_gate_index is not None
        else OBSERVATION_SIZE
    )
    if len(observation) != expected_size:
        raise RuntimeError(
            f"observation size mismatch: expected {expected_size}, got {len(observation)}"
        )
    return observation


PolicyCallable = Callable[[tuple[float, ...]], Sequence[float]]


def load_policy_callable(spec: str) -> PolicyCallable:
    path_str, sep, function_name = spec.rpartition(":")
    if not sep or not function_name:
        raise ValueError("--policy-callable must be in '<path.py>:<function>' format")
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"policy callable file not found: {path}")
    module_name = f"_policy_callable_{int(time.time() * 1e6)}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"unable to load policy callable module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"function '{function_name}' not found in {path}")
    function._policy_module = module  # type: ignore[attr-defined]
    return function


def reset_policy_callable(policy_callable: PolicyCallable | None) -> None:
    if policy_callable is None:
        return
    module = getattr(policy_callable, "_policy_module", None)
    if module is None:
        return
    reset_fn = getattr(module, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _runtime_file_record(path: Path | None) -> dict:
    if path is None:
        return {"path": None, "sha256": None}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "sha256": None}
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def build_deployment_manifest(policy_callable: PolicyCallable | None) -> dict:
    """Hash the exact source/checkpoint bytes used by this runtime."""

    scripts_dir = Path(__file__).resolve().parent
    module = getattr(policy_callable, "_policy_module", None)
    callable_file = getattr(module, "__file__", None)
    checkpoint_value = os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "")
    gate2_checkpoint_value = os.getenv(
        "PUFFER_POLICY_GATE2_CHECKPOINT_PATH", ""
    )
    gate4_checkpoint_value = os.getenv(
        "PUFFER_POLICY_GATE4_CHECKPOINT_PATH", ""
    )
    gate5_checkpoint_value = os.getenv(
        "PUFFER_POLICY_GATE5_CHECKPOINT_PATH", ""
    )
    gate6_checkpoint_value = os.getenv(
        "PUFFER_POLICY_GATE6_CHECKPOINT_PATH", ""
    )
    paths = {
        "runner": Path(__file__),
        "policy_callable": None if callable_file is None else Path(callable_file),
        "policy_contract": scripts_dir / "drone_policy_contract.py",
        "gate_detector": scripts_dir / "drone_gate_detector.py",
        "camera_receiver": scripts_dir / "drone_camera_receiver.py",
        "sitl_adapter": scripts_dir / "drone_sitl_adapter.py",
        "state_estimator": scripts_dir / "drone_state_estimator.py",
        "checkpoint": None if not checkpoint_value else Path(checkpoint_value),
        "gate2_checkpoint": (
            None if not gate2_checkpoint_value else Path(gate2_checkpoint_value)
        ),
        "gate4_checkpoint": (
            None if not gate4_checkpoint_value else Path(gate4_checkpoint_value)
        ),
        "gate5_checkpoint": (
            None if not gate5_checkpoint_value else Path(gate5_checkpoint_value)
        ),
        "gate6_checkpoint": (
            None if not gate6_checkpoint_value else Path(gate6_checkpoint_value)
        ),
    }
    return {name: _runtime_file_record(path) for name, path in paths.items()}


@dataclasses.dataclass
class SmokeVisionMetrics:
    frames_seen: int = 0
    detector_detections: int = 0
    detector_decode_failures: int = 0
    detector_no_quad_found: int = 0
    last_detection_age_s: float | None = None
    last_detection_confidence: float | None = None
    timesync_samples: int = 0
    average_timesync_error_ns: float = 0.0
    max_timesync_error_ns: int = 0


@dataclasses.dataclass
class ApproachDiagnostics:
    detections_sampled: int = 0
    closest_range_m: float | None = None
    closest_range_elapsed_s: float | None = None
    closest_range_confidence: float | None = None
    closest_range_body_vector_ned_m: tuple[float, float, float] | None = None
    closest_range_yaw_error_rad: float | None = None
    closest_range_command: dict = dataclasses.field(default_factory=dict)
    highest_confidence: float | None = None
    highest_confidence_elapsed_s: float | None = None
    latest_pose: dict = dataclasses.field(default_factory=dict)
    latest_command: dict = dataclasses.field(default_factory=dict)
    samples: list[dict] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AttitudeFinalApproachState:
    active_until_s: float | None = None
    active_command: AttitudeSetpoint | None = None
    activations: int = 0
    last_activation_s: float | None = None
    last_activation_elapsed_s: float | None = None
    last_trigger_pose: dict = dataclasses.field(default_factory=dict)
    last_command: dict = dataclasses.field(default_factory=dict)

    def is_active(self, now_s: float) -> bool:
        return self.active_until_s is not None and now_s < self.active_until_s

    def activate(
        self,
        *,
        now_s: float,
        started_s: float,
        duration_s: float,
        pose,
        command: AttitudeSetpoint,
    ) -> None:
        self.active_until_s = now_s + duration_s
        self.active_command = command
        self.activations += 1
        self.last_activation_s = now_s
        self.last_activation_elapsed_s = round_float(now_s - started_s)
        self.last_trigger_pose = pose_to_dict(pose)
        self.last_command = attitude_setpoint_to_dict(command)

    def to_summary(self, *, now_s: float, started_s: float) -> dict:
        return {
            "active": self.is_active(now_s),
            "active_until_elapsed_s": (
                None if self.active_until_s is None else round_float(self.active_until_s - started_s)
            ),
            "active_remaining_s": (
                None if self.active_until_s is None else round_float(max(0.0, self.active_until_s - now_s))
            ),
            "activations": self.activations,
            "last_activation_elapsed_s": self.last_activation_elapsed_s,
            "last_trigger_pose": self.last_trigger_pose,
            "last_command": self.last_command,
        }


@dataclasses.dataclass
class AttitudeAngleServoState:
    """Angle-mode visual servo for the v3379 sim FC.

    This class produces desired absolute roll/pitch/yaw angles plus thrust. The
    surrounding SITL loop closes those desired angles with the gyro AHRS because
    v3379 interprets the quaternion as a rate demand, not an absolute attitude.
    """

    yaw_cmd_rad: float = 0.0
    blind_command: AttitudeSetpoint | None = None
    blind_until_s: float | None = None
    blind_activations: int = 0
    last_pose_range_m: float | None = None
    # Filtered gate vector and its derivative. The gate is world-fixed, so the
    # pose derivative is a direct (negated) velocity estimate — the only one
    # available since v3379 publishes no attitude/position telemetry.
    filt_vec: tuple[float, float, float] | None = None
    filt_rate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    filt_time_s: float | None = None
    continuity_rejects: int = 0
    track_rejections: int = 0
    extrapolated_ticks: int = 0
    control_state: str = "search"
    # Integral trim on thrust: the true hover point is unknown (the sim
    # accelerometer is uninformative, so it cannot be measured on the pad)
    # and a wrong hover guess leaves a permanent altitude offset. The
    # integrator absorbs it using the visual vertical error.
    thrust_bias: float = 0.0

    def _predicted_vec(self, now_s: float) -> tuple[float, float, float] | None:
        if self.filt_vec is None or self.filt_time_s is None:
            return None
        dt = now_s - self.filt_time_s
        return tuple(v + r * dt for v, r in zip(self.filt_vec, self.filt_rate))

    def _observe_pose(
        self, pose, now_s: float, *, jump_gate_m: float, alpha: float = 0.35
    ) -> tuple[float, float, float] | None:
        """Fold a detection into the track; returns None if it fails continuity."""
        bx, by, bz = (float(v) for v in pose.body_vector_ned_m)
        predicted = self._predicted_vec(now_s)
        if predicted is not None and jump_gate_m > 0.0:
            jump = math.sqrt(sum((n - p) ** 2 for n, p in zip((bx, by, bz), predicted)))
            if jump > jump_gate_m:
                self.continuity_rejects += 1
                self.track_rejections += 1
                # Three consecutive far detections mean the old track is gone;
                # re-acquire on the new gate.
                if self.continuity_rejects < 3:
                    return None
                self.filt_vec = None
                self.filt_time_s = None
                self.filt_rate = (0.0, 0.0, 0.0)
        self.continuity_rejects = 0
        if self.filt_vec is None or self.filt_time_s is None:
            self.filt_vec = (bx, by, bz)
            self.filt_time_s = now_s
            self.filt_rate = (0.0, 0.0, 0.0)
            return self.filt_vec
        dt = now_s - self.filt_time_s
        prev = self.filt_vec
        new = tuple(alpha * n + (1.0 - alpha) * p for n, p in zip((bx, by, bz), prev))
        if dt > 1e-3:
            raw_rate = tuple((n - p) / dt for n, p in zip(new, prev))
            self.filt_rate = tuple(
                0.5 * r + 0.5 * f for r, f in zip(raw_rate, self.filt_rate)
            )
        self.filt_vec = new
        self.filt_time_s = now_s
        return new

    def update(self, *, now_s: float, pose, args, dt: float) -> AttitudeSetpoint:
        k_pitch = float(getattr(args, "angle_servo_k_pitch", 0.02))
        max_pitch = abs(float(getattr(args, "angle_servo_max_pitch_rad", 0.12)))
        k_roll = float(getattr(args, "angle_servo_k_roll", 0.0))
        max_roll = abs(float(getattr(args, "angle_servo_max_roll_rad", 0.10)))
        k_yaw = float(getattr(args, "angle_servo_k_yaw", 0.8))
        max_yaw_step = abs(float(getattr(args, "angle_servo_max_yaw_step_rad", 0.02)))
        hover = float(getattr(args, "attitude_servo_hover_thrust", 0.62))
        min_thrust = float(getattr(args, "attitude_servo_min_thrust", 0.5))
        max_thrust = float(getattr(args, "attitude_servo_max_thrust", 0.72))
        k_thrust = float(getattr(args, "attitude_servo_k_thrust", 0.05))
        blind_range = float(getattr(args, "angle_servo_blind_range_m", 4.0))
        blind_duration = float(getattr(args, "angle_servo_blind_duration_s", 2.0))
        blind_pitch = -abs(float(getattr(args, "angle_servo_blind_pitch_rad", 0.10)))
        search_yaw_rate = float(getattr(args, "angle_servo_search_yaw_rate_rad_s", 0.0))
        k_dz = float(getattr(args, "angle_servo_k_dz", 0.03))
        k_dx = float(getattr(args, "angle_servo_k_dx", 0.01))
        jump_gate_m = float(getattr(args, "angle_servo_track_jump_gate_m", 5.0))
        coast_s = float(getattr(args, "angle_servo_track_coast_s", 1.5))

        if self.blind_until_s is not None:
            if now_s < self.blind_until_s and self.blind_command is not None:
                self.control_state = "blind"
                return self.blind_command
            # Blind carry-through expired: drop state so it cannot re-trigger
            # off the same stale pose.
            self.blind_until_s = None
            self.blind_command = None
            self.last_pose_range_m = None
            self.filt_vec = None
            self.filt_time_s = None
            self.filt_rate = (0.0, 0.0, 0.0)

        vec = None
        if pose is not None:
            vec = self._observe_pose(pose, now_s, jump_gate_m=jump_gate_m)

        if vec is None:
            # No usable detection this tick: coast on the track prediction so a
            # brief FOV dropout or a continuity-rejected detection does not
            # reset the approach.
            if (
                self.filt_time_s is not None
                and coast_s > 0.0
                and now_s - self.filt_time_s <= coast_s
            ):
                vec = self._predicted_vec(now_s)
                self.extrapolated_ticks += 1

        if vec is not None:
            bx, by, bz = vec
            dx, _dy, dz = self.filt_rate
            yaw_err = math.atan2(by, max(bx, 1e-6))
            self.yaw_cmd_rad += clamp(k_yaw * yaw_err * dt, -max_yaw_step, max_yaw_step)
            # PD on forward range: dx is the gate-vector rate (negative when
            # closing), so -k_dx*dx opposes excess closure speed.
            pitch = -clamp(k_pitch * max(0.0, bx) + k_dx * (-dx), -max_pitch, max_pitch)
            roll = clamp(k_roll * by, -max_roll, max_roll)
            # PID on vertical offset: bz negative means gate above (climb);
            # dz damps the vertical rate; the bias integrator finds the true
            # hover thrust so the P term is not stuck holding a steady error.
            # The P input is clamped: a 10 m altitude error must produce a
            # *bounded* climb command, not seconds of saturated thrust that
            # launch the drone far above the whole course.
            bz_cap = float(getattr(args, "angle_servo_max_bz_m", 2.5))
            k_ithrust = float(getattr(args, "angle_servo_k_ithrust", 0.02))
            if abs(bz) < 3.0:
                # Anti-windup: only trim near the target altitude.
                self.thrust_bias = clamp(self.thrust_bias - k_ithrust * bz * dt, -0.15, 0.15)
            thrust = clamp(
                hover + self.thrust_bias - k_thrust * clamp(bz, -bz_cap, bz_cap) - k_dz * dz,
                min_thrust,
                max_thrust,
            )
            range_now = math.sqrt(bx * bx + by * by + bz * bz)
            self.last_pose_range_m = range_now
            self.control_state = "track"
            if blind_duration > 0.0 and range_now <= blind_range:
                # Stage the carry-through command; it activates when the gate
                # inevitably leaves the camera FOV on final approach.
                self.blind_command = AttitudeSetpoint(
                    roll=0.0,
                    pitch=blind_pitch,
                    yaw=self.yaw_cmd_rad,
                    thrust=clamp(hover + self.thrust_bias, min_thrust, max_thrust),
                )
            return AttitudeSetpoint(roll=roll, pitch=pitch, yaw=self.yaw_cmd_rad, thrust=thrust)

        if (
            self.blind_command is not None
            and self.last_pose_range_m is not None
            and self.last_pose_range_m <= blind_range
        ):
            self.blind_until_s = now_s + blind_duration
            self.blind_activations += 1
            self.control_state = "blind"
            return self.blind_command

        self.yaw_cmd_rad += search_yaw_rate * dt
        self.filt_vec = None
        self.filt_time_s = None
        self.filt_rate = (0.0, 0.0, 0.0)
        self.control_state = "search"
        search_thrust = getattr(args, "angle_servo_search_thrust", None)
        # Default search thrust sits below hover so a blind drone sinks back
        # into the gate sightline instead of climbing away from it.
        thrust = hover - 0.04 if search_thrust is None else float(search_thrust)
        return AttitudeSetpoint(
            roll=0.0,
            pitch=0.0,
            yaw=self.yaw_cmd_rad,
            thrust=clamp(thrust + self.thrust_bias, min_thrust, max_thrust),
        )

    def reset_for_gate_retarget(self, *, yaw_cmd_rad: float | None = None) -> None:
        """Drop the visual track after official active_gate_index advances."""
        self.blind_command = None
        self.blind_until_s = None
        self.last_pose_range_m = None
        self.filt_vec = None
        self.filt_time_s = None
        self.filt_rate = (0.0, 0.0, 0.0)
        self.continuity_rejects = 0
        self.control_state = "search"
        if yaw_cmd_rad is not None:
            self.yaw_cmd_rad = yaw_cmd_rad

    def to_summary(self) -> dict:
        return {
            "yaw_cmd_rad": round_float(self.yaw_cmd_rad),
            "thrust_bias": round_float(self.thrust_bias),
            "blind_activations": self.blind_activations,
            "blind_active": self.blind_until_s is not None,
            "last_pose_range_m": round_float(self.last_pose_range_m)
            if self.last_pose_range_m is not None
            else None,
            "track_rejections": self.track_rejections,
            "extrapolated_ticks": self.extrapolated_ticks,
            "control_state": self.control_state,
        }


@dataclasses.dataclass
class MapGuidanceState:
    """Fly at the estimator's mapped gate landmark when vision has no target.

    Vision goes blind inside ~4 m of the gate (the aperture leaves the FOV),
    which is exactly where the old open-loop blind carry-through tumbled the
    drone. With the AHRS + landmark map, the relative gate position stays
    known through the dropout, so the final approach can stay closed-loop:
    steer at the mapped gate center and, once close, at a carry-through point
    past the gate plane along the frozen approach bearing.
    """

    ticks: int = 0
    carry_activations: int = 0
    carry_bearing: tuple[float, float] | None = None

    def command(
        self, estimator, gate_id: int, args, thrust_bias: float = 0.0
    ) -> AttitudeSetpoint | None:
        landmark = estimator.landmarks.get(gate_id)
        if landmark is None or not estimator.state.calibrated:
            return None
        st = estimator.state
        px, py, pz = st.position_ned_m
        gx, gy, gz = landmark
        rel = (gx - px, gy - py, gz - pz)
        horiz = math.hypot(rel[0], rel[1])

        carry_m = float(getattr(args, "angle_servo_carry_through_m", 2.0))
        freeze_range = float(getattr(args, "angle_servo_carry_freeze_range_m", 3.0))
        if self.carry_bearing is None and 1e-6 < horiz <= freeze_range:
            self.carry_bearing = (rel[0] / horiz, rel[1] / horiz)
            self.carry_activations += 1
        elif self.carry_bearing is not None and horiz > freeze_range + 2.0:
            self.carry_bearing = None
        if self.carry_bearing is not None:
            rel = (
                gx + self.carry_bearing[0] * carry_m - px,
                gy + self.carry_bearing[1] * carry_m - py,
                gz - pz,
            )
            horiz = math.hypot(rel[0], rel[1])

        desired_yaw = math.atan2(rel[1], rel[0])
        yaw_err = wrap_angle(desired_yaw - st.yaw)
        vx, vy, vz = st.velocity_ned_m_s
        v_fwd = vx * math.cos(st.yaw) + vy * math.sin(st.yaw)

        max_speed = float(getattr(args, "angle_servo_max_speed_m_s", 2.0))
        k_speed = float(getattr(args, "angle_servo_k_speed", 0.06))
        max_pitch = abs(float(getattr(args, "angle_servo_max_pitch_rad", 0.12)))
        # Speed-loop pitch: only drive forward when roughly aimed at the target.
        desired_speed = 0.0 if abs(yaw_err) > 0.5 else min(max_speed, 0.6 * horiz)
        pitch = -clamp(k_speed * (desired_speed - v_fwd), -max_pitch, max_pitch)

        hover = float(getattr(args, "attitude_servo_hover_thrust", 0.6))
        min_thrust = float(getattr(args, "attitude_servo_min_thrust", 0.5))
        max_thrust = float(getattr(args, "attitude_servo_max_thrust", 0.72))
        k_thrust = float(getattr(args, "attitude_servo_k_thrust", 0.04))
        k_vz = float(getattr(args, "angle_servo_k_vz", 0.02))
        bz_cap = float(getattr(args, "angle_servo_max_bz_m", 2.5))
        # P on vertical offset (rel z down-positive), D opposing vertical rate.
        thrust = clamp(
            hover + thrust_bias - k_thrust * clamp(rel[2], -bz_cap, bz_cap) + k_vz * vz,
            min_thrust,
            max_thrust,
        )

        self.ticks += 1
        return AttitudeSetpoint(roll=0.0, pitch=pitch, yaw=desired_yaw, thrust=thrust)

    def reset_carry(self) -> None:
        self.carry_bearing = None

    def to_summary(self) -> dict:
        return {
            "ticks": self.ticks,
            "carry_activations": self.carry_activations,
            "carry_bearing": (
                None
                if self.carry_bearing is None
                else tuple(round(v, 6) for v in self.carry_bearing)
            ),
        }


@dataclasses.dataclass
class OfficialGateObservationEpoch:
    """Report official gate identity changes that invalidate cached vision."""

    gate_index: int | None = None

    def observe(self, official_gate_index: int | None) -> bool:
        if official_gate_index is None:
            return False
        current = int(official_gate_index)
        if self.gate_index is None:
            self.gate_index = current
            return False
        changed = current != self.gate_index
        self.gate_index = current
        return changed


@dataclasses.dataclass
class OfficialGateProgressState:
    """Retarget control when official active_gate_index advances."""

    last_official_gate_index: int | None = None
    advance_count: int = 0
    retarget_count: int = 0
    events: list[dict] = dataclasses.field(default_factory=list)

    def observe(
        self,
        *,
        elapsed_s: float,
        official_gate_index: int | None,
        estimator,
        angle_servo: AttitudeAngleServoState,
        map_guidance: MapGuidanceState,
        pass_tracker: VisionGatePassTracker,
    ) -> bool:
        if official_gate_index is None:
            return False
        idx = int(official_gate_index)
        if self.last_official_gate_index is None:
            self.last_official_gate_index = idx
            return False
        if idx <= self.last_official_gate_index:
            return False

        passed_gate_id = self.last_official_gate_index
        self.last_official_gate_index = idx
        self.advance_count += 1
        self.retarget_count += 1

        search_yaw_hint = None
        landmark = estimator.landmarks.get(passed_gate_id)
        if landmark is not None and estimator.state.calibrated:
            px, py, _pz = estimator.state.position_ned_m
            gx, gy, _gz = landmark
            dx, dy = px - gx, py - gy
            if math.hypot(dx, dy) > 0.5:
                search_yaw_hint = math.atan2(dy, dx)

        angle_servo.reset_for_gate_retarget(yaw_cmd_rad=search_yaw_hint)
        map_guidance.reset_carry()
        pass_tracker.sync_official_advance(idx)

        self.events.append(
            {
                "elapsed_s": round(elapsed_s, 3),
                "from_index": passed_gate_id,
                "to_index": idx,
                "search_yaw_hint_rad": (
                    None if search_yaw_hint is None else round(search_yaw_hint, 6)
                ),
            }
        )
        return True

    def to_summary(self) -> dict:
        return {
            "last_official_gate_index": self.last_official_gate_index,
            "advance_count": self.advance_count,
            "retarget_count": self.retarget_count,
            "events": list(self.events),
        }


@dataclasses.dataclass
class GroundedLimpState:
    """Detect when recovery cannot succeed and the airframe is down."""

    recovery_active_since_s: float | None = None
    last_collision_count: int = 0
    limp_events: int = 0
    active: bool = False
    reason: str | None = None
    elapsed_s_at_trigger: float | None = None

    def trigger(self, reason: str, elapsed_s: float) -> bool:
        """Activate the abort latch once and report whether it changed."""
        if self.active:
            return False
        self._trigger(reason, elapsed_s)
        return True

    def observe(
        self,
        *,
        now_s: float,
        elapsed_s: float,
        recovery_active: bool,
        collision_count: int,
        velocity_ned_m_s: tuple[float, float, float],
        recovery_timeout_s: float,
        max_speed_m_s: float,
        collision_grace_s: float,
    ) -> bool:
        """Returns True the first time a limp/grounded abort is triggered."""
        if self.active:
            return False

        vx, vy, vz = velocity_ned_m_s
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        if recovery_active:
            if self.recovery_active_since_s is None:
                self.recovery_active_since_s = now_s
            elif (
                now_s - self.recovery_active_since_s >= recovery_timeout_s
                and speed <= max_speed_m_s
            ):
                self._trigger("recovery_timeout_near_zero_velocity", elapsed_s)
                return True
        else:
            self.recovery_active_since_s = None

        if collision_count > self.last_collision_count:
            self.last_collision_count = collision_count
            self.recovery_active_since_s = now_s
        elif (
            self.recovery_active_since_s is not None
            and collision_count > 0
            and now_s - self.recovery_active_since_s >= collision_grace_s
            and speed <= max_speed_m_s
        ):
            self._trigger("post_collision_near_zero_velocity", elapsed_s)
            return True

        return False

    def _trigger(self, reason: str, elapsed_s: float) -> None:
        self.active = True
        self.reason = reason
        self.limp_events += 1
        self.elapsed_s_at_trigger = round(elapsed_s, 3)

    def to_summary(self) -> dict:
        return {
            "active": self.active,
            "limp_events": self.limp_events,
            "reason": self.reason,
            "elapsed_s_at_trigger": self.elapsed_s_at_trigger,
        }


@dataclasses.dataclass(frozen=True)
class GatePassEvent:
    pass_index: int
    event_time_s: float
    range_camera_m: float
    confidence: float


@dataclasses.dataclass
class GatePassConfig:
    confidence_arm_min: float = 0.35
    confidence_pass_min: float = 0.50
    arm_range_m: float = 3.0
    pass_range_m: float = 1.2
    rearm_range_m: float = 1.6
    min_consecutive_pass_frames: int = 2
    max_missed_detection_frames: int = 3
    pass_cooldown_s: float = 0.30


@dataclasses.dataclass
class VisionGatePassTracker:
    """Robust ordered-gate pass tracker from vision detections."""

    config: GatePassConfig = dataclasses.field(default_factory=GatePassConfig)
    target_gate_count: int = 1
    armed: bool = False
    awaiting_rearm: bool = False
    pass_count: int = 0
    completion_time_s: float | None = None
    events: list[GatePassEvent] = dataclasses.field(default_factory=list)
    last_seen_time_s: float | None = None
    last_pass_time_s: float | None = None
    consecutive_pass_frames: int = 0
    missed_detection_frames: int = 0
    rejection_counts: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def _reject(self, reason: str) -> None:
        self.rejection_counts[reason] += 1

    def _reset_debounce(self) -> None:
        self.consecutive_pass_frames = 0
        self.missed_detection_frames = 0

    def observe(
        self,
        *,
        now_s: float,
        gate_pose,
        detection_confidence: float | None = None,
        detection_present: bool,
    ) -> bool:
        """Returns True when a new pass event is emitted."""
        if not detection_present or gate_pose is None:
            self._reject("missing_detection")
            if self.consecutive_pass_frames > 0:
                self.missed_detection_frames += 1
                if self.missed_detection_frames > self.config.max_missed_detection_frames:
                    self._reject("debounce_reset_missed_frames")
                    self._reset_debounce()
            return False

        detection_confidence = clamp(float(0.0 if detection_confidence is None else detection_confidence), 0.0, 1.0)
        range_m = float(gate_pose.range_camera_m)
        self.last_seen_time_s = now_s

        if (
            self.last_pass_time_s is not None
            and now_s - self.last_pass_time_s < self.config.pass_cooldown_s
        ):
            self._reject("cooldown_active")
            return False

        if detection_confidence < self.config.confidence_arm_min:
            self._reject("below_arm_confidence")
            self._reset_debounce()
            return False

        if self.awaiting_rearm:
            if range_m >= self.config.rearm_range_m:
                self.awaiting_rearm = False
                self._reject("rearmed")
            else:
                self._reject("awaiting_rearm_range")
                return False

        if not self.armed and range_m >= self.config.arm_range_m:
            self.armed = True
            self._reject("armed")
            self._reset_debounce()

        if not self.armed:
            self._reject("not_armed")
            return False

        if detection_confidence < self.config.confidence_pass_min:
            self._reject("below_pass_confidence")
            self._reset_debounce()
            return False

        if range_m > self.config.pass_range_m:
            self._reject("pass_range_not_met")
            self._reset_debounce()
            return False

        if self.consecutive_pass_frames <= 0:
            self.consecutive_pass_frames = 1
        else:
            self.consecutive_pass_frames += 1
        self.missed_detection_frames = 0

        if self.consecutive_pass_frames < self.config.min_consecutive_pass_frames:
            self._reject("debounce_waiting")
            return False

        self.pass_count += 1
        self.last_pass_time_s = now_s
        self.armed = False
        self.awaiting_rearm = True
        self._reset_debounce()
        if self.completion_time_s is None and self.pass_count >= self.target_gate_count:
            self.completion_time_s = now_s
        self.events.append(
            GatePassEvent(
                pass_index=self.pass_count,
                event_time_s=now_s,
                range_camera_m=range_m,
                confidence=detection_confidence,
            )
        )
        return True

    def sync_official_advance(self, official_gate_index: int) -> None:
        """Reset vision pass debounce when the sim advances active_gate_index."""
        self.pass_count = max(self.pass_count, int(official_gate_index))
        self.armed = False
        self.awaiting_rearm = False
        self._reset_debounce()

    def to_summary(self, *, started_s: float) -> dict:
        return {
            "pass_count": self.pass_count,
            "target_gate_count": self.target_gate_count,
            "armed": self.armed,
            "awaiting_rearm": self.awaiting_rearm,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "config": dataclasses.asdict(self.config),
            "completion_time_s": (
                None if self.completion_time_s is None else round(self.completion_time_s - started_s, 6)
            ),
            "events": [
                {
                    "pass_index": event.pass_index,
                    "event_time_s": round(event.event_time_s - started_s, 6),
                    "range_camera_m": round(event.range_camera_m, 6),
                    "confidence": round(event.confidence, 6),
                }
                for event in self.events
            ],
        }


@dataclasses.dataclass
class CompetitionSmokeReport:
    sitl: SitlRunReport
    control_mode: str
    policy_source: str
    control_inputs: dict = dataclasses.field(default_factory=dict)
    crash_detected: bool | None = None
    invalid_run: bool | None = None
    completion_time_s: float | None = None
    ordered_gate_passes: int = 0
    ordered_gate_sequence_valid: bool | None = None
    official_active_gate_index: int | None = None
    official_last_gate_race_time: int | None = None
    official_race_finish_time_ns: int | None = None
    gate_pass_summary: dict = dataclasses.field(default_factory=dict)
    acceptance_passed: bool = False
    acceptance_blockers: list[str] = dataclasses.field(default_factory=list)
    acceptance_inputs: dict = dataclasses.field(default_factory=dict)
    acceptance_config_path: str = ""
    vision: SmokeVisionMetrics = dataclasses.field(default_factory=SmokeVisionMetrics)
    approach_diagnostics: ApproachDiagnostics = dataclasses.field(default_factory=ApproachDiagnostics)
    policy_trace: dict = dataclasses.field(default_factory=dict)
    camera_stream_metrics: dict = dataclasses.field(default_factory=dict)
    detector_metrics: dict = dataclasses.field(default_factory=dict)
    debug_frames: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        return dataclasses.asdict(self)


def update_timesync_metrics(metrics: SmokeVisionMetrics, *, sim_time_ns: int, timesync_ts1: int | None) -> None:
    if timesync_ts1 is None:
        return
    error = abs(int(sim_time_ns) - int(timesync_ts1))
    metrics.timesync_samples += 1
    n = metrics.timesync_samples
    metrics.average_timesync_error_ns += (error - metrics.average_timesync_error_ns) / n
    metrics.max_timesync_error_ns = max(metrics.max_timesync_error_ns, error)


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def fixed_policy_steps_due(
    *,
    elapsed_s: float,
    state_hz: float,
    completed_steps: int,
) -> int:
    """Return recurrent state steps due on a wall-clock-normalized schedule.

    The MAVLink publisher already runs independently at a fixed rate, but the
    recurrent policy historically advanced once per jittery main-loop update.
    That made identical checkpoints evolve at different effective time scales
    from run to run.  A positive ``state_hz`` schedules an immediate tick at
    time zero and exactly ``state_hz`` further ticks per elapsed second.  Zero
    retains the historical one-step-per-update contract.
    """

    hz = float(state_hz)
    if not math.isfinite(hz) or hz < 0.0:
        raise ValueError("policy state_hz must be finite and nonnegative")
    completed = int(completed_steps)
    if completed < 0:
        raise ValueError("completed policy steps must be nonnegative")
    if hz == 0.0:
        return 1
    elapsed = max(0.0, float(elapsed_s))
    # The epsilon prevents binary rounding at an exact schedule boundary from
    # delaying the corresponding recurrent tick to the next outer-loop pass.
    target_steps = int(math.floor(elapsed * hz + 1e-9)) + 1
    return max(0, target_steps - completed)


def policy_observation_with_last_command(
    observation: Sequence[float], last_cmd_norm: Sequence[float]
) -> tuple[float, ...]:
    """Patch TS-002 previous-action fields for an in-update catch-up tick."""

    if len(observation) < 23:
        raise ValueError("policy observation must contain TS-002 base fields")
    if len(last_cmd_norm) != 4:
        raise ValueError("last policy command must contain four values")
    values = [float(value) for value in observation]
    values[19:23] = [clamp(float(value), -1.0, 1.0) for value in last_cmd_norm]
    return tuple(values)


def append_policy_trace_sample(
    samples: list[dict],
    *,
    max_samples: int,
    now_s: float,
    started_s: float,
    official_gate_index: int | None,
    observation: Sequence[float],
    action_values: Sequence[float],
    decoded_command,
    gate_pose,
    motion_filter: ObservableGateMotionFilter,
    gate_detection=None,
) -> None:
    """Save a compact, replayable official-observable policy sample."""
    if len(samples) >= max_samples:
        return
    pose = None
    if gate_pose is not None:
        pose = {
            "body_vector_ned_m": [
                round(float(value), 6) for value in gate_pose.body_vector_ned_m
            ],
            "range_camera_m": round(float(gate_pose.range_camera_m), 6),
            "yaw_error_rad": round(float(gate_pose.yaw_error_rad), 6),
            "confidence": round(float(gate_pose.confidence), 6),
            "detection_source": (
                None
                if gate_detection is None
                else str(getattr(gate_detection, "source", "frame"))
            ),
        }
    samples.append(
        {
            "elapsed_s": round(now_s - started_s, 6),
            "official_active_gate_index": official_gate_index,
            "observation": [round(float(value), 7) for value in observation],
            "normalized_action": [round(float(value), 7) for value in action_values],
            "decoded_command": {
                key: round(float(value), 7)
                for key, value in dataclasses.asdict(decoded_command).items()
            },
            "gate_pose": pose,
            "gate_motion": motion_filter.snapshot(),
        }
    )


def setpoint_to_dict(target: LocalNedSetpoint | None) -> dict:
    if target is None:
        return {}
    return {
        "vx": round_float(target.vx),
        "vy": round_float(target.vy),
        "vz": round_float(target.vz),
        "yaw": round_float(target.yaw),
        "yaw_rate": round_float(target.yaw_rate),
    }


def attitude_setpoint_to_dict(target: AttitudeSetpoint | None) -> dict:
    if target is None:
        return {}
    return {
        "roll": round_float(target.roll),
        "pitch": round_float(target.pitch),
        "yaw": round_float(target.yaw),
        "body_roll_rate": round_float(target.body_roll_rate),
        "body_pitch_rate": round_float(target.body_pitch_rate),
        "body_yaw_rate": round_float(target.body_yaw_rate),
        "thrust": round_float(target.thrust),
    }


def pose_to_dict(pose) -> dict:
    if pose is None:
        return {}
    return {
        "image_center_px": [round_float(value, 3) for value in pose.image_center_px],
        "image_width_px": round_float(pose.image_width_px, 3),
        "image_height_px": round_float(pose.image_height_px, 3),
        "range_camera_m": round_float(pose.range_camera_m),
        "body_vector_ned_m": [round_float(value) for value in pose.body_vector_ned_m],
        "yaw_error_rad": round_float(pose.yaw_error_rad),
        "confidence": round_float(pose.confidence),
    }


def annotate_jpeg(jpeg: bytes, detection: GateDetection | None, path: Path) -> bool:
    if cv2 is None or np is None:
        return False
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if detection is not None:
        points = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"conf={detection.confidence:.3f}"
        cv2.putText(
            image,
            label,
            (max(0, int(points[:, 0, 0].min())), max(20, int(points[:, 0, 1].min()) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def save_debug_frame(output_dir: Path, frame, detection: GateDetection | None, label: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_path = output_dir / f"{label}.jpg"
    annotated_path = output_dir / f"{label}_annotated.jpg"
    jpeg_path.write_bytes(frame.jpeg)
    annotated_written = annotate_jpeg(frame.jpeg, detection, annotated_path)
    return {
        "label": label,
        "frame_id": int(frame.frame_id),
        "sim_time_ns": int(frame.sim_time_ns),
        "jpeg_path": str(jpeg_path),
        "annotated_path": str(annotated_path) if annotated_written else "",
        "detection": dataclasses.asdict(detection) if detection is not None else None,
    }


def defer_debug_frame(
    candidates: dict[str, tuple[float, object, GateDetection | None]],
    *,
    frame,
    detection: GateDetection | None,
    range_m: float,
) -> None:
    """Retain at most three JPEGs without performing flight-loop disk I/O."""
    entry = (float(range_m), frame, detection)
    if "first_detection" not in candidates:
        candidates["first_detection"] = entry
    closest = candidates.get("closest_detection")
    if closest is None or float(range_m) < closest[0]:
        candidates["closest_detection"] = entry
    candidates["latest_detection"] = entry


def flush_debug_frames(
    output_dir: Path | None,
    candidates: dict[str, tuple[float, object, GateDetection | None]],
) -> dict:
    """Write retained diagnostics after the fixed-rate publisher has stopped."""
    if output_dir is None:
        return {}
    saved = {}
    for label in ("first_detection", "closest_detection", "latest_detection"):
        candidate = candidates.get(label)
        if candidate is None:
            continue
        _range_m, frame, detection = candidate
        try:
            saved[label] = save_debug_frame(output_dir, frame, detection, label)
        except Exception as exc:  # Debug evidence must never invalidate a race report.
            saved[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return saved


def update_approach_diagnostics(
    diagnostics: ApproachDiagnostics,
    *,
    elapsed_s: float,
    sim_time_ns: int,
    gate_pose,
    detection: GateDetection,
    visual_servo_target: LocalNedSetpoint,
    actual_command: dict | None = None,
    max_samples: int,
) -> None:
    confidence = float(detection.confidence)
    range_m = float(gate_pose.range_camera_m)
    pose_dict = pose_to_dict(gate_pose)
    command_dict = setpoint_to_dict(visual_servo_target)
    actual_command_dict = dict(actual_command or command_dict)
    sample = {
        "elapsed_s": round_float(elapsed_s),
        "sim_time_ns": int(sim_time_ns),
        "confidence": round_float(confidence),
        "pose": pose_dict,
        "visual_servo_command": command_dict,
        "actual_command": actual_command_dict,
    }
    diagnostics.detections_sampled += 1
    diagnostics.latest_pose = pose_dict
    diagnostics.latest_command = actual_command_dict

    if diagnostics.closest_range_m is None or range_m < diagnostics.closest_range_m:
        diagnostics.closest_range_m = round_float(range_m)
        diagnostics.closest_range_elapsed_s = round_float(elapsed_s)
        diagnostics.closest_range_confidence = round_float(confidence)
        diagnostics.closest_range_body_vector_ned_m = tuple(
            round_float(value) for value in gate_pose.body_vector_ned_m
        )
        diagnostics.closest_range_yaw_error_rad = round_float(gate_pose.yaw_error_rad)
        diagnostics.closest_range_command = actual_command_dict

    if diagnostics.highest_confidence is None or confidence > diagnostics.highest_confidence:
        diagnostics.highest_confidence = round_float(confidence)
        diagnostics.highest_confidence_elapsed_s = round_float(elapsed_s)

    if max_samples <= 0:
        return
    if len(diagnostics.samples) < max_samples:
        diagnostics.samples.append(sample)
    else:
        # Keep deterministic coverage across the run without storing every frame.
        replace_idx = diagnostics.detections_sampled % max_samples
        diagnostics.samples[replace_idx] = sample


def visual_servo_command_from_args(pose, *, yaw_rad: float, args) -> object:
    return visual_servo_command(
        pose,
        yaw_rad=yaw_rad,
        desired_standoff_m=float(getattr(args, "visual_servo_desired_standoff_m", 1.0)),
        max_forward_m_s=float(getattr(args, "visual_servo_max_forward_m_s", 1.0)),
        max_lateral_m_s=float(getattr(args, "visual_servo_max_lateral_m_s", 0.5)),
        max_vertical_m_s=float(getattr(args, "visual_servo_max_vertical_m_s", 0.4)),
        max_yaw_rate_rad_s=float(getattr(args, "visual_servo_max_yaw_rate_rad_s", 0.6)),
        k_forward=float(getattr(args, "visual_servo_k_forward", 0.45)),
        k_lateral=float(getattr(args, "visual_servo_k_lateral", 0.7)),
        k_vertical=float(getattr(args, "visual_servo_k_vertical", 0.7)),
        k_yaw=float(getattr(args, "visual_servo_k_yaw", 1.2)),
    )


def attitude_setpoint_from_args(args) -> AttitudeSetpoint:
    return AttitudeSetpoint(
        roll=float(getattr(args, "attitude_roll_rad", 0.0)),
        pitch=float(getattr(args, "attitude_pitch_rad", 0.0)),
        yaw=float(getattr(args, "attitude_yaw_rad", 0.0)),
        body_roll_rate=float(getattr(args, "body_roll_rate_rad_s", 0.0)),
        body_pitch_rate=float(getattr(args, "body_pitch_rate_rad_s", 0.0)),
        body_yaw_rate=float(getattr(args, "body_yaw_rate_rad_s", 0.0)),
        thrust=float(getattr(args, "attitude_thrust", 0.5)),
    )


def attitude_servo_command_from_args(pose, args) -> AttitudeSetpoint:
    bx, by, bz = pose.body_vector_ned_m
    desired_standoff_m = float(getattr(args, "attitude_servo_desired_standoff_m", 0.0))
    max_pitch_rate = abs(float(getattr(args, "attitude_servo_max_pitch_rate_rad_s", 0.5)))
    max_roll_rate = abs(float(getattr(args, "attitude_servo_max_roll_rate_rad_s", 0.4)))
    max_yaw_rate = abs(float(getattr(args, "attitude_servo_max_yaw_rate_rad_s", 0.7)))
    hover_thrust = float(getattr(args, "attitude_servo_hover_thrust", 0.58))
    min_thrust = float(getattr(args, "attitude_servo_min_thrust", 0.35))
    max_thrust = float(getattr(args, "attitude_servo_max_thrust", 0.75))
    k_pitch = float(getattr(args, "attitude_servo_k_pitch", 0.16))
    k_roll = float(getattr(args, "attitude_servo_k_roll", 0.0))
    k_yaw = float(getattr(args, "attitude_servo_k_yaw", 1.2))
    k_thrust = float(getattr(args, "attitude_servo_k_thrust", 0.08))
    k_image_roll = float(getattr(args, "attitude_servo_k_image_roll", 0.0))
    forward_yaw_tolerance = getattr(args, "attitude_servo_forward_yaw_tolerance_rad", None)
    forward_z_tolerance = getattr(args, "attitude_servo_forward_z_tolerance_m", None)
    uncentered_forward_scale = float(getattr(args, "attitude_servo_uncentered_forward_scale", 1.0))

    if desired_standoff_m < 0.0:
        raise ValueError("attitude_servo_desired_standoff_m must be non-negative")
    if min_thrust > max_thrust:
        raise ValueError("attitude_servo_min_thrust must be <= attitude_servo_max_thrust")
    if uncentered_forward_scale < 0.0:
        raise ValueError("attitude_servo_uncentered_forward_scale must be non-negative")

    forward_error = max(0.0, bx - desired_standoff_m)
    pitch_rate = -clamp(k_pitch * forward_error, 0.0, max_pitch_rate)
    image_roll_error = 0.0
    if pose is not None and hasattr(pose, "image_width_px") and hasattr(pose, "image_center_px"):
        image_width_m = float(pose.image_width_px)
        if image_width_m < float(CAMERA_WIDTH) * 0.25:
            image_width_m = float(CAMERA_WIDTH)
        if image_width_m > 1.0:
            image_roll_error = (float(pose.image_center_px[0]) - image_width_m / 2.0) / (image_width_m / 2.0)
    roll_rate = clamp(
        k_roll * by + k_image_roll * image_roll_error,
        -max_roll_rate,
        max_roll_rate,
    )
    yaw_rate = clamp(k_yaw * pose.yaw_error_rad, -max_yaw_rate, max_yaw_rate)
    thrust = clamp(hover_thrust - k_thrust * bz, min_thrust, max_thrust)
    centered = True
    if forward_yaw_tolerance is not None and float(forward_yaw_tolerance) >= 0.0:
        centered = centered and abs(float(pose.yaw_error_rad)) <= float(forward_yaw_tolerance)
    if forward_z_tolerance is not None and float(forward_z_tolerance) >= 0.0:
        centered = centered and abs(float(bz)) <= float(forward_z_tolerance)
    if not centered:
        pitch_rate *= uncentered_forward_scale
    return AttitudeSetpoint(
        body_roll_rate=roll_rate,
        body_pitch_rate=pitch_rate,
        body_yaw_rate=yaw_rate,
        thrust=thrust,
    )


def attitude_search_command_from_args(args) -> AttitudeSetpoint:
    hover_thrust = float(getattr(args, "attitude_servo_hover_thrust", 0.58))
    search_thrust = getattr(args, "attitude_servo_search_thrust", None)
    return AttitudeSetpoint(
        body_pitch_rate=float(getattr(args, "attitude_servo_search_pitch_rate_rad_s", 0.0)),
        body_yaw_rate=float(getattr(args, "attitude_servo_search_yaw_rate_rad_s", 0.0)),
        thrust=hover_thrust if search_thrust is None else float(search_thrust),
    )


def attitude_control_pose_allowed(pose, detection: GateDetection | None, args) -> bool:
    min_size_px = float(getattr(args, "attitude_servo_min_control_size_px", 0.0))
    max_range_m = float(getattr(args, "attitude_servo_max_control_range_m", 0.0))
    min_confidence = float(getattr(args, "attitude_servo_min_control_confidence", 0.0))
    if min_size_px < 0.0:
        raise ValueError("attitude_servo_min_control_size_px must be non-negative")
    if max_range_m < 0.0:
        raise ValueError("attitude_servo_max_control_range_m must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("attitude_servo_min_control_confidence must be in [0, 1]")
    if pose is None:
        return False
    if min(pose.image_width_px, pose.image_height_px) < min_size_px:
        return False
    if max_range_m > 0.0 and pose.range_camera_m > max_range_m:
        return False
    if detection is not None and float(detection.confidence) < min_confidence:
        return False
    return True


def should_stop_before_policy_gate(
    *,
    control_mode: str,
    official_gate_index: int | None,
    pose,
    stop_gate_index: int,
    stop_forward_m: float,
) -> bool:
    """Return whether a policy diagnostic reached its no-crossing boundary."""

    if control_mode != "policy-attitude":
        return False
    if stop_gate_index < 0 or stop_forward_m <= 0.0:
        return False
    if official_gate_index is None or official_gate_index < stop_gate_index:
        return False
    if pose is None:
        return False
    forward_m = float(pose.body_vector_ned_m[0])
    return 0.0 < forward_m <= stop_forward_m


def should_stop_after_official_finish(race_status) -> bool:
    """Return whether the simulator has published an authoritative finish."""

    return bool(
        race_status is not None
        and int(getattr(race_status, "race_finish_time_ns", -1)) >= 0
    )


def should_stop_after_official_gate_index(
    official_gate_index: int | None,
    stop_after_gate_index: int,
) -> bool:
    """Stop immediately after an authoritative bounded milestone is reached."""

    return bool(
        int(stop_after_gate_index) >= 0
        and official_gate_index is not None
        and int(official_gate_index) >= int(stop_after_gate_index)
    )


def should_start_attitude_final_approach(
    pose,
    detection: GateDetection | None,
    args,
    state: AttitudeFinalApproachState,
    *,
    now_s: float,
) -> bool:
    if not bool(getattr(args, "attitude_servo_final_approach", False)):
        return False
    if pose is None or detection is None:
        return False
    if state.is_active(now_s):
        return False

    max_activations = int(getattr(args, "attitude_servo_final_max_activations", 1))
    duration_s = float(getattr(args, "attitude_servo_final_duration_s", 1.0))
    trigger_range_m = float(getattr(args, "attitude_servo_final_trigger_range_m", 6.5))
    min_size_px = float(getattr(args, "attitude_servo_final_min_size_px", 40.0))
    min_confidence = float(getattr(args, "attitude_servo_final_min_confidence", 0.45))
    max_yaw_error_rad = float(getattr(args, "attitude_servo_final_max_yaw_error_rad", 0.25))
    max_abs_z_m = getattr(args, "attitude_servo_final_max_abs_z_m", 1.0)

    if max_activations < 0:
        raise ValueError("attitude_servo_final_max_activations must be non-negative")
    if duration_s <= 0.0:
        return False
    if trigger_range_m <= 0.0:
        raise ValueError("attitude_servo_final_trigger_range_m must be positive")
    if min_size_px < 0.0:
        raise ValueError("attitude_servo_final_min_size_px must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("attitude_servo_final_min_confidence must be in [0, 1]")
    if max_yaw_error_rad < 0.0:
        raise ValueError("attitude_servo_final_max_yaw_error_rad must be non-negative")
    if max_abs_z_m is not None and float(max_abs_z_m) < 0.0:
        raise ValueError("attitude_servo_final_max_abs_z_m must be non-negative")

    if max_activations > 0 and state.activations >= max_activations:
        return False
    if pose.range_camera_m > trigger_range_m:
        return False
    if min(pose.image_width_px, pose.image_height_px) < min_size_px:
        return False
    if float(detection.confidence) < min_confidence:
        return False
    if abs(float(pose.yaw_error_rad)) > max_yaw_error_rad:
        return False
    if max_abs_z_m is not None:
        _bx, _by, bz = pose.body_vector_ned_m
        if abs(float(bz)) > float(max_abs_z_m):
            return False
    return True


def attitude_final_approach_command_from_args(pose, args) -> AttitudeSetpoint:
    min_thrust = float(getattr(args, "attitude_servo_min_thrust", 0.35))
    max_thrust = float(getattr(args, "attitude_servo_max_thrust", 0.75))
    max_roll_rate = abs(float(getattr(args, "attitude_servo_max_roll_rate_rad_s", 0.4)))
    if min_thrust > max_thrust:
        raise ValueError("attitude_servo_min_thrust must be <= attitude_servo_max_thrust")

    pitch_rate = -abs(float(getattr(args, "attitude_servo_final_pitch_rate_rad_s", 0.2)))
    max_yaw_rate = abs(float(getattr(args, "attitude_servo_final_max_yaw_rate_rad_s", 0.35)))
    k_yaw = float(getattr(args, "attitude_servo_k_yaw", 1.2))
    k_image_roll = float(getattr(args, "attitude_servo_k_image_roll", 0.0))
    if pose is None:
        roll_rate = 0.0
        yaw_rate = 0.0
    else:
        image_roll_error = 0.0
        if hasattr(pose, "image_width_px") and hasattr(pose, "image_center_px"):
            image_width_m = float(pose.image_width_px)
            if image_width_m < float(CAMERA_WIDTH) * 0.25:
                image_width_m = float(CAMERA_WIDTH)
            if image_width_m > 1.0:
                image_roll_error = (float(pose.image_center_px[0]) - image_width_m / 2.0) / (image_width_m / 2.0)
        roll_rate = clamp(
            k_image_roll * image_roll_error,
            -max_roll_rate,
            max_roll_rate,
        )
        yaw_rate = clamp(k_yaw * pose.yaw_error_rad, -max_yaw_rate, max_yaw_rate)
    thrust = clamp(float(getattr(args, "attitude_servo_final_thrust", 0.62)), min_thrust, max_thrust)
    return AttitudeSetpoint(
        body_pitch_rate=pitch_rate,
        body_yaw_rate=yaw_rate,
        body_roll_rate=roll_rate,
        thrust=thrust,
    )


def maybe_write_csv(path: str, report: CompetitionSmokeReport) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    row = {
        "mode": report.control_mode,
        "policy_source": report.policy_source,
        "command_kind": report.sitl.command_kind,
        "duration_s": report.sitl.duration_s,
        "heartbeats_sent": report.sitl.heartbeats_sent,
        "commands_sent": report.sitl.commands_sent,
        "effective_command_hz": report.sitl.effective_command_hz,
        "command_rate_violations": report.sitl.command_rate_violations,
        "telemetry_messages_seen": report.sitl.telemetry.messages_seen,
        "telemetry_dropouts": report.sitl.telemetry.telemetry_dropouts,
        "min_telemetry_messages": report.acceptance_inputs.get("min_telemetry_messages"),
        "vision_frames_seen": report.vision.frames_seen,
        "min_camera_frames": report.acceptance_inputs.get("min_camera_frames"),
        "vision_detector_detections": report.vision.detector_detections,
        "vision_decode_failures": report.vision.detector_decode_failures,
        "vision_no_quad_found": report.vision.detector_no_quad_found,
        "vision_avg_timesync_error_ns": round(report.vision.average_timesync_error_ns, 2),
        "vision_max_timesync_error_ns": report.vision.max_timesync_error_ns,
        "ordered_gate_passes": report.ordered_gate_passes,
        "ordered_gate_sequence_valid": report.ordered_gate_sequence_valid,
        "official_active_gate_index": report.official_active_gate_index,
        "official_last_gate_race_time": report.official_last_gate_race_time,
        "official_race_finish_time_ns": report.official_race_finish_time_ns,
        "completion_time_s": report.completion_time_s,
        "target_gate_count": report.gate_pass_summary.get("target_gate_count"),
        "crash_detected": report.crash_detected,
        "invalid_run": report.invalid_run,
        "acceptance_passed": report.acceptance_passed,
        "acceptance_blockers": ";".join(report.acceptance_blockers),
        "acceptance_config_path": report.acceptance_config_path,
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run_smoke(args) -> CompetitionSmokeReport:
    validate_rates(args.heartbeat_hz, args.command_hz)
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.idle_sleep_s < 0.0:
        raise ValueError("idle_sleep_s must be non-negative")
    arm_on_start = bool(getattr(args, "arm_on_start", True))
    policy_shadow_only = bool(getattr(args, "policy_shadow_only", False))
    arm_attempts = int(getattr(args, "arm_attempts", 3))
    prearm_heartbeat_timeout_s = float(getattr(args, "prearm_heartbeat_timeout_s", 2.0))
    policy_shadow_calibration_timeout_s = float(
        getattr(args, "policy_shadow_calibration_timeout_s", 6.0)
    )
    if arm_attempts < 0:
        raise ValueError("arm_attempts must be non-negative")
    if prearm_heartbeat_timeout_s < 0.0:
        raise ValueError("prearm_heartbeat_timeout_s must be non-negative")
    if policy_shadow_calibration_timeout_s <= 0.0:
        raise ValueError("policy_shadow_calibration_timeout_s must be positive")
    if args.target_gate_count <= 0:
        raise ValueError("target_gate_count must be positive")
    max_approach_diagnostic_samples = int(getattr(args, "max_approach_diagnostic_samples", 12))
    if max_approach_diagnostic_samples < 0:
        raise ValueError("max_approach_diagnostic_samples must be non-negative")

    acceptance_config = load_acceptance_config(args.acceptance_config)
    gate_pass_config = GatePassConfig(
        confidence_arm_min=float(
            args.gate_confidence_arm_min
            if args.gate_confidence_arm_min is not None
            else resolve_config_value(acceptance_config, "gate_pass", "confidence_arm_min", 0.35)
        ),
        confidence_pass_min=float(
            args.gate_confidence_pass_min
            if args.gate_confidence_pass_min is not None
            else resolve_config_value(acceptance_config, "gate_pass", "confidence_pass_min", 0.5)
        ),
        arm_range_m=float(
            args.gate_pass_arm_range_m
            if args.gate_pass_arm_range_m is not None
            else resolve_config_value(acceptance_config, "gate_pass", "arm_range_m", 3.0)
        ),
        pass_range_m=float(
            args.gate_pass_range_m
            if args.gate_pass_range_m is not None
            else resolve_config_value(acceptance_config, "gate_pass", "pass_range_m", 1.2)
        ),
        rearm_range_m=float(
            args.gate_pass_rearm_range_m
            if args.gate_pass_rearm_range_m is not None
            else resolve_config_value(acceptance_config, "gate_pass", "rearm_range_m", 1.6)
        ),
        min_consecutive_pass_frames=int(
            args.gate_pass_min_consecutive_frames
            if args.gate_pass_min_consecutive_frames is not None
            else resolve_config_value(acceptance_config, "gate_pass", "min_consecutive_pass_frames", 2)
        ),
        max_missed_detection_frames=int(
            args.gate_pass_max_missed_frames
            if args.gate_pass_max_missed_frames is not None
            else resolve_config_value(acceptance_config, "gate_pass", "max_missed_detection_frames", 3)
        ),
        pass_cooldown_s=float(
            args.gate_pass_cooldown_s
            if args.gate_pass_cooldown_s is not None
            else resolve_config_value(acceptance_config, "gate_pass", "pass_cooldown_s", 0.3)
        ),
    )
    if gate_pass_config.arm_range_m <= 0.0:
        raise ValueError("gate_pass arm_range_m must be positive")
    if gate_pass_config.pass_range_m <= 0.0:
        raise ValueError("gate_pass pass_range_m must be positive")
    if gate_pass_config.rearm_range_m <= gate_pass_config.pass_range_m:
        raise ValueError("gate_pass rearm_range_m must be greater than pass_range_m")
    if gate_pass_config.arm_range_m <= gate_pass_config.pass_range_m:
        raise ValueError("gate_pass arm_range_m must be greater than pass_range_m")
    if gate_pass_config.min_consecutive_pass_frames <= 0:
        raise ValueError("gate_pass min_consecutive_pass_frames must be positive")
    if gate_pass_config.max_missed_detection_frames < 0:
        raise ValueError("gate_pass max_missed_detection_frames must be non-negative")
    if gate_pass_config.pass_cooldown_s <= 0.0:
        raise ValueError("gate_pass pass_cooldown_s must be positive")

    require_telemetry = bool(args.require_telemetry) or bool(
        resolve_config_value(acceptance_config, "smoke_defaults", "require_telemetry", False)
    )
    require_camera = bool(args.require_camera) or bool(
        resolve_config_value(acceptance_config, "smoke_defaults", "require_camera", False)
    )
    min_gate_passes = int(
        args.min_gate_passes
        if args.min_gate_passes is not None
        else resolve_config_value(acceptance_config, "smoke_defaults", "min_gate_passes", 1)
    )
    if min_gate_passes <= 0:
        raise ValueError("min_gate_passes must be positive")
    max_command_rate_violations = int(
        args.max_command_rate_violations
        if args.max_command_rate_violations is not None
        else resolve_config_value(acceptance_config, "health_thresholds", "max_command_rate_violations", 0)
    )
    min_telemetry_messages = int(
        args.min_telemetry_messages
        if args.min_telemetry_messages is not None
        else resolve_config_value(acceptance_config, "health_thresholds", "min_telemetry_messages", 1)
    )
    min_camera_frames = int(
        args.min_camera_frames
        if args.min_camera_frames is not None
        else resolve_config_value(acceptance_config, "health_thresholds", "min_camera_frames", 1)
    )
    max_telemetry_dropouts = int(
        args.max_telemetry_dropouts
        if args.max_telemetry_dropouts is not None
            else resolve_config_value(acceptance_config, "health_thresholds", "max_telemetry_dropouts", 2)
    )
    min_effective_command_hz = float(
        resolve_config_value(
            acceptance_config,
            "health_thresholds",
            "min_effective_command_hz",
            0.0,
        )
    )
    max_telemetry_drain_limit_hits = int(
        resolve_config_value(
            acceptance_config,
            "health_thresholds",
            "max_telemetry_drain_limit_hits",
            0,
        )
    )
    require_official_race_progress = bool(args.require_official_race_progress) or bool(
        resolve_config_value(acceptance_config, "smoke_defaults", "require_official_race_progress", False)
    )

    control_mode = getattr(args, "control_mode", "visual-servo")
    if control_mode not in {
        "visual-servo",
        "policy",
        "policy-attitude",
        "attitude-rates",
        "visual-servo-attitude",
        "visual-servo-angles",
        "course-fsm",
    }:
        raise ValueError(
            "control_mode must be 'visual-servo', 'policy', 'policy-attitude', "
            "'attitude-rates', 'visual-servo-attitude', 'visual-servo-angles', "
            "or 'course-fsm'"
        )
    fixed_rate_attitude_mode = control_mode in {"course-fsm", "policy-attitude"}
    official_reset_on_start = bool(getattr(args, "official_reset_on_start", False))
    official_reset_start_timeout_s = float(
        getattr(args, "official_reset_start_timeout_s", 12.0)
    )
    official_policy_lead_s = float(getattr(args, "official_policy_lead_s", 0.05))
    if official_reset_on_start and control_mode != "policy-attitude":
        raise ValueError(
            "official_reset_on_start is reserved for the full policy-attitude runner"
        )
    if policy_shadow_only:
        if control_mode != "policy-attitude":
            raise ValueError("policy_shadow_only requires control_mode=policy-attitude")
        if arm_on_start:
            raise ValueError("policy_shadow_only requires --no-arm-on-start")
        if official_reset_on_start:
            raise ValueError("policy_shadow_only forbids official_reset_on_start")
        if not getattr(args, "policy_callable", ""):
            raise ValueError("policy_shadow_only requires a policy callable")
        # A passive shadow run proves transport, observation, and inference
        # behavior. It must neither require gate progress nor pretend to meet
        # the fixed-rate actuation threshold.
        min_gate_passes = 0
        min_effective_command_hz = 0.0

    policy_source = "visual_servo"
    policy_callable = None
    normalized_action = (0.0, 0.0, 0.0, 0.0)
    command_frame = getattr(args, "command_frame", "local_ned")
    if command_frame not in {"body_ned", "local_ned"}:
        raise ValueError("command_frame must be 'body_ned' or 'local_ned'")
    command_yaw_mode = getattr(args, "command_yaw_mode", "yaw_and_rate")
    if command_yaw_mode not in {"yaw_and_rate", "ignore"}:
        raise ValueError("command_yaw_mode must be 'yaw_and_rate' or 'ignore'")
    attitude_mode = getattr(args, "attitude_mode", "body_rates")
    if attitude_mode not in {"body_rates", "attitude", "attitude_and_rates"}:
        raise ValueError("attitude_mode must be 'body_rates', 'attitude', or 'attitude_and_rates'")
    attitude_target = attitude_setpoint_from_args(args)
    if control_mode == "policy":
        policy_source = "constant_action"
        normalized_action = maybe_parse_action_json(args.policy_action_json)
        if args.policy_callable:
            policy_callable = load_policy_callable(args.policy_callable)
            policy_source = args.policy_callable
            reset_policy_callable(policy_callable)
    elif control_mode == "policy-attitude":
        policy_source = "constant_attitude_action"
        normalized_action = maybe_parse_action_json(args.policy_action_json)
        if args.policy_callable:
            policy_callable = load_policy_callable(args.policy_callable)
            policy_source = args.policy_callable
            reset_policy_callable(policy_callable)
        attitude_mode = "body_rates"
    elif control_mode == "attitude-rates":
        policy_source = "constant_attitude_rates"
    elif control_mode == "visual-servo-attitude":
        policy_source = "visual_servo_attitude_rates"
    elif control_mode == "visual-servo-angles":
        policy_source = "visual_servo_angles"
        attitude_mode = "attitude"
    elif control_mode == "course-fsm":
        policy_source = "cyan_guidance_gate_fsm"
        if args.policy_callable:
            policy_callable = load_policy_callable(args.policy_callable)
            reset_policy_callable(policy_callable)
            policy_source = f"cyan_guidance_gate_fsm_then:{args.policy_callable}"
        attitude_mode = "body_rates"

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    receiver = None
    detector = None
    guidance_detector = None
    if not args.no_camera:
        receiver_class = LatestFrameCameraReceiver if fixed_rate_attitude_mode else UdpCameraReceiver
        receiver = receiver_class(
            host=args.camera_host,
            port=args.camera_port,
            socket_timeout_s=args.camera_timeout_s,
        )
        detector = SquareGateDetector(
            min_area_px=args.detector_min_area_px,
            max_aspect_error=args.detector_max_aspect_error,
            min_fill_ratio=args.detector_min_fill_ratio,
            allow_grayscale_fallback=not bool(
                getattr(args, "detector_require_color", False)
                or fixed_rate_attitude_mode
            ),
        )
        if fixed_rate_attitude_mode:
            guidance_detector = CyanGuidanceDetector(
                hue_low=int(getattr(args, "course_guidance_hue_low", 75)),
                hue_high=int(getattr(args, "course_guidance_hue_high", 105)),
                saturation_min=int(getattr(args, "course_guidance_saturation_min", 80)),
                value_min=int(getattr(args, "course_guidance_value_min", 80)),
                min_pixels=int(getattr(args, "course_guidance_min_pixels", 80)),
                min_supporting_rows=int(
                    getattr(args, "course_guidance_min_supporting_rows", 8)
                ),
            )

    heartbeat_period_s = 1.0 / args.heartbeat_hz
    command_period_s = 1.0 / args.command_hz
    next_heartbeat_s = 0.0
    next_arm_s = 0.0
    next_command_s = 0.0
    command_rate_violations = 0
    heartbeats_sent = 0
    commands_sent = 0
    disarm_commands_sent = 0
    first_command_sent_s = None
    last_command_sent_s = None
    last_detection_s = None
    last_control_detection_s = None
    gate_detection = None
    gate_pose = None
    control_gate_detection = None
    control_gate_pose = None
    guidance_detection = None
    policy_control_aware_gate_predictor_enabled = (
        os.getenv("PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR", "0") == "1"
    )
    policy_control_aware_gate_predictor_start_index = int(
        os.getenv("PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX", "0")
    )
    if policy_control_aware_gate_predictor_start_index < 0:
        raise ValueError(
            "PUFFER_POLICY_CONTROL_AWARE_GATE_PREDICTOR_START_INDEX must be nonnegative"
        )
    policy_predict_gate_dropout_enabled = (
        os.getenv("PUFFER_POLICY_PREDICT_GATE_DROPOUT", "0") == "1"
    )
    policy_predict_gate_dropout_start_index = int(
        os.getenv("PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX", "0")
    )
    if policy_predict_gate_dropout_start_index < 0:
        raise ValueError(
            "PUFFER_POLICY_PREDICT_GATE_DROPOUT_START_INDEX must be nonnegative"
        )
    policy_gate_motion_state = ObservableGateMotionFilter(
        GateMotionFilterConfig(
            max_innovation_speed_m_s=float(
                os.getenv("PUFFER_POLICY_GATE_ASSOCIATION_MAX_SPEED_M_S", "30.0")
            ),
            predict_without_measurement=gate_dropout_prediction_enabled(
                enabled=policy_predict_gate_dropout_enabled,
                official_gate_index=None,
                start_index=policy_predict_gate_dropout_start_index,
            ),
            control_accel_gain_m_s2_per_tan_roll=(
                control_aware_gate_predictor_gain(
                    enabled=policy_control_aware_gate_predictor_enabled,
                    official_gate_index=None,
                    start_index=policy_control_aware_gate_predictor_start_index,
                )
            ),
            reacquire_consecutive_samples=(
                2
                if os.getenv(
                    "PUFFER_POLICY_BOUNDED_GATE_REACQUISITION", "0"
                )
                == "1"
                else 0
            ),
            reacquire_max_candidate_jump_m=float(
                os.getenv(
                    "PUFFER_POLICY_REACQUIRE_MAX_CANDIDATE_JUMP_M", "3.0"
                )
            ),
            reacquire_closer_margin_m=float(
                os.getenv("PUFFER_POLICY_REACQUIRE_CLOSER_MARGIN_M", "0.5")
            ),
        )
    )
    policy_gate_transition_preseed_index = int(
        os.getenv("PUFFER_POLICY_GATE_TRANSITION_PRESEED_INDEX", "-1")
    )
    policy_gate_transition_preseed_max_age_s = float(
        os.getenv("PUFFER_POLICY_GATE_TRANSITION_PRESEED_MAX_AGE_S", "0.25")
    )
    policy_gate_transition_preseed_min_range_margin_m = float(
        os.getenv("PUFFER_POLICY_GATE_TRANSITION_PRESEED_MIN_RANGE_MARGIN_M", "2.0")
    )
    if (
        not math.isfinite(policy_gate_transition_preseed_max_age_s)
        or policy_gate_transition_preseed_max_age_s < 0.0
        or not math.isfinite(policy_gate_transition_preseed_min_range_margin_m)
        or policy_gate_transition_preseed_min_range_margin_m < 0.0
    ):
        raise ValueError("policy gate-transition preseed bounds must be nonnegative")
    policy_trace_samples: list[dict] = []
    policy_inference_ticks = 0
    policy_state_hz = float(getattr(args, "policy_state_hz", 0.0))
    if not math.isfinite(policy_state_hz) or policy_state_hz < 0.0:
        raise ValueError("policy state Hz must be finite and nonnegative")
    policy_state_schedule = {
        "configured_hz": policy_state_hz,
        "fixed_time_enabled": policy_state_hz > 0.0,
        "outer_updates": 0,
        "zero_step_updates": 0,
        "catchup_updates": 0,
        "max_steps_per_update": 0,
    }
    policy_trace_hz = max(0.0, float(getattr(args, "policy_trace_hz", 10.0)))
    policy_trace_max_samples = max(
        0, int(getattr(args, "policy_trace_max_samples", 1200))
    )
    policy_raw_gate_observation_index = int(
        os.getenv("PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX", "-1")
    )
    policy_raw_gate_observation_end_index = int(
        os.getenv("PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX", "-1")
    )
    if (
        policy_raw_gate_observation_end_index >= 0
        and policy_raw_gate_observation_end_index
        < policy_raw_gate_observation_index
    ):
        raise ValueError(
            "PUFFER_POLICY_RAW_GATE_OBSERVATION_END_INDEX must be at least "
            "PUFFER_POLICY_RAW_GATE_OBSERVATION_INDEX"
        )
    policy_final_prefer_any_edge_frame = os.getenv(
        "PUFFER_POLICY_FINAL_PREFER_ANY_EDGE_FRAME", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    policy_final_any_edge_delay_s = float(
        os.getenv("PUFFER_POLICY_FINAL_ANY_EDGE_DELAY_S", "0")
    )
    if (
        not math.isfinite(policy_final_any_edge_delay_s)
        or policy_final_any_edge_delay_s < 0.0
    ):
        raise ValueError("PUFFER_POLICY_FINAL_ANY_EDGE_DELAY_S must be nonnegative")
    policy_final_edge_phase_started_s: float | None = None
    policy_final_edge_gate_index: int | None = None
    policy_stop_gate_index = int(
        getattr(args, "policy_stop_before_gate_index", -1)
    )
    policy_stop_forward_m = float(getattr(args, "policy_stop_forward_m", 0.0))
    policy_gate_stop = {
        "configured_gate_index": policy_stop_gate_index,
        "configured_forward_m": round_float(policy_stop_forward_m),
        "triggered": False,
        "elapsed_s": None,
        "official_gate_index": None,
        "forward_m": None,
        "right_m": None,
        "down_m": None,
        "yaw_error_rad": None,
    }
    official_finish_stop = {
        "triggered": False,
        "elapsed_s": None,
        "official_gate_index": None,
        "race_finish_time_ns": None,
        "race_finish_time_s": None,
    }
    official_gate_index_stop = {
        "configured_gate_index": int(
            getattr(args, "stop_after_official_gate_index", -1)
        ),
        "triggered": False,
        "elapsed_s": None,
        "official_gate_index": None,
    }
    next_policy_trace_s = 0.0
    last_guidance_detection_s = None
    next_course_vision_s = 0.0
    vision_metrics = SmokeVisionMetrics()
    approach_diagnostics = ApproachDiagnostics()
    final_approach_state = AttitudeFinalApproachState()
    angle_servo_state = AttitudeAngleServoState()
    map_guidance_state = MapGuidanceState()
    angle_recovery_active = False
    angle_mode_counters = {
        "recovery_events": 0,
        "recovery_ticks": 0,
        "map_ticks": 0,
        "visual_ticks": 0,
        "servo_blind_ticks": 0,
        "governor_ticks": 0,
        "retarget_events": 0,
        "limp_events": 0,
        "limp_ticks": 0,
    }
    angle_mode_trace: list[dict] = []
    next_angle_trace_s = 0.0
    last_landmark_obs_s: float | None = None
    official_gate_state = OfficialGateProgressState()
    official_gate_observation_epoch = OfficialGateObservationEpoch()
    limp_state = GroundedLimpState()
    limp_abort_exit = bool(getattr(args, "limp_abort_exit", True))
    limp_auto_reset = bool(getattr(args, "limp_auto_reset", False))
    course_controller = CourseController(load_course_controller_config(args))
    estimator = DeadReckoningEstimator()
    attitude_loop_config = AttitudeLoopConfig()
    angle_servo_closed_loop = bool(getattr(args, "angle_servo_closed_loop", True))
    estimator_calibration: dict = {}
    estimator_trajectory: list[dict] = []
    last_estimator_imu_us = None
    next_trajectory_sample_s = 0.0
    debug_frame_dir = str(getattr(args, "debug_frame_dir", "") or "")
    debug_output_dir = Path(debug_frame_dir) if debug_frame_dir else None
    debug_frames = {}
    deferred_debug_frames = {}
    last_cmd_norm = (0.0, 0.0, 0.0, 0.0)
    course_policy_activated_s: float | None = None
    course_policy_ticks = 0
    last_sent_body_rates = (0.0, 0.0, 0.0)
    pass_tracker = VisionGatePassTracker(
        config=gate_pass_config,
        target_gate_count=args.target_gate_count,
    )
    arm_commands_sent = 0
    prearm_heartbeats_seen = 0
    spool_verified = False
    spool_wait_s = 0.0
    actuation_verified = False
    actuation_wait_s = 0.0
    course_send_lock = threading.Lock()
    course_publisher: FixedRateAttitudePublisher | None = None
    shadow_cadence_publisher: FixedRateAttitudePublisher | None = None
    official_reset_report: dict = {
        "reset_sent": False,
        "reset_detected": False,
    }
    official_race_start_s: float | None = None

    if official_reset_on_start:
        reset_result = reset_and_wait_for_official_policy_start(
            adapter,
            estimator,
            heartbeat_hz=args.heartbeat_hz,
            timeout_s=official_reset_start_timeout_s,
            policy_lead_s=official_policy_lead_s,
        )
        official_reset_report = reset_result.report
        official_race_start_s = reset_result.race_start_monotonic_s
        last_estimator_imu_us = reset_result.last_imu_time_usec
        estimator_calibration = dict(
            official_reset_report.get("calibration", {})
        )
        # No inference or hidden state from a previous rollout may cross the
        # official reset boundary.
        reset_policy_callable(policy_callable)
    elif policy_shadow_only:
        # Register for the live stream with a heartbeat, then reproduce the
        # official preflight's stationary IMU calibration without arming or
        # emitting any control command.
        adapter.send_heartbeat()
        heartbeats_sent += 1
        prearm_deadline_s = time.monotonic() + prearm_heartbeat_timeout_s
        while adapter.telemetry.metrics.heartbeats <= 0 and time.monotonic() < prearm_deadline_s:
            adapter.drain_telemetry(
                timeout_s=min(0.05, max(0.0, prearm_deadline_s - time.monotonic()))
            )
        prearm_heartbeats_seen = adapter.telemetry.metrics.heartbeats
        calibration_deadline_s = time.monotonic() + policy_shadow_calibration_timeout_s
        while not estimator.state.calibrated and time.monotonic() < calibration_deadline_s:
            adapter.drain_telemetry(timeout_s=0.05)
            imu_us = adapter.telemetry.state.imu_time_usec
            if imu_us is None or imu_us == last_estimator_imu_us:
                continue
            last_estimator_imu_us = imu_us
            st = adapter.telemetry.state
            if st.xacc is None or st.yacc is None or st.zacc is None:
                continue
            estimator.add_calibration_sample(
                (st.xacc, st.yacc, st.zacc),
                (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
            )
            if estimator.calibration_samples >= 60:
                estimator_calibration = estimator.finish_calibration()
        if not estimator.state.calibrated:
            raise RuntimeError(
                "policy shadow did not receive 60 stationary IMU samples: "
                f"samples={estimator.calibration_samples}, "
                f"messages={adapter.telemetry.metrics.messages_seen}, "
                f"heartbeats={adapter.telemetry.metrics.heartbeats}"
            )
    elif arm_on_start:
        prearm_deadline_s = time.monotonic() + prearm_heartbeat_timeout_s
        while adapter.telemetry.metrics.heartbeats <= 0 and time.monotonic() < prearm_deadline_s:
            adapter.drain_telemetry(
                timeout_s=min(0.05, max(0.0, prearm_deadline_s - time.monotonic()))
            )
        prearm_heartbeats_seen = adapter.telemetry.metrics.heartbeats
        # Calibrate gravity/mount while the drone is motionless on the pad;
        # after arming, spool-up vibration corrupts the estimate.
        calibration_deadline_s = time.monotonic() + 3.0
        while not estimator.state.calibrated and time.monotonic() < calibration_deadline_s:
            adapter.drain_telemetry(timeout_s=0.05)
            imu_us = adapter.telemetry.state.imu_time_usec
            if imu_us is None or imu_us == last_estimator_imu_us:
                continue
            last_estimator_imu_us = imu_us
            st = adapter.telemetry.state
            if st.xacc is None or st.yacc is None or st.zacc is None:
                continue
            estimator.add_calibration_sample(
                (st.xacc, st.yacc, st.zacc), (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0)
            )
            if estimator.calibration_samples >= 60:
                estimator_calibration = estimator.finish_calibration()
        for _attempt in range(arm_attempts):
            adapter.send_heartbeat()
            heartbeats_sent += 1
            adapter.send_arm_command()
            arm_commands_sent += 1
            adapter.drain_telemetry(timeout_s=0.0)
            time.sleep(min(0.05, heartbeat_period_s))

        if fixed_rate_attitude_mode:
            spool_started_s = time.monotonic()
            spool_deadline_s = spool_started_s + 6.0
            neutral = AttitudeSetpoint(thrust=course_controller.config.hover_thrust)
            next_spool_heartbeat_s = 0.0
            next_spool_arm_s = 0.0
            next_spool_command_s = 0.0
            while time.monotonic() < spool_deadline_s:
                spool_now_s = time.monotonic()
                if spool_now_s >= next_spool_heartbeat_s:
                    adapter.send_heartbeat()
                    heartbeats_sent += 1
                    next_spool_heartbeat_s = spool_now_s + 0.5
                if spool_now_s >= next_spool_arm_s:
                    adapter.send_arm_command()
                    arm_commands_sent += 1
                    next_spool_arm_s = spool_now_s + 0.2
                if spool_now_s >= next_spool_command_s:
                    adapter.send_attitude_setpoint(neutral, mode="body_rates")
                    next_spool_command_s = spool_now_s + 0.1
                adapter.drain_telemetry(timeout_s=0.02)
                outputs = adapter.telemetry.state.actuator_outputs or ()
                if outputs and max(outputs[:4]) > 0.1:
                    spool_verified = True
                    break
                time.sleep(0.02)
            spool_wait_s = time.monotonic() - spool_started_s
            if not spool_verified:
                raise RuntimeError(
                    "fixed-rate attitude motors did not spool before the 6 s preflight timeout"
                )

            actuation_started_s = time.monotonic()
            actuation_deadline_s = actuation_started_s + 6.0
            pulse = AttitudeSetpoint(
                body_yaw_rate=0.02,
                thrust=course_controller.config.hover_thrust,
            )
            next_pulse_heartbeat_s = 0.0
            next_pulse_arm_s = 0.0
            next_pulse_command_s = 0.0
            last_pulse_imu_us = adapter.telemetry.state.imu_time_usec
            pulse_response_samples = 0
            while time.monotonic() < actuation_deadline_s:
                pulse_now_s = time.monotonic()
                if pulse_now_s >= next_pulse_heartbeat_s:
                    adapter.send_heartbeat()
                    heartbeats_sent += 1
                    next_pulse_heartbeat_s = pulse_now_s + 0.5
                if pulse_now_s >= next_pulse_arm_s:
                    adapter.send_arm_command()
                    arm_commands_sent += 1
                    next_pulse_arm_s = pulse_now_s + 0.2
                if pulse_now_s >= next_pulse_command_s:
                    adapter.send_attitude_setpoint(pulse, mode="body_rates")
                    next_pulse_command_s = pulse_now_s + 0.05
                adapter.drain_telemetry(timeout_s=0.02)
                pulse_imu_us = adapter.telemetry.state.imu_time_usec
                if pulse_imu_us is not None and pulse_imu_us != last_pulse_imu_us:
                    last_pulse_imu_us = pulse_imu_us
                    if float(adapter.telemetry.state.zgyro or 0.0) < -0.01:
                        pulse_response_samples += 1
                    else:
                        pulse_response_samples = 0
                    if (
                        time.monotonic() - actuation_started_s >= 0.2
                        and pulse_response_samples >= 3
                    ):
                        actuation_verified = True
                        break
            actuation_wait_s = time.monotonic() - actuation_started_s
            if not actuation_verified:
                raise RuntimeError(
                    "body-rate actuation was not accepted before the 6 s preflight timeout"
                )

            settle_deadline_s = time.monotonic() + 1.5
            settle_samples = 0
            last_settle_imu_us = adapter.telemetry.state.imu_time_usec
            while time.monotonic() < settle_deadline_s and settle_samples < 3:
                adapter.send_attitude_setpoint(neutral, mode="body_rates")
                adapter.drain_telemetry(timeout_s=0.01)
                time.sleep(0.04)
                settle_imu_us = adapter.telemetry.state.imu_time_usec
                if settle_imu_us is not None and settle_imu_us != last_settle_imu_us:
                    last_settle_imu_us = settle_imu_us
                    if abs(float(adapter.telemetry.state.zgyro or 0.0)) < 0.005:
                        settle_samples += 1
                    else:
                        settle_samples = 0
            if settle_samples < 3:
                raise RuntimeError(
                    "body-rate preflight did not settle back to neutral"
                )

    started_s = (
        official_race_start_s
        if official_race_start_s is not None
        else time.monotonic()
    )
    deadline_s = started_s + args.duration
    if fixed_rate_attitude_mode and not official_reset_on_start and not policy_shadow_only:
        course_publisher = FixedRateAttitudePublisher(
            adapter,
            command_hz=args.command_hz,
            initial_target=AttitudeSetpoint(thrust=course_controller.config.hover_thrust),
            send_lock=course_send_lock,
        )
        course_publisher.start()
    elif policy_shadow_only:
        # Measure the publisher scheduler under the live camera/inference load
        # without handing any setpoint to the MAVLink adapter.
        shadow_cadence_publisher = FixedRateAttitudePublisher(
            _NoopAttitudeAdapter(),
            command_hz=args.command_hz,
            initial_target=AttitudeSetpoint(thrust=0.0),
            send_lock=threading.Lock(),
        )
        shadow_cadence_publisher.start()

    try:
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if (
                fixed_rate_attitude_mode
                and adapter.telemetry.metrics.collisions > 0
            ):
                limp_state.trigger(
                    "official_collision_message",
                    now_s - started_s,
                )
            race_status = adapter.telemetry.state.race_status
            official_gate_index = (
                int(race_status.active_gate_index) if race_status is not None else None
            )
            predictor_gain = control_aware_gate_predictor_gain(
                enabled=policy_control_aware_gate_predictor_enabled,
                official_gate_index=official_gate_index,
                start_index=policy_control_aware_gate_predictor_start_index,
            )
            predict_without_measurement = gate_dropout_prediction_enabled(
                enabled=policy_predict_gate_dropout_enabled,
                official_gate_index=official_gate_index,
                start_index=policy_predict_gate_dropout_start_index,
            )
            if (
                policy_gate_motion_state.config.control_accel_gain_m_s2_per_tan_roll
                != predictor_gain
                or policy_gate_motion_state.config.predict_without_measurement
                != predict_without_measurement
            ):
                policy_gate_motion_state.config = dataclasses.replace(
                    policy_gate_motion_state.config,
                    control_accel_gain_m_s2_per_tan_roll=predictor_gain,
                    predict_without_measurement=predict_without_measurement,
                )
            official_gate_observation_changed = (
                official_gate_observation_epoch.observe(official_gate_index)
            )
            if (
                official_gate_observation_changed
                and official_gate_index is not None
                and official_gate_index >= 2
                and official_gate_index != policy_gate_transition_preseed_index
            ):
                # Camera and command loops are asynchronous. Immediately after
                # Gate 2, the most recent allowed pose can still belong to the
                # gate just crossed. Preserve the historically proven Gate-2
                # handoff, but never initialize Gate 3 or a later gate's motion
                # association from that stale cached pose: clear the cache and
                # wait for the next frame carrying the new identity.
                gate_detection = None
                gate_pose = None
                last_detection_s = None
                control_gate_detection = None
                control_gate_pose = None
                last_control_detection_s = None
                last_landmark_obs_s = None
                policy_gate_motion_state.reset()
            if (
                official_gate_observation_changed
                and official_gate_index is not None
                and official_gate_index == policy_gate_transition_preseed_index
            ):
                policy_gate_motion_state.preseed_identity_from_last_rejection(
                    official_gate_index,
                    time_s=now_s,
                    max_age_s=policy_gate_transition_preseed_max_age_s,
                    min_range_margin_m=(
                        policy_gate_transition_preseed_min_range_margin_m
                    ),
                )
            if (
                fixed_rate_attitude_mode
                and official_gate_index is not None
                and official_gate_index > pass_tracker.pass_count
            ):
                # Official race status is authoritative. Keep the diagnostic
                # vision counter aligned so acceptance cannot reject a real
                # pass merely because the gate left the FOV before debounce.
                pass_tracker.sync_official_advance(official_gate_index)
            if should_stop_after_official_finish(race_status):
                finish_time_ns = int(race_status.race_finish_time_ns)
                official_finish_stop.update(
                    {
                        "triggered": True,
                        "elapsed_s": round_float(now_s - started_s),
                        "official_gate_index": official_gate_index,
                        "race_finish_time_ns": finish_time_ns,
                        "race_finish_time_s": round_float(finish_time_ns / 1e9),
                    }
                )
                break
            if should_stop_after_official_gate_index(
                official_gate_index,
                official_gate_index_stop["configured_gate_index"],
            ):
                official_gate_index_stop.update(
                    {
                        "triggered": True,
                        "elapsed_s": round_float(now_s - started_s),
                        "official_gate_index": official_gate_index,
                    }
                )
                break
            if control_mode == "visual-servo-angles" and official_gate_state.observe(
                elapsed_s=now_s - started_s,
                official_gate_index=official_gate_index,
                estimator=estimator,
                angle_servo=angle_servo_state,
                map_guidance=map_guidance_state,
                pass_tracker=pass_tracker,
            ):
                angle_mode_counters["retarget_events"] += 1
                control_gate_pose = None
                control_gate_detection = None
                last_control_detection_s = None
                last_landmark_obs_s = None
                angle_recovery_active = False

            if limp_state.active and limp_abort_exit:
                break

            if now_s >= next_heartbeat_s:
                if course_publisher is not None:
                    with course_send_lock:
                        adapter.send_heartbeat()
                else:
                    adapter.send_heartbeat()
                heartbeats_sent += 1
                next_heartbeat_s = now_s + heartbeat_period_s

            if arm_on_start and fixed_rate_attitude_mode and now_s >= next_arm_s:
                with course_send_lock:
                    adapter.send_arm_command()
                arm_commands_sent += 1
                next_arm_s = now_s + 0.2

            if receiver is not None and detector is not None:
                vision_due = not fixed_rate_attitude_mode or now_s >= next_course_vision_s
                frames = (
                    receiver.poll_frames(max_packets=args.camera_max_packets_per_loop)
                    if vision_due
                    else []
                )
                if frames:
                    if fixed_rate_attitude_mode:
                        next_course_vision_s = now_s + 1.0 / 15.0
                    frame = frames[-1]
                    vision_metrics.frames_seen += 1
                    final_phase_edge_preference = (
                        control_mode == "policy-attitude"
                        and raw_gate_observation_enabled(
                            official_gate_index=official_gate_index,
                            start_index=policy_raw_gate_observation_index,
                            end_index=policy_raw_gate_observation_end_index,
                        )
                    )
                    (
                        policy_final_edge_gate_index,
                        policy_final_edge_phase_started_s,
                    ) = update_post_prefix_edge_phase(
                        enabled=final_phase_edge_preference,
                        official_gate_index=official_gate_index,
                        now_s=now_s,
                        phase_gate_index=policy_final_edge_gate_index,
                        phase_started_s=policy_final_edge_phase_started_s,
                    )
                    final_edge_phase_elapsed_s = (
                        0.0
                        if policy_final_edge_phase_started_s is None
                        else now_s - policy_final_edge_phase_started_s
                    )
                    detection = detector.detect_jpeg(
                        frame.jpeg,
                        prefer_edge_frame=final_phase_edge_preference,
                        prefer_any_edge_frame=final_phase_any_edge_priority_enabled(
                            final_phase_edge_preference=final_phase_edge_preference,
                            configured=policy_final_prefer_any_edge_frame,
                            phase_elapsed_s=final_edge_phase_elapsed_s,
                            delay_s=policy_final_any_edge_delay_s,
                        ),
                    )
                    if guidance_detector is not None:
                        guidance_detection = guidance_detector.detect_jpeg(frame.jpeg)
                        if guidance_detection is not None:
                            last_guidance_detection_s = now_s
                    gate_detection = detection
                    vision_metrics.detector_decode_failures = detector.metrics.decode_failures
                    vision_metrics.detector_no_quad_found = detector.metrics.no_quad_found
                    vision_metrics.detector_detections = detector.metrics.detections
                    tracked_pose = None
                    if detection is not None:
                        # Frame (outer red structure) detections subtend ~2x the
                        # inner aperture; use the matching physical width so the
                        # range estimate does not flip between sources.
                        physical_width_m = (
                            float(getattr(args, "gate_inner_width_m", 1.5))
                            if getattr(detection, "source", "frame") == "aperture"
                            else float(getattr(args, "gate_outer_width_m", 3.0))
                        )
                        tracked_pose = estimate_gate_pose_from_corners(
                            detection.corners,
                            gate_inner_width_m=physical_width_m,
                            camera_uptilt_deg=float(
                                getattr(args, "camera_uptilt_deg", 0.0)
                            ),
                        )
                        gate_pose = tracked_pose
                        vision_metrics.last_detection_confidence = float(detection.confidence)
                        last_detection_s = now_s
                        if control_mode in {
                            "visual-servo-attitude",
                            "visual-servo-angles",
                            "policy-attitude",
                            "course-fsm",
                        } and attitude_control_pose_allowed(
                            tracked_pose,
                            detection,
                            args,
                        ):
                            control_gate_pose = tracked_pose
                            control_gate_detection = detection
                            last_control_detection_s = now_s
                            if estimator.state.calibrated and tracked_pose.range_camera_m <= float(
                                getattr(args, "landmark_max_range_m", 35.0)
                            ):
                                # Far gates down the track alias as the active
                                # gate; folding them into the landmark whipsaws
                                # the position fix by tens of meters.
                                gate_id = official_gate_index if official_gate_index is not None else 0
                                estimator.observe_gate(
                                    gate_id,
                                    tracked_pose.body_vector_ned_m,
                                    time_s=now_s,
                                )
                                last_landmark_obs_s = now_s
                        diagnostic_cmd = visual_servo_command_from_args(
                            tracked_pose,
                            yaw_rad=safe_float(adapter.telemetry.state.yaw),
                            args=args,
                        )
                        diagnostic_target = LocalNedSetpoint(
                            vx=diagnostic_cmd.vx,
                            vy=diagnostic_cmd.vy,
                            vz=diagnostic_cmd.vz,
                            yaw_rate=diagnostic_cmd.yaw_rate,
                        )
                        if control_mode == "visual-servo-attitude":
                            actual_command = {
                                "kind": "visual_servo_attitude_target",
                                **attitude_setpoint_to_dict(attitude_servo_command_from_args(tracked_pose, args)),
                            }
                        elif control_mode == "visual-servo-angles":
                            actual_command = {
                                "kind": "visual_servo_angles_target",
                                **attitude_setpoint_to_dict(attitude_target),
                            }
                        elif control_mode == "course-fsm":
                            # Detection is processed before this tick's command
                            # update, so report the most recently sent FSM
                            # attitude target rather than an unrelated velocity
                            # diagnostic.
                            actual_command = {
                                "kind": "course_fsm_attitude_target",
                                **attitude_setpoint_to_dict(attitude_target),
                            }
                        elif control_mode == "policy-attitude":
                            actual_command = {
                                "kind": "policy_attitude_target",
                                **attitude_setpoint_to_dict(attitude_target),
                            }
                        elif control_mode == "attitude-rates":
                            actual_command = {
                                "kind": "attitude_target",
                                **attitude_setpoint_to_dict(attitude_setpoint_from_args(args)),
                            }
                        else:
                            actual_command = {
                                "kind": "local_ned_setpoint",
                                **setpoint_to_dict(diagnostic_target),
                            }
                        update_approach_diagnostics(
                            approach_diagnostics,
                            elapsed_s=now_s - started_s,
                            sim_time_ns=frame.sim_time_ns,
                            gate_pose=tracked_pose,
                            detection=detection,
                            visual_servo_target=diagnostic_target,
                            actual_command=actual_command,
                            max_samples=max_approach_diagnostic_samples,
                        )
                        if debug_output_dir is not None:
                            range_m = float(tracked_pose.range_camera_m)
                            defer_debug_frame(
                                deferred_debug_frames,
                                frame=frame,
                                detection=detection,
                                range_m=range_m,
                            )
                    pass_tracker.observe(
                        now_s=now_s,
                        gate_pose=tracked_pose,
                        detection_confidence=(None if detection is None else detection.confidence),
                        detection_present=(detection is not None),
                    )
                    if should_stop_before_policy_gate(
                        control_mode=control_mode,
                        official_gate_index=official_gate_index,
                        pose=tracked_pose,
                        stop_gate_index=policy_stop_gate_index,
                        stop_forward_m=policy_stop_forward_m,
                    ):
                        bx, by, bz = tracked_pose.body_vector_ned_m
                        policy_gate_stop.update(
                            {
                                "triggered": True,
                                "elapsed_s": round_float(now_s - started_s),
                                "official_gate_index": official_gate_index,
                                "forward_m": round_float(bx),
                                "right_m": round_float(by),
                                "down_m": round_float(bz),
                                "yaw_error_rad": round_float(
                                    tracked_pose.yaw_error_rad
                                ),
                            }
                        )
                        break
                    update_timesync_metrics(
                        vision_metrics,
                        sim_time_ns=frame.sim_time_ns,
                        timesync_ts1=adapter.telemetry.state.timesync_ts1,
                    )

            if last_detection_s is not None:
                vision_metrics.last_detection_age_s = max(0.0, now_s - last_detection_s)

            if now_s >= next_command_s:
                if (
                    not fixed_rate_attitude_mode
                    and last_command_sent_s is not None
                    and now_s - last_command_sent_s < 0.01
                ):
                    command_rate_violations += 1

                telemetry = adapter.telemetry.state
                yaw = safe_float(telemetry.yaw)
                if control_mode == "visual-servo":
                    if gate_pose is not None and (
                        last_detection_s is None or now_s - last_detection_s <= args.max_detection_age_s
                    ):
                        cmd = visual_servo_command_from_args(gate_pose, yaw_rad=yaw, args=args)
                        target = LocalNedSetpoint(vx=cmd.vx, vy=cmd.vy, vz=cmd.vz, yaw_rate=cmd.yaw_rate)
                    else:
                        target = LocalNedSetpoint()
                    last_cmd_norm = (
                        clamp(target.vx / 2.0, -1.0, 1.0),
                        clamp(target.vy / 1.0, -1.0, 1.0),
                        clamp(target.vz / 0.8, -1.0, 1.0),
                        clamp(target.yaw_rate / 1.0, -1.0, 1.0),
                    )
                elif control_mode == "course-fsm":
                    fresh_course_pose = None
                    fresh_course_confidence = 0.0
                    if (
                        control_gate_pose is not None
                        and last_control_detection_s is not None
                        and now_s - last_control_detection_s <= args.max_detection_age_s
                    ):
                        fresh_course_pose = control_gate_pose
                        fresh_course_confidence = float(
                            getattr(control_gate_detection, "confidence", 0.0)
                        )
                    fresh_guidance = None
                    if (
                        guidance_detection is not None
                        and last_guidance_detection_s is not None
                        and now_s - last_guidance_detection_s
                        <= float(getattr(args, "course_guidance_max_age_s", 0.3))
                    ):
                        fresh_guidance = guidance_detection
                    measured_yaw = estimator.state.yaw if estimator.state.calibrated else yaw
                    course_command = course_controller.update(
                        now_s=now_s - started_s,
                        measured_yaw_rad=measured_yaw,
                        official_gate_index=official_gate_index,
                        gate_pose=fresh_course_pose,
                        gate_confidence=fresh_course_confidence,
                        guidance=fresh_guidance,
                    )
                    policy_activation_gate = int(
                        getattr(args, "course_policy_activate_gate_index", 1)
                    )
                    brake_completed = any(
                        transition.get("reason") == "brake_complete"
                        and int(course_controller.last_official_gate_index or 0)
                        >= policy_activation_gate
                        for transition in course_controller.transitions
                    )
                    if (
                        policy_callable is not None
                        and official_gate_index is not None
                        and official_gate_index >= policy_activation_gate
                        and brake_completed
                    ):
                        if course_policy_activated_s is None:
                            course_policy_activated_s = now_s
                        policy_elapsed = now_s - course_policy_activated_s
                        policy_duration = max(
                            1e-6,
                            args.duration - (course_policy_activated_s - started_s),
                        )
                        observation = build_policy_observation(
                            telemetry,
                            gate_detection=(
                                control_gate_detection
                                if fresh_course_pose is not None
                                else None
                            ),
                            gate_pose=fresh_course_pose,
                            elapsed_fraction=clamp(
                                policy_elapsed / policy_duration, 0.0, 1.0
                            ),
                            last_cmd_norm=last_cmd_norm,
                            estimator_state=(
                                estimator.state if estimator.state.calibrated else None
                            ),
                            guidance=fresh_guidance,
                            observation_time_s=now_s,
                            gate_motion_state=policy_gate_motion_state,
                            gate_identity=official_gate_index,
                            race_phase_gate_index=(
                                official_gate_index
                                if getattr(args, "policy_race_phase_observation", False) else None
                            ),
                            race_phase_denominator=getattr(
                                args, "policy_race_phase_denominator", 3
                            ),
                            phase_adapter_gate_index=(
                                official_gate_index
                                if getattr(args, "policy_phase_adapter_observation", False)
                                else None
                            ),
                            gate_progress_adapter_gate_index=(
                                official_gate_index
                                if getattr(
                                    args,
                                    "policy_gate_progress_adapter_observation",
                                    False,
                                )
                                else None
                            ),
                            gate_progress_adapter_denominator=getattr(
                                args, "policy_race_phase_denominator", 6
                            ),
                            gate_phase_onehot_adapter_gate_index=(
                                official_gate_index
                                if getattr(
                                    args,
                                    "policy_gate_phase_onehot_adapter_observation",
                                    False,
                                )
                                else None
                            ),
                            gate_phase_onehot_adapter_denominator=getattr(
                                args, "policy_race_phase_denominator", 6
                            ),
                            hybrid_prefix_confidence_observation=getattr(
                                args,
                                "policy_hybrid_prefix_confidence_observation",
                                False,
                            ),
                            hybrid_prefix_elapsed_fraction=clamp(
                                policy_elapsed / LEGACY_HYBRID_PREFIX_DURATION_S,
                                0.0,
                                1.0,
                            ),
                        )
                        action_values = policy_callable(observation)
                        action, setpoint = decode_attitude_policy_action(action_values)
                        attitude_target = AttitudeSetpoint(
                            roll=setpoint.roll_rad,
                            pitch=setpoint.pitch_rad,
                            yaw=setpoint.yaw_rad,
                            thrust=setpoint.thrust,
                        )
                        last_cmd_norm = attitude_last_cmd_normalized(action)
                        course_policy_ticks += 1
                        if policy_trace_hz > 0.0 and now_s >= next_policy_trace_s:
                            append_policy_trace_sample(
                                policy_trace_samples,
                                max_samples=policy_trace_max_samples,
                                now_s=now_s,
                                started_s=started_s,
                                official_gate_index=official_gate_index,
                                observation=observation,
                                action_values=action.normalized,
                                decoded_command=setpoint,
                                gate_pose=fresh_course_pose,
                                motion_filter=policy_gate_motion_state,
                            )
                            next_policy_trace_s = now_s + 1.0 / policy_trace_hz
                    else:
                        attitude_target = AttitudeSetpoint(
                            roll=course_command.roll,
                            pitch=course_command.pitch,
                            yaw=course_command.yaw,
                            thrust=course_command.thrust,
                        )
                        last_cmd_norm = (
                            clamp(attitude_target.pitch / 0.5, -1.0, 1.0),
                            clamp(attitude_target.roll / 0.5, -1.0, 1.0),
                            normalize_attitude_thrust(attitude_target.thrust),
                            clamp(attitude_target.yaw / math.pi, -1.0, 1.0),
                        )
                elif control_mode == "policy":
                    elapsed_fraction = clamp((now_s - started_s) / args.duration, 0.0, 1.0)
                    observation = build_policy_observation(
                        telemetry,
                        gate_detection=gate_detection,
                        gate_pose=gate_pose,
                        elapsed_fraction=elapsed_fraction,
                        last_cmd_norm=last_cmd_norm,
                        estimator_state=estimator.state if estimator.state.calibrated else None,
                        observation_time_s=now_s,
                        gate_motion_state=policy_gate_motion_state,
                        gate_identity=official_gate_index,
                        race_phase_gate_index=(
                            official_gate_index
                            if getattr(args, "policy_race_phase_observation", False) else None
                        ),
                        race_phase_denominator=getattr(
                            args, "policy_race_phase_denominator", 3
                        ),
                        phase_adapter_gate_index=(
                            official_gate_index
                            if getattr(args, "policy_phase_adapter_observation", False)
                            else None
                        ),
                        gate_progress_adapter_gate_index=(
                            official_gate_index
                            if getattr(
                                args,
                                "policy_gate_progress_adapter_observation",
                                False,
                            )
                            else None
                        ),
                        gate_progress_adapter_denominator=getattr(
                            args, "policy_race_phase_denominator", 6
                        ),
                        gate_phase_onehot_adapter_gate_index=(
                            official_gate_index
                            if getattr(
                                args,
                                "policy_gate_phase_onehot_adapter_observation",
                                False,
                            )
                            else None
                        ),
                        gate_phase_onehot_adapter_denominator=getattr(
                            args, "policy_race_phase_denominator", 6
                        ),
                        hybrid_prefix_confidence_observation=getattr(
                            args,
                            "policy_hybrid_prefix_confidence_observation",
                            False,
                        ),
                        hybrid_prefix_elapsed_fraction=clamp(
                            (now_s - started_s) / LEGACY_HYBRID_PREFIX_DURATION_S,
                            0.0,
                            1.0,
                        ),
                    )
                    action_values = policy_callable(observation) if policy_callable else normalized_action
                    action, setpoint = decode_policy_action(action_values, yaw_rad=yaw)
                    target = LocalNedSetpoint(
                        vx=setpoint.vx_m_s,
                        vy=setpoint.vy_m_s,
                        vz=setpoint.vz_m_s,
                        yaw_rate=setpoint.yaw_rate_rad_s,
                    )
                    last_cmd_norm = action.normalized
                    if policy_trace_hz > 0.0 and now_s >= next_policy_trace_s:
                        append_policy_trace_sample(
                            policy_trace_samples,
                            max_samples=policy_trace_max_samples,
                            now_s=now_s,
                            started_s=started_s,
                            official_gate_index=official_gate_index,
                            observation=observation,
                            action_values=action.normalized,
                            decoded_command=setpoint,
                            gate_pose=gate_pose,
                            motion_filter=policy_gate_motion_state,
                        )
                        next_policy_trace_s = now_s + 1.0 / policy_trace_hz
                elif control_mode == "policy-attitude":
                    elapsed_fraction = clamp((now_s - started_s) / args.duration, 0.0, 1.0)
                    fresh_policy_pose = None
                    fresh_policy_detection = None
                    if (
                        control_gate_pose is not None
                        and last_control_detection_s is not None
                        and now_s - last_control_detection_s <= args.max_detection_age_s
                    ):
                        fresh_policy_pose = control_gate_pose
                        fresh_policy_detection = control_gate_detection
                    fresh_policy_guidance = None
                    if (
                        guidance_detection is not None
                        and last_guidance_detection_s is not None
                        and now_s - last_guidance_detection_s
                        <= float(getattr(args, "course_guidance_max_age_s", 0.3))
                    ):
                        fresh_policy_guidance = guidance_detection
                    observation = build_policy_observation(
                        telemetry,
                        gate_detection=fresh_policy_detection,
                        gate_pose=fresh_policy_pose,
                        elapsed_fraction=elapsed_fraction,
                        last_cmd_norm=last_cmd_norm,
                        estimator_state=estimator.state if estimator.state.calibrated else None,
                        guidance=fresh_policy_guidance,
                        observation_time_s=now_s,
                        gate_motion_state=(
                            None
                            if raw_gate_observation_enabled(
                                official_gate_index=official_gate_index,
                                start_index=policy_raw_gate_observation_index,
                                end_index=policy_raw_gate_observation_end_index,
                            )
                            else policy_gate_motion_state
                        ),
                        gate_identity=official_gate_index,
                        race_phase_gate_index=(
                            official_gate_index
                            if getattr(args, "policy_race_phase_observation", False) else None
                        ),
                        race_phase_denominator=getattr(
                            args, "policy_race_phase_denominator", 3
                        ),
                        phase_adapter_gate_index=(
                            official_gate_index
                            if getattr(args, "policy_phase_adapter_observation", False)
                            else None
                        ),
                        gate_progress_adapter_gate_index=(
                            official_gate_index
                            if getattr(
                                args,
                                "policy_gate_progress_adapter_observation",
                                False,
                            )
                            else None
                        ),
                        gate_progress_adapter_denominator=getattr(
                            args, "policy_race_phase_denominator", 6
                        ),
                        gate_phase_onehot_adapter_gate_index=(
                            official_gate_index
                            if getattr(
                                args,
                                "policy_gate_phase_onehot_adapter_observation",
                                False,
                            )
                            else None
                        ),
                        gate_phase_onehot_adapter_denominator=getattr(
                            args, "policy_race_phase_denominator", 6
                        ),
                        hybrid_prefix_confidence_observation=getattr(
                            args,
                            "policy_hybrid_prefix_confidence_observation",
                            False,
                        ),
                        hybrid_prefix_elapsed_fraction=clamp(
                            (now_s - started_s) / LEGACY_HYBRID_PREFIX_DURATION_S,
                            0.0,
                            1.0,
                        ),
                    )
                    policy_steps_due = fixed_policy_steps_due(
                        elapsed_s=now_s - started_s,
                        state_hz=policy_state_hz,
                        completed_steps=policy_inference_ticks,
                    )
                    policy_state_schedule["outer_updates"] += 1
                    policy_state_schedule["max_steps_per_update"] = max(
                        policy_state_schedule["max_steps_per_update"],
                        policy_steps_due,
                    )
                    if policy_steps_due == 0:
                        policy_state_schedule["zero_step_updates"] += 1
                    elif policy_steps_due > 1:
                        policy_state_schedule["catchup_updates"] += 1

                    step_observation = observation
                    for _policy_step in range(policy_steps_due):
                        action_values = (
                            policy_callable(step_observation)
                            if policy_callable
                            else normalized_action
                        )
                        policy_inference_ticks += 1
                        action, setpoint = decode_attitude_policy_action(
                            action_values
                        )
                        attitude_target = AttitudeSetpoint(
                            roll=setpoint.roll_rad,
                            pitch=setpoint.pitch_rad,
                            yaw=setpoint.yaw_rad,
                            thrust=setpoint.thrust,
                        )
                        last_cmd_norm = attitude_last_cmd_normalized(action)
                        trace_now_s = (
                            now_s
                            if policy_state_hz <= 0.0
                            else started_s
                            + (policy_inference_ticks - 1) / policy_state_hz
                        )
                        if (
                            policy_trace_hz > 0.0
                            and trace_now_s >= next_policy_trace_s
                        ):
                            append_policy_trace_sample(
                                policy_trace_samples,
                                max_samples=policy_trace_max_samples,
                                now_s=trace_now_s,
                                started_s=started_s,
                                official_gate_index=official_gate_index,
                                observation=step_observation,
                                action_values=action.normalized,
                                decoded_command=setpoint,
                                gate_pose=fresh_policy_pose,
                                motion_filter=policy_gate_motion_state,
                                gate_detection=fresh_policy_detection,
                            )
                            next_policy_trace_s = (
                                trace_now_s + 1.0 / policy_trace_hz
                            )
                        step_observation = policy_observation_with_last_command(
                            step_observation,
                            last_cmd_norm,
                        )
                elif control_mode == "visual-servo-attitude":
                    fresh_control_pose = None
                    fresh_control_detection = None
                    if (
                        control_gate_pose is not None
                        and last_control_detection_s is not None
                        and now_s - last_control_detection_s <= args.max_detection_age_s
                    ):
                        fresh_control_pose = control_gate_pose
                        fresh_control_detection = control_gate_detection

                    if (
                        final_approach_state.is_active(now_s)
                        and final_approach_state.active_command is not None
                    ):
                        attitude_target = final_approach_state.active_command
                    elif (
                        fresh_control_pose is not None
                        and should_start_attitude_final_approach(
                            fresh_control_pose,
                            fresh_control_detection,
                            args,
                            final_approach_state,
                            now_s=now_s,
                        )
                    ):
                        attitude_target = attitude_final_approach_command_from_args(fresh_control_pose, args)
                        final_approach_state.activate(
                            now_s=now_s,
                            started_s=started_s,
                            duration_s=float(getattr(args, "attitude_servo_final_duration_s", 1.0)),
                            pose=fresh_control_pose,
                            command=attitude_target,
                        )
                    elif fresh_control_pose is not None:
                        attitude_target = attitude_servo_command_from_args(fresh_control_pose, args)
                    else:
                        attitude_target = attitude_search_command_from_args(args)
                    last_cmd_norm = (
                        clamp(attitude_target.body_pitch_rate / 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_roll_rate / 1.0, -1.0, 1.0),
                        normalize_attitude_thrust(attitude_target.thrust),
                        clamp(attitude_target.body_yaw_rate / 1.0, -1.0, 1.0),
                    )
                elif control_mode == "visual-servo-angles":
                    fresh_control_pose = None
                    if (
                        control_gate_pose is not None
                        and last_control_detection_s is not None
                        and now_s - last_control_detection_s <= args.max_detection_age_s
                    ):
                        fresh_control_pose = control_gate_pose

                    est_state = estimator.state
                    ahrs_ready = estimator.state.calibrated and angle_servo_closed_loop
                    recovery_trigger = float(
                        getattr(args, "angle_servo_recovery_trigger_rad", 0.9)
                    )
                    recovery_release = float(
                        getattr(args, "angle_servo_recovery_release_rad", 0.25)
                    )
                    limp_recovery_timeout_s = float(
                        getattr(args, "limp_recovery_timeout_s", 3.0)
                    )
                    limp_max_speed_m_s = float(getattr(args, "limp_max_speed_m_s", 0.15))
                    limp_collision_grace_s = float(
                        getattr(args, "limp_collision_grace_s", 2.0)
                    )

                    angle_path = "servo"
                    if limp_state.active:
                        angle_mode_counters["limp_ticks"] += 1
                        angle_path = "limp"
                        hover = float(getattr(args, "attitude_servo_hover_thrust", 0.6))
                        min_thrust = float(getattr(args, "attitude_servo_min_thrust", 0.5))
                        max_thrust = float(getattr(args, "attitude_servo_max_thrust", 0.72))
                        attitude_target = AttitudeSetpoint(
                            roll=0.0,
                            pitch=0.0,
                            yaw=est_state.yaw if ahrs_ready else angle_servo_state.yaw_cmd_rad,
                            thrust=clamp(
                                hover + angle_servo_state.thrust_bias,
                                min_thrust,
                                max_thrust,
                            ),
                        )
                    else:
                        if ahrs_ready:
                            tilt = max(abs(est_state.roll), abs(est_state.pitch))
                            if angle_recovery_active:
                                if tilt < recovery_release:
                                    angle_recovery_active = False
                            elif tilt > recovery_trigger:
                                angle_recovery_active = True
                                angle_mode_counters["recovery_events"] += 1
                        else:
                            angle_recovery_active = False

                        if limp_state.observe(
                            now_s=now_s,
                            elapsed_s=now_s - started_s,
                            recovery_active=angle_recovery_active,
                            collision_count=est_state.collision_events,
                            velocity_ned_m_s=est_state.velocity_ned_m_s,
                            recovery_timeout_s=limp_recovery_timeout_s,
                            max_speed_m_s=limp_max_speed_m_s,
                            collision_grace_s=limp_collision_grace_s,
                        ):
                            angle_mode_counters["limp_events"] += 1
                            angle_recovery_active = False
                            if limp_auto_reset:
                                adapter.send_heartbeat()
                                adapter.send_sim_reset_command()
                            if limp_abort_exit:
                                attitude_target = AttitudeSetpoint(
                                    roll=0.0,
                                    pitch=0.0,
                                    yaw=est_state.yaw if ahrs_ready else angle_servo_state.yaw_cmd_rad,
                                    thrust=float(
                                        getattr(args, "attitude_servo_hover_thrust", 0.6)
                                    ),
                                )
                            else:
                                angle_path = "limp"
                                hover = float(
                                    getattr(args, "attitude_servo_hover_thrust", 0.6)
                                )
                                min_thrust = float(
                                    getattr(args, "attitude_servo_min_thrust", 0.5)
                                )
                                max_thrust = float(
                                    getattr(args, "attitude_servo_max_thrust", 0.72)
                                )
                                attitude_target = AttitudeSetpoint(
                                    roll=0.0,
                                    pitch=0.0,
                                    yaw=est_state.yaw if ahrs_ready else angle_servo_state.yaw_cmd_rad,
                                    thrust=clamp(
                                        hover + angle_servo_state.thrust_bias,
                                        min_thrust,
                                        max_thrust,
                                    ),
                                )

                        if not limp_state.active:
                            if angle_recovery_active:
                                # Tumbling: abandon the approach and right the airframe.
                                angle_mode_counters["recovery_ticks"] += 1
                                angle_path = "recovery"
                                attitude_target = AttitudeSetpoint(
                                    roll=0.0,
                                    pitch=0.0,
                                    yaw=est_state.yaw,
                                    thrust=clamp(
                                        float(getattr(args, "attitude_servo_hover_thrust", 0.6))
                                        + angle_servo_state.thrust_bias,
                                        float(getattr(args, "attitude_servo_min_thrust", 0.5)),
                                        float(getattr(args, "attitude_servo_max_thrust", 0.72)),
                                    ),
                                )
                            else:
                                if ahrs_ready and abs(
                                    wrap_angle(angle_servo_state.yaw_cmd_rad - est_state.yaw)
                                ) > 0.6:
                                    # Re-anchor the servo's yaw integrator to the AHRS
                                    # yaw so recovery/tumble episodes cannot leave the
                                    # desired yaw pointing somewhere the drone is not.
                                    angle_servo_state.yaw_cmd_rad = est_state.yaw
                                attitude_target = None
                                # Map guidance is only as good as the last landmark
                                # fix; once dead reckoning runs open-loop for a while
                                # the mapped relative position is fiction, so hand
                                # back to the servo's blind/search behavior instead.
                                map_fix_fresh = (
                                    last_landmark_obs_s is not None
                                    and now_s - last_landmark_obs_s
                                    <= float(getattr(args, "map_guidance_max_fix_age_s", 2.0))
                                )
                                if fresh_control_pose is None and ahrs_ready and map_fix_fresh:
                                    active_gate_now = (
                                        official_gate_index
                                        if official_gate_index is not None
                                        else 0
                                    )
                                    attitude_target = map_guidance_state.command(
                                        estimator,
                                        active_gate_now,
                                        args,
                                        thrust_bias=angle_servo_state.thrust_bias,
                                    )
                                    if attitude_target is not None:
                                        angle_mode_counters["map_ticks"] += 1
                                        angle_path = "map"
                                if attitude_target is None:
                                    attitude_target = angle_servo_state.update(
                                        now_s=now_s,
                                        pose=fresh_control_pose,
                                        args=args,
                                        dt=command_period_s,
                                    )
                                    if fresh_control_pose is not None:
                                        angle_mode_counters["visual_ticks"] += 1
                                        angle_path = "visual"
                                    else:
                                        angle_mode_counters["servo_blind_ticks"] += 1
                                        angle_path = "blind"
                                if ahrs_ready:
                                    # Speed governor: the closed-loop plant actually
                                    # reaches commanded pitch now, so cap forward speed
                                    # with the estimator velocity instead of trusting
                                    # pose-rate damping alone.
                                    vx, vy, _vz = est_state.velocity_ned_m_s
                                    v_fwd = vx * math.cos(est_state.yaw) + vy * math.sin(est_state.yaw)
                                    max_speed = float(getattr(args, "angle_servo_max_speed_m_s", 2.0))
                                    k_speed = float(getattr(args, "angle_servo_k_speed", 0.06))
                                    if v_fwd > max_speed and attitude_target.pitch < 0.0:
                                        eased = min(
                                            0.0,
                                            attitude_target.pitch + k_speed * (v_fwd - max_speed),
                                        )
                                        angle_mode_counters["governor_ticks"] += 1
                                        attitude_target = AttitudeSetpoint(
                                            roll=attitude_target.roll,
                                            pitch=eased,
                                            yaw=attitude_target.yaw,
                                            thrust=attitude_target.thrust,
                                        )
                    if now_s >= next_angle_trace_s:
                        next_angle_trace_s = now_s + 0.5
                        angle_mode_trace.append(
                            {
                                "elapsed_s": round(now_s - started_s, 3),
                                "path": angle_path,
                                "cmd_rpy_thrust": (
                                    round(attitude_target.roll, 4),
                                    round(attitude_target.pitch, 4),
                                    round(attitude_target.yaw, 4),
                                    round(attitude_target.thrust, 4),
                                ),
                                "est_rpy": (
                                    round(est_state.roll, 4),
                                    round(est_state.pitch, 4),
                                    round(est_state.yaw, 4),
                                ),
                                "est_pos": tuple(round(v, 2) for v in est_state.position_ned_m),
                                "est_vel": tuple(round(v, 2) for v in est_state.velocity_ned_m_s),
                            }
                        )
                    last_cmd_norm = (
                        clamp(attitude_target.pitch / 0.5, -1.0, 1.0),
                        clamp(attitude_target.roll / 0.5, -1.0, 1.0),
                        normalize_attitude_thrust(attitude_target.thrust),
                        clamp(attitude_target.yaw / math.pi, -1.0, 1.0),
                    )
                else:
                    last_cmd_norm = (
                        clamp(attitude_target.body_pitch_rate / 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_roll_rate / 1.0, -1.0, 1.0),
                        clamp(attitude_target.thrust * 2.0 - 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_yaw_rate / 1.0, -1.0, 1.0),
                    )

                if policy_shadow_only:
                    # The complete observation/callable/decoder path above is
                    # exercised, including recurrent state and counterfactual
                    # last-action feedback, but no target reaches MAVLink.
                    pass
                elif control_mode in {
                    "attitude-rates",
                    "visual-servo-attitude",
                    "visual-servo-angles",
                    "policy-attitude",
                    "course-fsm",
                }:
                    use_closed_loop = (
                        angle_servo_closed_loop
                        and estimator.state.calibrated
                        and control_mode in {
                            "visual-servo-angles",
                            "policy-attitude",
                            "course-fsm",
                        }
                    )
                    if use_closed_loop:
                        desired_rpy = (
                            attitude_target.roll,
                            attitude_target.pitch,
                            attitude_target.yaw,
                        )
                        measured_rpy = (
                            estimator.state.roll,
                            estimator.state.pitch,
                            estimator.state.yaw,
                        )
                        if fixed_rate_attitude_mode:
                            # v3385 documents and accepts body rates directly.
                            # Close desired course angles with the gyro AHRS
                            # instead of relying on v3379's quaternion-as-rate
                            # quirk.
                            cmd_roll, cmd_pitch, cmd_yaw = attitude_body_rate_command(
                                desired_rpy,
                                measured_rpy,
                                attitude_loop_config,
                            )
                            last_sent_body_rates = (cmd_roll, cmd_pitch, cmd_yaw)
                            wire_target = AttitudeSetpoint(
                                body_roll_rate=cmd_roll,
                                body_pitch_rate=cmd_pitch,
                                body_yaw_rate=cmd_yaw,
                                thrust=attitude_target.thrust,
                            )
                            if course_publisher is None:
                                if official_reset_on_start:
                                    # The publisher's very first wire command
                                    # is policy-derived and is installed during
                                    # the countdown lead window. Deterministic
                                    # lifecycle code never contributes a timed
                                    # flight target.
                                    course_publisher = FixedRateAttitudePublisher(
                                        adapter,
                                        command_hz=args.command_hz,
                                        initial_target=wire_target,
                                        send_lock=course_send_lock,
                                    )
                                    course_publisher.start()
                                else:
                                    adapter.send_attitude_setpoint(
                                        wire_target,
                                        mode="body_rates",
                                    )
                            else:
                                course_publisher.update(wire_target)
                        else:
                            # Legacy v3379 quaternion-as-rate path.
                            cmd_roll, cmd_pitch, cmd_yaw = attitude_loop_command(
                                desired_rpy,
                                measured_rpy,
                                attitude_loop_config,
                            )
                            adapter.send_attitude_setpoint(
                                AttitudeSetpoint(
                                    roll=cmd_roll,
                                    pitch=cmd_pitch,
                                    yaw=cmd_yaw,
                                    thrust=attitude_target.thrust,
                                ),
                                mode=attitude_mode,
                            )
                    else:
                        adapter.send_attitude_setpoint(attitude_target, mode=attitude_mode)
                        if attitude_mode == "attitude":
                            estimator.set_commanded_attitude(
                                attitude_target.roll, attitude_target.pitch, attitude_target.yaw
                            )
                else:
                    adapter.send_local_ned_setpoint(target, frame=command_frame, yaw_mode=command_yaw_mode)
                if not fixed_rate_attitude_mode:
                    commands_sent += 1
                    last_command_sent_s = now_s
                next_command_s = now_s + command_period_s

                if estimator.state.calibrated and now_s - started_s >= next_trajectory_sample_s:
                    next_trajectory_sample_s = (now_s - started_s) + 0.5
                    estimator_trajectory.append(
                        {
                            "elapsed_s": round(now_s - started_s, 3),
                            "position_ned_m": tuple(
                                round(v, 3) for v in estimator.state.position_ned_m
                            ),
                            "velocity_ned_m_s": tuple(
                                round(v, 3) for v in estimator.state.velocity_ned_m_s
                            ),
                            "yaw_rad": round(estimator.state.yaw, 4),
                            "roll_rad": round(estimator.state.roll, 4),
                            "pitch_rad": round(estimator.state.pitch, 4),
                            "phase": (
                                course_controller.phase.value
                                if control_mode == "course-fsm"
                                else None
                            ),
                            "desired_rpy": (
                                round(attitude_target.roll, 4),
                                round(attitude_target.pitch, 4),
                                round(attitude_target.yaw, 4),
                            ),
                            "desired_thrust": round(attitude_target.thrust, 4),
                            "sent_body_rates": tuple(
                                round(value, 4) for value in last_sent_body_rates
                            ),
                            "gyro_rpy": (
                                round(float(adapter.telemetry.state.xgyro or 0.0), 4),
                                round(float(adapter.telemetry.state.ygyro or 0.0), 4),
                                round(float(adapter.telemetry.state.zgyro or 0.0), 4),
                            ),
                        }
                    )

            adapter.drain_telemetry(timeout_s=args.telemetry_timeout_s)

            imu_us = adapter.telemetry.state.imu_time_usec
            if imu_us is not None and imu_us != last_estimator_imu_us:
                last_estimator_imu_us = imu_us
                accel = (
                    adapter.telemetry.state.xacc,
                    adapter.telemetry.state.yacc,
                    adapter.telemetry.state.zacc,
                )
                if all(v is not None for v in accel) and estimator.state.calibrated:
                    st = adapter.telemetry.state
                    estimator.update_imu(
                        float(imu_us) * 1e-6,
                        accel,
                        (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                    )

            if args.telemetry_timeout_s <= 0.0 and (receiver is None or args.camera_timeout_s <= 0.0):
                time.sleep(args.idle_sleep_s)
    finally:
        if shadow_cadence_publisher is not None:
            shadow_cadence_publisher.stop()
        if course_publisher is not None:
            course_publisher.stop()
            commands_sent = course_publisher.commands_sent
            command_rate_violations = course_publisher.command_rate_violations
            first_command_sent_s = course_publisher.first_command_sent_s
            last_command_sent_s = course_publisher.last_command_sent_s
        # Do not leave an armed controller holding the final attitude target
        # after the attempt process exits. A later race reset must begin from a
        # stationary vehicle, not from a target retained by the flight stack.
        send_disarm = getattr(adapter, "send_disarm_command", None)
        if not policy_shadow_only and callable(send_disarm):
            try:
                with course_send_lock:
                    adapter.send_heartbeat()
                    send_disarm()
                    disarm_commands_sent += 1
            except Exception:
                pass
        if receiver is not None:
            receiver.close()
        master = getattr(adapter, "master", None)
        if master is not None:
            close_fn = getattr(master, "close", None)
            if callable(close_fn):
                close_fn()

    if course_publisher is not None and course_publisher.error is not None:
        raise RuntimeError("course setpoint publisher failed") from course_publisher.error

    # Capture flight duration before any deferred diagnostic I/O so command-rate
    # evidence measures the publisher interval, not post-flight artifact writes.
    actual_duration_s = time.monotonic() - started_s
    effective_command_hz, command_publication_duration_s = (
        calculate_effective_command_rate(
            commands_sent=commands_sent,
            first_command_sent_s=first_command_sent_s,
            last_command_sent_s=last_command_sent_s,
            fallback_duration_s=actual_duration_s,
        )
    )
    shadow_cadence_report = {
        "commands_counted": 0,
        "measurement_duration_s": 0.0,
        "effective_command_hz": 0.0,
        "command_rate_violations": 0,
        "mavlink_setpoints_sent": 0,
    }
    if shadow_cadence_publisher is not None:
        shadow_rate_hz, shadow_duration_s = calculate_effective_command_rate(
            commands_sent=shadow_cadence_publisher.commands_sent,
            first_command_sent_s=shadow_cadence_publisher.first_command_sent_s,
            last_command_sent_s=shadow_cadence_publisher.last_command_sent_s,
            fallback_duration_s=actual_duration_s,
        )
        shadow_cadence_report.update(
            {
                "commands_counted": shadow_cadence_publisher.commands_sent,
                "measurement_duration_s": round(shadow_duration_s, 6),
                "effective_command_hz": round(shadow_rate_hz, 3),
                "command_rate_violations": (
                    shadow_cadence_publisher.command_rate_violations
                ),
            }
        )
    # JPEG and JSON writes previously ran in the camera branch and reduced the
    # observed 60 Hz publisher to 48 Hz. Preserve the same first/closest/latest
    # failure evidence, but write it only after flight command publication ends.
    debug_frames = flush_debug_frames(debug_output_dir, deferred_debug_frames)

    adapter.telemetry.update_ages()
    if policy_shadow_only:
        command_kind = "shadow_policy_attitude_no_setpoint"
    elif control_mode == "attitude-rates":
        command_kind = f"{attitude_mode}_attitude_target"
    elif control_mode == "visual-servo-attitude":
        command_kind = f"{attitude_mode}_visual_servo_attitude_target"
    elif control_mode == "visual-servo-angles":
        command_kind = f"{attitude_mode}_visual_servo_angles_target"
    elif control_mode == "policy-attitude":
        command_kind = f"{attitude_mode}_policy_attitude_target"
    elif control_mode == "course-fsm":
        command_kind = f"{attitude_mode}_course_fsm_target"
    elif control_mode == "policy":
        command_kind = f"{command_frame}_{command_yaw_mode}_velocity"
    else:
        command_kind = f"{command_frame}_{command_yaw_mode}_velocity"
    sitl_report = SitlRunReport(
        endpoint=args.endpoint,
        mode="competition-smoke",
        duration_s=round(actual_duration_s, 6),
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        command_kind=command_kind,
        heartbeats_sent=heartbeats_sent,
        commands_sent=commands_sent,
        command_rate_violations=command_rate_violations,
        effective_command_hz=round(effective_command_hz, 3),
        command_publication_duration_s=round(
            command_publication_duration_s, 6
        ),
        telemetry=adapter.telemetry.metrics,
        latest_telemetry=adapter.telemetry.state,
    )
    pass_summary = pass_tracker.to_summary(started_s=started_s)
    ordered_gate_sequence_valid = pass_tracker.pass_count >= args.target_gate_count
    race_status = sitl_report.latest_telemetry.race_status
    smoke_report = CompetitionSmokeReport(
        sitl=sitl_report,
        control_mode=control_mode,
        policy_source=policy_source,
        control_inputs={
            "command_frame": command_frame,
            "command_yaw_mode": command_yaw_mode,
            "camera_uptilt_deg": round_float(
                getattr(args, "camera_uptilt_deg", 0.0)
            ),
            "arm_on_start": arm_on_start,
            "policy_shadow_only": policy_shadow_only,
            "policy_shadow_cadence_probe": shadow_cadence_report,
            "policy_shadow_calibration_timeout_s": round_float(
                policy_shadow_calibration_timeout_s
            ),
            "arm_attempts": arm_attempts,
            "arm_commands_sent": arm_commands_sent,
            "disarm_commands_sent": disarm_commands_sent,
            "official_reset_on_start": official_reset_on_start,
            "official_reset_start": official_reset_report,
            "prearm_heartbeat_timeout_s": round_float(prearm_heartbeat_timeout_s),
            "prearm_heartbeats_seen": prearm_heartbeats_seen,
            "spool_verified": spool_verified,
            "spool_wait_s": round_float(spool_wait_s),
            "actuation_verified": actuation_verified,
            "actuation_wait_s": round_float(actuation_wait_s),
            "attitude_mode": attitude_mode,
            "attitude_target": attitude_setpoint_to_dict(attitude_target),
            "angle_servo": {
                "k_pitch": round_float(getattr(args, "angle_servo_k_pitch", 0.02)),
                "max_pitch_rad": round_float(getattr(args, "angle_servo_max_pitch_rad", 0.12)),
                "k_roll": round_float(getattr(args, "angle_servo_k_roll", 0.0)),
                "max_roll_rad": round_float(getattr(args, "angle_servo_max_roll_rad", 0.10)),
                "k_yaw": round_float(getattr(args, "angle_servo_k_yaw", 0.8)),
                "max_yaw_step_rad": round_float(getattr(args, "angle_servo_max_yaw_step_rad", 0.02)),
                "blind_range_m": round_float(getattr(args, "angle_servo_blind_range_m", 4.0)),
                "blind_duration_s": round_float(getattr(args, "angle_servo_blind_duration_s", 2.0)),
                "blind_pitch_rad": round_float(getattr(args, "angle_servo_blind_pitch_rad", 0.10)),
                "search_yaw_rate_rad_s": round_float(
                    getattr(args, "angle_servo_search_yaw_rate_rad_s", 0.0)
                ),
                "state": angle_servo_state.to_summary(),
                "mode_counters": dict(angle_mode_counters),
                "map_guidance": map_guidance_state.to_summary(),
                "trace": angle_mode_trace,
            },
            "official_gate_progress": official_gate_state.to_summary(),
            "limp_abort": limp_state.to_summary(),
            "state_estimator": {
                "calibration": estimator_calibration,
                "summary": estimator.to_summary(),
                "trajectory": estimator_trajectory,
            },
            "course_fsm": {
                "policy_activate_gate_index": int(
                    getattr(args, "course_policy_activate_gate_index", 1)
                ),
                "policy_activated_elapsed_s": (
                    None
                    if course_policy_activated_s is None
                    else round_float(course_policy_activated_s - started_s)
                ),
                "policy_ticks": course_policy_ticks,
                "controller": course_controller.to_summary(),
                "guidance_detector": (
                    dataclasses.asdict(guidance_detector.metrics)
                    if guidance_detector is not None
                    else {}
                ),
                "latest_guidance": (
                    dataclasses.asdict(guidance_detection)
                    if guidance_detection is not None
                    else None
                ),
            },
            "attitude_servo": {
                "desired_standoff_m": round_float(getattr(args, "attitude_servo_desired_standoff_m", 0.0)),
                "max_pitch_rate_rad_s": round_float(getattr(args, "attitude_servo_max_pitch_rate_rad_s", 0.5)),
                "max_roll_rate_rad_s": round_float(getattr(args, "attitude_servo_max_roll_rate_rad_s", 0.4)),
                "max_yaw_rate_rad_s": round_float(getattr(args, "attitude_servo_max_yaw_rate_rad_s", 0.7)),
                "hover_thrust": round_float(getattr(args, "attitude_servo_hover_thrust", 0.58)),
                "min_thrust": round_float(getattr(args, "attitude_servo_min_thrust", 0.35)),
                "max_thrust": round_float(getattr(args, "attitude_servo_max_thrust", 0.75)),
                "k_pitch": round_float(getattr(args, "attitude_servo_k_pitch", 0.16)),
                "k_roll": round_float(getattr(args, "attitude_servo_k_roll", 0.0)),
                "k_yaw": round_float(getattr(args, "attitude_servo_k_yaw", 1.2)),
                "k_image_roll": round_float(getattr(args, "attitude_servo_k_image_roll", 0.0)),
                "k_thrust": round_float(getattr(args, "attitude_servo_k_thrust", 0.08)),
                "search_pitch_rate_rad_s": round_float(
                    getattr(args, "attitude_servo_search_pitch_rate_rad_s", 0.0)
                ),
                "search_yaw_rate_rad_s": round_float(
                    getattr(args, "attitude_servo_search_yaw_rate_rad_s", 0.0)
                ),
                "search_thrust": (
                    None
                    if getattr(args, "attitude_servo_search_thrust", None) is None
                    else round_float(getattr(args, "attitude_servo_search_thrust"))
                ),
                "forward_yaw_tolerance_rad": (
                    None
                    if getattr(args, "attitude_servo_forward_yaw_tolerance_rad", None) is None
                    else round_float(getattr(args, "attitude_servo_forward_yaw_tolerance_rad"))
                ),
                "forward_z_tolerance_m": (
                    None
                    if getattr(args, "attitude_servo_forward_z_tolerance_m", None) is None
                    else round_float(getattr(args, "attitude_servo_forward_z_tolerance_m"))
                ),
                "uncentered_forward_scale": round_float(
                    getattr(args, "attitude_servo_uncentered_forward_scale", 1.0)
                ),
                "min_control_size_px": round_float(
                    getattr(args, "attitude_servo_min_control_size_px", 0.0)
                ),
                "max_control_range_m": round_float(
                    getattr(args, "attitude_servo_max_control_range_m", 0.0)
                ),
                "min_control_confidence": round_float(
                    getattr(args, "attitude_servo_min_control_confidence", 0.0)
                ),
                "final_approach": {
                    "enabled": bool(getattr(args, "attitude_servo_final_approach", False)),
                    "trigger_range_m": round_float(
                        getattr(args, "attitude_servo_final_trigger_range_m", 6.5)
                    ),
                    "min_size_px": round_float(
                        getattr(args, "attitude_servo_final_min_size_px", 40.0)
                    ),
                    "min_confidence": round_float(
                        getattr(args, "attitude_servo_final_min_confidence", 0.45)
                    ),
                    "max_yaw_error_rad": round_float(
                        getattr(args, "attitude_servo_final_max_yaw_error_rad", 0.25)
                    ),
                    "max_abs_z_m": (
                        None
                        if getattr(args, "attitude_servo_final_max_abs_z_m", 1.0) is None
                        else round_float(getattr(args, "attitude_servo_final_max_abs_z_m", 1.0))
                    ),
                    "duration_s": round_float(
                        getattr(args, "attitude_servo_final_duration_s", 1.0)
                    ),
                    "pitch_rate_rad_s": round_float(
                        -abs(float(getattr(args, "attitude_servo_final_pitch_rate_rad_s", 0.2)))
                    ),
                    "max_yaw_rate_rad_s": round_float(
                        getattr(args, "attitude_servo_final_max_yaw_rate_rad_s", 0.35)
                    ),
                    "thrust": round_float(getattr(args, "attitude_servo_final_thrust", 0.62)),
                    "max_activations": int(getattr(args, "attitude_servo_final_max_activations", 1)),
                    "state": final_approach_state.to_summary(now_s=time.monotonic(), started_s=started_s),
                },
            },
            "policy_action_json": args.policy_action_json
            if control_mode in {"policy", "policy-attitude"}
            else "",
            "policy_final_prefer_any_edge_frame": (
                policy_final_prefer_any_edge_frame
            ),
            "policy_raw_gate_observation_index": (
                policy_raw_gate_observation_index
            ),
            "policy_raw_gate_observation_end_index": (
                policy_raw_gate_observation_end_index
            ),
            "policy_gate_transition_preseed": {
                "configured_gate_index": policy_gate_transition_preseed_index,
                "max_age_s": round_float(
                    policy_gate_transition_preseed_max_age_s),
                "min_range_margin_m": round_float(
                    policy_gate_transition_preseed_min_range_margin_m),
                "transition_preseeds": (
                    policy_gate_motion_state.transition_preseeds),
            },
            "policy_final_any_edge_delay_s": round_float(
                policy_final_any_edge_delay_s
            ),
            "policy_gate_stop": policy_gate_stop,
            "official_finish_stop": official_finish_stop,
            "official_gate_index_stop": official_gate_index_stop,
            "visual_servo": {
                "desired_standoff_m": round_float(args.visual_servo_desired_standoff_m),
                "max_forward_m_s": round_float(args.visual_servo_max_forward_m_s),
                "max_lateral_m_s": round_float(args.visual_servo_max_lateral_m_s),
                "max_vertical_m_s": round_float(args.visual_servo_max_vertical_m_s),
                "max_yaw_rate_rad_s": round_float(args.visual_servo_max_yaw_rate_rad_s),
            },
        },
        crash_detected=(
            (
                sitl_report.latest_telemetry.system_status is not None
                and int(sitl_report.latest_telemetry.system_status) in {5, 6, 7, 8}
            )
            or (
                fixed_rate_attitude_mode
                and (
                    int(sitl_report.telemetry.collisions) > 0
                    or int(estimator.state.collision_events) > 0
                )
            )
        ),
        completion_time_s=pass_summary["completion_time_s"],
        ordered_gate_passes=pass_tracker.pass_count,
        ordered_gate_sequence_valid=ordered_gate_sequence_valid,
        official_active_gate_index=(None if race_status is None else race_status.active_gate_index),
        official_last_gate_race_time=(None if race_status is None else race_status.last_gate_race_time),
        official_race_finish_time_ns=(None if race_status is None else race_status.race_finish_time_ns),
        gate_pass_summary=pass_summary,
        acceptance_inputs={
            "require_telemetry": require_telemetry,
            "require_camera": require_camera,
            "min_gate_passes": min_gate_passes,
            "max_command_rate_violations": max_command_rate_violations,
            "min_telemetry_messages": min_telemetry_messages,
            "min_camera_frames": min_camera_frames,
            "max_telemetry_dropouts": max_telemetry_dropouts,
            "min_effective_command_hz": min_effective_command_hz,
            "max_telemetry_drain_limit_hits": max_telemetry_drain_limit_hits,
            "require_official_race_progress": require_official_race_progress,
        },
        acceptance_config_path=os.path.abspath(args.acceptance_config),
        vision=vision_metrics,
        approach_diagnostics=approach_diagnostics,
        policy_trace={
            "sample_hz": policy_trace_hz,
            "max_samples": policy_trace_max_samples,
            "inference_ticks": policy_inference_ticks,
            "state_schedule": {
                **policy_state_schedule,
                "completed_steps": policy_inference_ticks,
                "realized_hz": round_float(
                    policy_inference_ticks
                    / max(float(sitl_report.duration_s), 1e-9)
                ),
            },
            "observation_fields": [
                *OBSERVATION_FIELDS[:-1],
                (
                    "official_active_gate_index_norm"
                    if getattr(args, "policy_race_phase_observation", False)
                    else OBSERVATION_FIELDS[-1]
                ),
                *(
                    PHASE_ADAPTER_OBSERVATION_FIELDS
                    if getattr(args, "policy_phase_adapter_observation", False)
                    else ()
                ),
                *(
                    GATE_PROGRESS_ADAPTER_OBSERVATION_FIELDS
                    if getattr(
                        args, "policy_gate_progress_adapter_observation", False
                    )
                    else ()
                ),
                *(
                    (
                        HYBRID_GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS
                        if getattr(
                            args,
                            "policy_hybrid_prefix_confidence_observation",
                            False,
                        )
                        else GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS
                    )
                    if getattr(
                        args,
                        "policy_gate_phase_onehot_adapter_observation",
                        False,
                    )
                    else ()
                ),
            ],
            "action_fields": list(ATTITUDE_ACTION_FIELDS),
            "motion_filter": policy_gate_motion_state.snapshot(),
            "deployment_manifest": build_deployment_manifest(policy_callable),
            "samples": policy_trace_samples,
        },
        camera_stream_metrics=(
            dataclasses.asdict(receiver.reassembler.metrics) if receiver is not None else {}
        ),
        detector_metrics=(dataclasses.asdict(detector.metrics) if detector is not None else {}),
        debug_frames=debug_frames,
    )
    smoke_report.invalid_run = bool(smoke_report.crash_detected)
    smoke_report.acceptance_passed, smoke_report.acceptance_blockers = evaluate_acceptance(
        smoke_report,
        require_telemetry=require_telemetry,
        require_camera=require_camera,
        min_gate_passes=min_gate_passes,
        max_command_rate_violations=max_command_rate_violations,
        min_telemetry_messages=min_telemetry_messages,
        min_camera_frames=min_camera_frames,
        max_telemetry_dropouts=max_telemetry_dropouts,
        min_effective_command_hz=min_effective_command_hz,
        max_telemetry_drain_limit_hits=max_telemetry_drain_limit_hits,
        require_official_race_progress=require_official_race_progress,
    )
    if policy_shadow_only:
        shadow_blockers = []
        if smoke_report.sitl.commands_sent != 0:
            shadow_blockers.append("policy_shadow_sent_control_commands")
        if arm_commands_sent != 0:
            shadow_blockers.append("policy_shadow_sent_arm_commands")
        if official_reset_report.get("reset_sent"):
            shadow_blockers.append("policy_shadow_sent_reset")
        if disarm_commands_sent != 0:
            shadow_blockers.append("policy_shadow_sent_disarm_commands")
        if not policy_trace_samples:
            shadow_blockers.append("policy_shadow_has_no_inference_samples")
        cadence = smoke_report.control_inputs["policy_shadow_cadence_probe"]
        if (
            cadence["measurement_duration_s"] >= 0.5
            and cadence["effective_command_hz"] < 50.0
        ):
            shadow_blockers.append(
                "policy_shadow_cadence_too_low:"
                f"{cadence['effective_command_hz']:.3f}<50.000"
            )
        if cadence["command_rate_violations"] != 0:
            shadow_blockers.append("policy_shadow_cadence_rate_violation")
        smoke_report.acceptance_blockers.extend(shadow_blockers)
        smoke_report.acceptance_passed = not smoke_report.acceptance_blockers
    return smoke_report


def evaluate_acceptance(
    report: CompetitionSmokeReport,
    *,
    require_telemetry: bool,
    require_camera: bool,
    min_gate_passes: int,
    max_command_rate_violations: int,
    min_telemetry_messages: int,
    min_camera_frames: int,
    max_telemetry_dropouts: int,
    min_effective_command_hz: float = 0.0,
    max_telemetry_drain_limit_hits: int = 0,
    require_official_race_progress: bool = False,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if report.sitl.command_rate_violations > max_command_rate_violations:
        blockers.append("command_rate_violations_present")
    if report.sitl.telemetry.telemetry_dropouts > max_telemetry_dropouts:
        blockers.append(
            f"telemetry_dropouts_exceeded:{report.sitl.telemetry.telemetry_dropouts}>{max_telemetry_dropouts}"
        )
    if (
        report.sitl.duration_s >= 1.0
        and report.sitl.effective_command_hz < min_effective_command_hz
    ):
        blockers.append(
            "effective_command_rate_too_low:"
            f"{report.sitl.effective_command_hz:.3f}<{min_effective_command_hz:.3f}"
        )
    if report.sitl.telemetry.drain_limit_hits > max_telemetry_drain_limit_hits:
        blockers.append(
            "telemetry_drain_limit_hits_exceeded:"
            f"{report.sitl.telemetry.drain_limit_hits}>{max_telemetry_drain_limit_hits}"
        )
    if require_telemetry and report.sitl.telemetry.messages_seen < min_telemetry_messages:
        blockers.append("no_telemetry_messages")
    if require_camera and report.vision.frames_seen < min_camera_frames:
        blockers.append("no_camera_frames")
    if report.ordered_gate_passes < min_gate_passes:
        blockers.append(
            f"insufficient_gate_passes:{report.ordered_gate_passes}<{min_gate_passes}"
        )
    if require_official_race_progress:
        if report.sitl.telemetry.race_statuses <= 0:
            blockers.append("no_official_race_status")
        elif report.official_active_gate_index is None:
            blockers.append("missing_official_active_gate_index")
        elif int(report.official_active_gate_index) < min_gate_passes:
            blockers.append(
                f"insufficient_official_gate_progress:{report.official_active_gate_index}<{min_gate_passes}"
            )
    if bool(report.crash_detected):
        blockers.append("crash_detected")
    if bool(report.invalid_run):
        blockers.append("invalid_run")
    return len(blockers) == 0, blockers


def main() -> None:
    parser = argparse.ArgumentParser(description="Competition SITL telemetry+camera smoke loop")
    parser.add_argument(
        "--acceptance-config",
        default=os.path.join("config", "sitl_competition_acceptance.json"),
        help="JSON config for frozen acceptance + gate-pass thresholds",
    )
    parser.add_argument("--endpoint", default="udpout:127.0.0.1:14540")
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--no-arm-on-start", dest="arm_on_start", action="store_false")
    parser.set_defaults(arm_on_start=True)
    parser.add_argument(
        "--policy-shadow-only",
        action="store_true",
        help=(
            "Passively build live policy-attitude observations and recurrent actions "
            "without arming, resetting, sending setpoints, or disarming. Requires "
            "--no-arm-on-start and --policy-callable."
        ),
    )
    parser.add_argument("--arm-attempts", type=int, default=3)
    parser.add_argument("--prearm-heartbeat-timeout-s", type=float, default=2.0)
    parser.add_argument("--policy-shadow-calibration-timeout-s", type=float, default=6.0)
    parser.add_argument(
        "--official-reset-on-start",
        action="store_true",
        help=(
            "For a full policy-attitude attempt, bind transport and load the "
            "policy first, then issue MAVLink 31000 and install the first "
            "policy target at the scheduled race-start boundary."
        ),
    )
    parser.add_argument("--official-reset-start-timeout-s", type=float, default=12.0)
    parser.add_argument("--official-policy-lead-s", type=float, default=0.05)
    parser.add_argument(
        "--control-mode",
        choices=[
            "visual-servo",
            "policy",
            "policy-attitude",
            "attitude-rates",
            "visual-servo-attitude",
            "visual-servo-angles",
            "course-fsm",
        ],
        default="visual-servo",
    )
    parser.add_argument("--command-frame", choices=["body_ned", "local_ned"], default="local_ned")
    parser.add_argument("--command-yaw-mode", choices=["yaw_and_rate", "ignore"], default="yaw_and_rate")
    parser.add_argument(
        "--attitude-mode",
        choices=["body_rates", "attitude", "attitude_and_rates"],
        default="body_rates",
    )
    parser.add_argument("--attitude-roll-rad", type=float, default=0.0)
    parser.add_argument("--attitude-pitch-rad", type=float, default=0.0)
    parser.add_argument("--attitude-yaw-rad", type=float, default=0.0)
    parser.add_argument("--body-roll-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--body-pitch-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--body-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-thrust", type=float, default=0.5)
    parser.add_argument("--attitude-servo-desired-standoff-m", type=float, default=0.0)
    parser.add_argument("--attitude-servo-max-pitch-rate-rad-s", type=float, default=0.5)
    parser.add_argument("--attitude-servo-max-roll-rate-rad-s", type=float, default=0.4)
    parser.add_argument("--attitude-servo-max-yaw-rate-rad-s", type=float, default=0.7)
    parser.add_argument("--attitude-servo-hover-thrust", type=float, default=0.58)
    parser.add_argument("--attitude-servo-min-thrust", type=float, default=0.35)
    parser.add_argument("--attitude-servo-max-thrust", type=float, default=0.75)
    parser.add_argument("--attitude-servo-k-pitch", type=float, default=0.16)
    parser.add_argument("--attitude-servo-k-roll", type=float, default=0.0)
    parser.add_argument("--attitude-servo-k-image-roll", type=float, default=0.0)
    parser.add_argument("--attitude-servo-k-yaw", type=float, default=1.2)
    parser.add_argument("--attitude-servo-k-thrust", type=float, default=0.08)
    parser.add_argument("--attitude-servo-search-pitch-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-servo-search-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-servo-search-thrust", type=float, default=None)
    parser.add_argument("--attitude-servo-forward-yaw-tolerance-rad", type=float, default=None)
    parser.add_argument("--attitude-servo-forward-z-tolerance-m", type=float, default=None)
    parser.add_argument("--attitude-servo-uncentered-forward-scale", type=float, default=1.0)
    parser.add_argument("--attitude-servo-min-control-size-px", type=float, default=0.0)
    parser.add_argument("--attitude-servo-max-control-range-m", type=float, default=0.0)
    parser.add_argument("--attitude-servo-min-control-confidence", type=float, default=0.0)
    parser.add_argument("--attitude-servo-final-approach", action="store_true")
    parser.add_argument("--attitude-servo-final-trigger-range-m", type=float, default=6.5)
    parser.add_argument("--attitude-servo-final-min-size-px", type=float, default=40.0)
    parser.add_argument("--attitude-servo-final-min-confidence", type=float, default=0.45)
    parser.add_argument("--attitude-servo-final-max-yaw-error-rad", type=float, default=0.25)
    parser.add_argument("--attitude-servo-final-max-abs-z-m", type=float, default=1.0)
    parser.add_argument("--attitude-servo-final-duration-s", type=float, default=1.0)
    parser.add_argument("--attitude-servo-final-pitch-rate-rad-s", type=float, default=0.2)
    parser.add_argument("--attitude-servo-final-max-yaw-rate-rad-s", type=float, default=0.35)
    parser.add_argument("--attitude-servo-final-thrust", type=float, default=0.62)
    parser.add_argument("--attitude-servo-final-max-activations", type=int, default=1)
    parser.add_argument("--angle-servo-k-pitch", type=float, default=0.02)
    parser.add_argument("--angle-servo-max-pitch-rad", type=float, default=0.12)
    parser.add_argument("--angle-servo-k-roll", type=float, default=0.0)
    parser.add_argument("--angle-servo-max-roll-rad", type=float, default=0.10)
    parser.add_argument("--angle-servo-k-yaw", type=float, default=0.8)
    parser.add_argument("--angle-servo-max-yaw-step-rad", type=float, default=0.02)
    parser.add_argument("--angle-servo-blind-range-m", type=float, default=4.0)
    parser.add_argument("--angle-servo-blind-duration-s", type=float, default=2.0)
    parser.add_argument("--angle-servo-blind-pitch-rad", type=float, default=0.10)
    parser.add_argument("--angle-servo-search-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--angle-servo-k-dx", type=float, default=0.01)
    parser.add_argument("--angle-servo-k-dz", type=float, default=0.03)
    parser.add_argument("--angle-servo-search-thrust", type=float, default=None)
    parser.add_argument("--angle-servo-track-jump-gate-m", type=float, default=5.0)
    parser.add_argument("--angle-servo-track-coast-s", type=float, default=1.5)
    parser.add_argument(
        "--policy-action-json",
        default="[0.0, 0.0, 0.0, 0.0]",
        help="Normalized action fallback when --control-mode=policy and no --policy-callable is provided",
    )
    parser.add_argument(
        "--policy-callable",
        default="",
        help="Optional callable for policy actions in '<path.py>:<function>' format",
    )
    parser.add_argument("--policy-trace-hz", type=float, default=10.0)
    parser.add_argument("--policy-trace-max-samples", type=int, default=1200)
    parser.add_argument(
        "--policy-state-hz",
        type=float,
        default=0.0,
        help=(
            "Positive fixed recurrent-state cadence. Missing ticks are caught "
            "up using the latest observation while MAVLink publication remains "
            "on its independent --command-hz schedule; zero preserves legacy "
            "one-step-per-main-loop behavior."
        ),
    )
    parser.add_argument(
        "--policy-stop-before-gate-index",
        type=int,
        default=-1,
        help=(
            "For policy-attitude diagnostics, disarm before crossing when the "
            "active gate reaches --policy-stop-forward-m; negative disables."
        ),
    )
    parser.add_argument(
        "--policy-stop-forward-m",
        type=float,
        default=0.0,
        help="Positive observable camera-forward boundary for the policy stop guard.",
    )
    parser.add_argument(
        "--policy-race-phase-observation",
        action="store_true",
        help=(
            "Replace observation[22] (previous yaw command) with normalized official "
            "active_gate_index for phase-conditioned full policies."
        ),
    )
    parser.add_argument("--policy-race-phase-denominator", type=int, default=3)
    parser.add_argument(
        "--policy-phase-adapter-observation",
        action="store_true",
        help=(
            "Append one-hot official active-gate status and phase-gated copies "
            "of existing camera pose/rate values for 32-input full policies."
        ),
    )
    parser.add_argument(
        "--policy-gate-progress-adapter-observation",
        action="store_true",
        help=(
            "Append normalized official gate progress plus eight reserved zeros "
            "for the deployable 32-input six-gate policy contract."
        ),
    )
    parser.add_argument(
        "--policy-gate-phase-onehot-adapter-observation",
        action="store_true",
        help=(
            "Append normalized official gate progress, six active-gate flags, "
            "and two reserved zeros for the 32-input six-gate policy contract."
        ),
    )
    parser.add_argument(
        "--policy-hybrid-prefix-confidence-observation",
        action="store_true",
        help=(
            "Carry exact legacy gate-pose confidence and 10.5-second elapsed "
            "fraction in reserved slots 30/31 for a 23-input prefix/32-input "
            "tail hybrid."
        ),
    )
    parser.add_argument(
        "--course-policy-activate-gate-index",
        type=int,
        default=1,
        help=(
            "With --control-mode=course-fsm and --policy-callable, hand control "
            "to the policy after this official gate and the FSM brake completes"
        ),
    )
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=DEFAULT_CAMERA_UDP_PORT)
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    parser.add_argument(
        "--camera-uptilt-deg",
        type=float,
        default=0.0,
        help=(
            "Measured UDP optical-axis uptilt relative to the policy body frame; "
            "positive is upward. VQ2 must use passive calibration evidence, not "
            "the HUD camera label."
        ),
    )
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--debug-frame-dir", default="")
    parser.add_argument("--max-detection-age-s", type=float, default=0.25)
    parser.add_argument("--visual-servo-desired-standoff-m", type=float, default=1.0)
    parser.add_argument("--visual-servo-max-forward-m-s", type=float, default=1.0)
    parser.add_argument("--visual-servo-max-lateral-m-s", type=float, default=0.5)
    parser.add_argument("--visual-servo-max-vertical-m-s", type=float, default=0.4)
    parser.add_argument("--visual-servo-max-yaw-rate-rad-s", type=float, default=0.6)
    parser.add_argument("--visual-servo-k-forward", type=float, default=0.45)
    parser.add_argument("--visual-servo-k-lateral", type=float, default=0.7)
    parser.add_argument("--visual-servo-k-vertical", type=float, default=0.7)
    parser.add_argument("--visual-servo-k-yaw", type=float, default=1.2)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--detector-require-color", action="store_true")
    parser.add_argument("--gate-inner-width-m", type=float, default=1.5)
    parser.add_argument("--gate-outer-width-m", type=float, default=2.7)
    parser.add_argument("--course-guidance-hue-low", type=int, default=75)
    parser.add_argument("--course-guidance-hue-high", type=int, default=105)
    parser.add_argument("--course-guidance-saturation-min", type=int, default=80)
    parser.add_argument("--course-guidance-value-min", type=int, default=80)
    parser.add_argument("--course-guidance-min-pixels", type=int, default=80)
    parser.add_argument("--course-guidance-min-supporting-rows", type=int, default=8)
    parser.add_argument("--course-guidance-max-age-s", type=float, default=0.3)
    parser.add_argument(
        "--course-controller-config",
        default="config/course_fsm_defaults.json",
        help="Single JSON source for course-FSM tuning values",
    )
    parser.add_argument("--course-hover-thrust", type=float, default=None)
    parser.add_argument("--course-min-thrust", type=float, default=None)
    parser.add_argument("--course-max-thrust", type=float, default=None)
    parser.add_argument("--course-guidance-pitch-rad", type=float, default=None)
    parser.add_argument("--course-guidance-yaw-gain", type=float, default=None)
    parser.add_argument("--course-guidance-max-yaw-step-rad", type=float, default=None)
    parser.add_argument("--course-guidance-search-rate-rad-s", type=float, default=None)
    parser.add_argument("--course-align-yaw-tolerance-rad", type=float, default=None)
    parser.add_argument("--course-align-vertical-tolerance-m", type=float, default=None)
    parser.add_argument("--course-align-ticks", type=int, default=None)
    parser.add_argument("--course-approach-pitch-rad", type=float, default=None)
    parser.add_argument("--course-commit-range-m", type=float, default=None)
    parser.add_argument("--course-commit-pitch-rad", type=float, default=None)
    parser.add_argument("--course-commit-duration-s", type=float, default=None)
    parser.add_argument("--course-brake-pitch-rad", type=float, default=None)
    parser.add_argument("--course-brake-duration-s", type=float, default=None)
    parser.add_argument("--course-brake-settle-s", type=float, default=None)
    parser.add_argument(
        "--angle-servo-closed-loop",
        dest="angle_servo_closed_loop",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-angle-servo-closed-loop",
        dest="angle_servo_closed_loop",
        action="store_false",
    )
    parser.add_argument("--angle-servo-max-speed-m-s", type=float, default=2.0)
    parser.add_argument("--angle-servo-k-speed", type=float, default=0.06)
    parser.add_argument("--angle-servo-k-vz", type=float, default=0.02)
    parser.add_argument("--angle-servo-recovery-trigger-rad", type=float, default=0.9)
    parser.add_argument("--angle-servo-recovery-release-rad", type=float, default=0.25)
    parser.add_argument("--angle-servo-carry-through-m", type=float, default=2.0)
    parser.add_argument("--angle-servo-carry-freeze-range-m", type=float, default=3.0)
    parser.add_argument("--map-guidance-max-fix-age-s", type=float, default=2.0)
    parser.add_argument("--landmark-max-range-m", type=float, default=35.0)
    parser.add_argument("--angle-servo-k-ithrust", type=float, default=0.02)
    parser.add_argument("--angle-servo-max-bz-m", type=float, default=2.5)
    parser.add_argument("--limp-recovery-timeout-s", type=float, default=3.0)
    parser.add_argument("--limp-max-speed-m-s", type=float, default=0.15)
    parser.add_argument("--limp-collision-grace-s", type=float, default=2.0)
    parser.add_argument("--limp-abort-exit", action="store_true", default=True)
    parser.add_argument("--no-limp-abort-exit", dest="limp_abort_exit", action="store_false")
    parser.add_argument("--limp-auto-reset", action="store_true")
    parser.add_argument("--max-approach-diagnostic-samples", type=int, default=12)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--stop-after-official-gate-index", type=int, default=-1)
    parser.add_argument("--require-official-race-progress", action="store_true")
    parser.add_argument("--gate-confidence-arm-min", type=float, default=None)
    parser.add_argument("--gate-confidence-pass-min", type=float, default=None)
    parser.add_argument("--gate-pass-arm-range-m", type=float, default=None)
    parser.add_argument("--gate-pass-range-m", type=float, default=None)
    parser.add_argument("--gate-pass-rearm-range-m", type=float, default=None)
    parser.add_argument("--gate-pass-min-consecutive-frames", type=int, default=None)
    parser.add_argument("--gate-pass-max-missed-frames", type=int, default=None)
    parser.add_argument("--gate-pass-cooldown-s", type=float, default=None)
    parser.add_argument("--require-telemetry", action="store_true")
    parser.add_argument("--require-camera", action="store_true")
    parser.add_argument("--min-gate-passes", type=int, default=None)
    parser.add_argument("--max-command-rate-violations", type=int, default=None)
    parser.add_argument("--min-telemetry-messages", type=int, default=None)
    parser.add_argument("--min-camera-frames", type=int, default=None)
    parser.add_argument("--max-telemetry-dropouts", type=int, default=None)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--csv-path", default="")
    args = parser.parse_args()
    observation_modes = sum(
        bool(value)
        for value in (
            args.policy_race_phase_observation,
            args.policy_phase_adapter_observation,
            args.policy_gate_progress_adapter_observation,
            args.policy_gate_phase_onehot_adapter_observation,
        )
    )
    if observation_modes > 1:
        parser.error(
            "choose only one policy phase/progress observation adapter"
        )
    if (
        args.policy_hybrid_prefix_confidence_observation
        and not args.policy_gate_phase_onehot_adapter_observation
    ):
        parser.error(
            "--policy-hybrid-prefix-confidence-observation requires "
            "--policy-gate-phase-onehot-adapter-observation"
        )
    if (
        args.min_gate_passes is not None
        and args.min_gate_passes <= 0
        and not args.policy_shadow_only
    ):
        raise ValueError("min_gate_passes must be positive")
    if args.stop_after_official_gate_index < -1:
        raise ValueError("stop_after_official_gate_index must be -1 or nonnegative")

    smoke_report = run_smoke(args)
    if args.json_path:
        # Keep JSON output compatible with the existing SITL helper convention.
        directory = os.path.dirname(args.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(smoke_report.to_dict(), f, indent=2, sort_keys=True)
    maybe_write_csv(args.csv_path, smoke_report)
    print(json.dumps(smoke_report.to_dict(), indent=2, sort_keys=True))
    if not smoke_report.acceptance_passed:
        raise SystemExit(
            "smoke acceptance failed: " + ", ".join(smoke_report.acceptance_blockers)
        )


if __name__ == "__main__":
    main()
