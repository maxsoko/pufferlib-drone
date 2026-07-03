#!/usr/bin/env python3
"""Competition-facing SITL smoke loop for telemetry + camera + controller wiring."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from drone_camera_receiver import DEFAULT_CAMERA_UDP_PORT, UdpCameraReceiver
from drone_gate_detector import GateDetection, SquareGateDetector, cv2, np
from drone_policy_contract import (
    OBSERVATION_SIZE,
    decode_policy_action,
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
from drone_visual_servo import estimate_gate_pose_from_corners, visual_servo_command


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    out = float(value)
    return out if math.isfinite(out) else float(default)


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


def build_policy_observation(
    telemetry: TelemetryState,
    *,
    gate_detection: GateDetection | None,
    gate_pose,
    elapsed_fraction: float,
    last_cmd_norm: Sequence[float],
) -> tuple[float, ...]:
    roll = safe_float(telemetry.roll)
    pitch = safe_float(telemetry.pitch)
    yaw = safe_float(telemetry.yaw)
    quat = roll_pitch_yaw_to_quat(roll, pitch, yaw)

    gate_visible = 1.0 if gate_pose is not None else -1.0
    gate_forward = 0.0
    gate_right = 0.0
    gate_down = 0.0
    gate_yaw_error = 0.0
    gate_pitch_error = 0.0
    gate_size = -1.0
    gate_confidence = -1.0
    if gate_pose is not None:
        gx, gy, gz = gate_pose.body_vector_ned_m
        gate_forward = clamp(gx / 10.0, -1.0, 1.0)
        gate_right = clamp(gy / 5.0, -1.0, 1.0)
        gate_down = clamp(gz / 5.0, -1.0, 1.0)
        gate_yaw_error = clamp(gate_pose.yaw_error_rad / (math.pi * 0.5), -1.0, 1.0)
        pitch_error = math.atan2(gz, max(gx, 1e-6))
        gate_pitch_error = clamp(pitch_error / (math.pi * 0.5), -1.0, 1.0)
        gate_confidence = clamp(float(gate_pose.confidence), 0.0, 1.0)
    if gate_detection is not None:
        max_dim = max(gate_detection.bounding_width_px, gate_detection.bounding_height_px, 1.0)
        gate_size = clamp(max_dim / 640.0, -1.0, 1.0)

    observation = (
        clamp(0.0, -1.0, 1.0),  # body_vel_forward_norm (placeholder until TS-002 velocity field is confirmed)
        clamp(0.0, -1.0, 1.0),  # body_vel_right_norm
        clamp(0.0, -1.0, 1.0),  # body_vel_down_norm
        clamp(safe_float(telemetry.rollspeed) / 4.0, -1.0, 1.0),
        clamp(safe_float(telemetry.pitchspeed) / 4.0, -1.0, 1.0),
        clamp(safe_float(telemetry.yawspeed) / 4.0, -1.0, 1.0),
        clamp(quat[0], -1.0, 1.0),
        clamp(quat[1], -1.0, 1.0),
        clamp(quat[2], -1.0, 1.0),
        clamp(quat[3], -1.0, 1.0),
        gate_visible,
        gate_forward,
        gate_right,
        gate_down,
        gate_yaw_error,
        gate_pitch_error,
        gate_size,
        gate_confidence,
        clamp(float(elapsed_fraction), -1.0, 1.0),
        clamp(float(last_cmd_norm[0]), -1.0, 1.0),
        clamp(float(last_cmd_norm[1]), -1.0, 1.0),
        clamp(float(last_cmd_norm[2]), -1.0, 1.0),
        clamp(float(last_cmd_norm[3]), -1.0, 1.0),
    )
    if len(observation) != OBSERVATION_SIZE:
        raise RuntimeError(f"observation size mismatch: expected {OBSERVATION_SIZE}, got {len(observation)}")
    return validate_observation(observation)


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
    return function


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
    roll_rate = clamp(k_roll * by, -max_roll_rate, max_roll_rate)
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
    if min_thrust > max_thrust:
        raise ValueError("attitude_servo_min_thrust must be <= attitude_servo_max_thrust")

    pitch_rate = -abs(float(getattr(args, "attitude_servo_final_pitch_rate_rad_s", 0.2)))
    max_yaw_rate = abs(float(getattr(args, "attitude_servo_final_max_yaw_rate_rad_s", 0.35)))
    k_yaw = float(getattr(args, "attitude_servo_k_yaw", 1.2))
    yaw_rate = 0.0 if pose is None else clamp(k_yaw * pose.yaw_error_rad, -max_yaw_rate, max_yaw_rate)
    thrust = clamp(float(getattr(args, "attitude_servo_final_thrust", 0.62)), min_thrust, max_thrust)
    return AttitudeSetpoint(
        body_pitch_rate=pitch_rate,
        body_yaw_rate=yaw_rate,
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
    arm_attempts = int(getattr(args, "arm_attempts", 3))
    prearm_heartbeat_timeout_s = float(getattr(args, "prearm_heartbeat_timeout_s", 2.0))
    if arm_attempts < 0:
        raise ValueError("arm_attempts must be non-negative")
    if prearm_heartbeat_timeout_s < 0.0:
        raise ValueError("prearm_heartbeat_timeout_s must be non-negative")
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
    require_official_race_progress = bool(args.require_official_race_progress) or bool(
        resolve_config_value(acceptance_config, "smoke_defaults", "require_official_race_progress", False)
    )

    control_mode = getattr(args, "control_mode", "visual-servo")
    if control_mode not in {"visual-servo", "policy", "attitude-rates", "visual-servo-attitude"}:
        raise ValueError(
            "control_mode must be 'visual-servo', 'policy', 'attitude-rates', or 'visual-servo-attitude'"
        )

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
    elif control_mode == "attitude-rates":
        policy_source = "constant_attitude_rates"
    elif control_mode == "visual-servo-attitude":
        policy_source = "visual_servo_attitude_rates"

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    receiver = None
    detector = None
    if not args.no_camera:
        receiver = UdpCameraReceiver(
            host=args.camera_host,
            port=args.camera_port,
            socket_timeout_s=args.camera_timeout_s,
        )
        detector = SquareGateDetector(
            min_area_px=args.detector_min_area_px,
            max_aspect_error=args.detector_max_aspect_error,
            min_fill_ratio=args.detector_min_fill_ratio,
            allow_grayscale_fallback=not bool(getattr(args, "detector_require_color", False)),
        )

    heartbeat_period_s = 1.0 / args.heartbeat_hz
    command_period_s = 1.0 / args.command_hz
    next_heartbeat_s = 0.0
    next_command_s = 0.0
    command_rate_violations = 0
    heartbeats_sent = 0
    commands_sent = 0
    last_command_sent_s = None
    last_detection_s = None
    last_control_detection_s = None
    gate_detection = None
    gate_pose = None
    control_gate_detection = None
    control_gate_pose = None
    vision_metrics = SmokeVisionMetrics()
    approach_diagnostics = ApproachDiagnostics()
    final_approach_state = AttitudeFinalApproachState()
    debug_frame_dir = str(getattr(args, "debug_frame_dir", "") or "")
    debug_output_dir = Path(debug_frame_dir) if debug_frame_dir else None
    debug_frames = {}
    debug_closest_range_m = None
    last_cmd_norm = (0.0, 0.0, 0.0, 0.0)
    pass_tracker = VisionGatePassTracker(
        config=gate_pass_config,
        target_gate_count=args.target_gate_count,
    )
    arm_commands_sent = 0
    prearm_heartbeats_seen = 0

    if arm_on_start:
        prearm_deadline_s = time.monotonic() + prearm_heartbeat_timeout_s
        while adapter.telemetry.metrics.heartbeats <= 0 and time.monotonic() < prearm_deadline_s:
            adapter.poll_telemetry(timeout_s=min(0.05, max(0.0, prearm_deadline_s - time.monotonic())))
        prearm_heartbeats_seen = adapter.telemetry.metrics.heartbeats
        for _attempt in range(arm_attempts):
            adapter.send_heartbeat()
            heartbeats_sent += 1
            adapter.send_arm_command()
            arm_commands_sent += 1
            adapter.poll_telemetry(timeout_s=0.0)
            time.sleep(min(0.05, heartbeat_period_s))

    started_s = time.monotonic()
    deadline_s = started_s + args.duration

    try:
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if now_s >= next_heartbeat_s:
                adapter.send_heartbeat()
                heartbeats_sent += 1
                next_heartbeat_s = now_s + heartbeat_period_s

            if receiver is not None and detector is not None:
                frames = receiver.poll_frames(max_packets=args.camera_max_packets_per_loop)
                if frames:
                    frame = frames[-1]
                    vision_metrics.frames_seen += 1
                    detection = detector.detect_jpeg(frame.jpeg)
                    gate_detection = detection
                    vision_metrics.detector_decode_failures = detector.metrics.decode_failures
                    vision_metrics.detector_no_quad_found = detector.metrics.no_quad_found
                    vision_metrics.detector_detections = detector.metrics.detections
                    tracked_pose = None
                    if detection is not None:
                        tracked_pose = estimate_gate_pose_from_corners(detection.corners)
                        gate_pose = tracked_pose
                        vision_metrics.last_detection_confidence = float(detection.confidence)
                        last_detection_s = now_s
                        if control_mode == "visual-servo-attitude" and attitude_control_pose_allowed(
                            tracked_pose,
                            detection,
                            args,
                        ):
                            control_gate_pose = tracked_pose
                            control_gate_detection = detection
                            last_control_detection_s = now_s
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
                            if "first_detection" not in debug_frames:
                                debug_frames["first_detection"] = save_debug_frame(
                                    debug_output_dir, frame, detection, "first_detection"
                                )
                            range_m = float(tracked_pose.range_camera_m)
                            if debug_closest_range_m is None or range_m < debug_closest_range_m:
                                debug_closest_range_m = range_m
                                debug_frames["closest_detection"] = save_debug_frame(
                                    debug_output_dir, frame, detection, "closest_detection"
                                )
                            debug_frames["latest_detection"] = save_debug_frame(
                                debug_output_dir, frame, detection, "latest_detection"
                            )
                    pass_tracker.observe(
                        now_s=now_s,
                        gate_pose=tracked_pose,
                        detection_confidence=(None if detection is None else detection.confidence),
                        detection_present=(detection is not None),
                    )
                    update_timesync_metrics(
                        vision_metrics,
                        sim_time_ns=frame.sim_time_ns,
                        timesync_ts1=adapter.telemetry.state.timesync_ts1,
                    )

            if last_detection_s is not None:
                vision_metrics.last_detection_age_s = max(0.0, now_s - last_detection_s)

            if now_s >= next_command_s:
                if last_command_sent_s is not None and now_s - last_command_sent_s < 0.01:
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
                elif control_mode == "policy":
                    elapsed_fraction = clamp((now_s - started_s) / args.duration, 0.0, 1.0)
                    observation = build_policy_observation(
                        telemetry,
                        gate_detection=gate_detection,
                        gate_pose=gate_pose,
                        elapsed_fraction=elapsed_fraction,
                        last_cmd_norm=last_cmd_norm,
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
                        clamp(attitude_target.thrust * 2.0 - 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_yaw_rate / 1.0, -1.0, 1.0),
                    )
                else:
                    last_cmd_norm = (
                        clamp(attitude_target.body_pitch_rate / 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_roll_rate / 1.0, -1.0, 1.0),
                        clamp(attitude_target.thrust * 2.0 - 1.0, -1.0, 1.0),
                        clamp(attitude_target.body_yaw_rate / 1.0, -1.0, 1.0),
                    )

                if control_mode in {"attitude-rates", "visual-servo-attitude"}:
                    adapter.send_attitude_setpoint(attitude_target, mode=attitude_mode)
                else:
                    adapter.send_local_ned_setpoint(target, frame=command_frame, yaw_mode=command_yaw_mode)
                commands_sent += 1
                last_command_sent_s = now_s
                next_command_s = now_s + command_period_s

            adapter.poll_telemetry(timeout_s=args.telemetry_timeout_s)
            if args.telemetry_timeout_s <= 0.0 and (receiver is None or args.camera_timeout_s <= 0.0):
                time.sleep(args.idle_sleep_s)
    finally:
        if receiver is not None:
            receiver.close()
        master = getattr(adapter, "master", None)
        if master is not None:
            close_fn = getattr(master, "close", None)
            if callable(close_fn):
                close_fn()

    adapter.telemetry.update_ages()
    if control_mode == "attitude-rates":
        command_kind = f"{attitude_mode}_attitude_target"
    elif control_mode == "visual-servo-attitude":
        command_kind = f"{attitude_mode}_visual_servo_attitude_target"
    else:
        command_kind = f"{command_frame}_{command_yaw_mode}_velocity"
    sitl_report = SitlRunReport(
        endpoint=args.endpoint,
        mode="competition-smoke",
        duration_s=round(time.monotonic() - started_s, 6),
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        command_kind=command_kind,
        heartbeats_sent=heartbeats_sent,
        commands_sent=commands_sent,
        command_rate_violations=command_rate_violations,
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
            "arm_on_start": arm_on_start,
            "arm_attempts": arm_attempts,
            "arm_commands_sent": arm_commands_sent,
            "prearm_heartbeat_timeout_s": round_float(prearm_heartbeat_timeout_s),
            "prearm_heartbeats_seen": prearm_heartbeats_seen,
            "attitude_mode": attitude_mode,
            "attitude_target": attitude_setpoint_to_dict(attitude_target),
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
            "policy_action_json": args.policy_action_json if control_mode == "policy" else "",
            "visual_servo": {
                "desired_standoff_m": round_float(args.visual_servo_desired_standoff_m),
                "max_forward_m_s": round_float(args.visual_servo_max_forward_m_s),
                "max_lateral_m_s": round_float(args.visual_servo_max_lateral_m_s),
                "max_vertical_m_s": round_float(args.visual_servo_max_vertical_m_s),
                "max_yaw_rate_rad_s": round_float(args.visual_servo_max_yaw_rate_rad_s),
            },
        },
        crash_detected=(
            sitl_report.latest_telemetry.system_status is not None
            and int(sitl_report.latest_telemetry.system_status) in {5, 6, 7, 8}
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
            "require_official_race_progress": require_official_race_progress,
        },
        acceptance_config_path=os.path.abspath(args.acceptance_config),
        vision=vision_metrics,
        approach_diagnostics=approach_diagnostics,
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
        require_official_race_progress=require_official_race_progress,
    )
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
    require_official_race_progress: bool = False,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if report.sitl.command_rate_violations > max_command_rate_violations:
        blockers.append("command_rate_violations_present")
    if report.sitl.telemetry.telemetry_dropouts > max_telemetry_dropouts:
        blockers.append(
            f"telemetry_dropouts_exceeded:{report.sitl.telemetry.telemetry_dropouts}>{max_telemetry_dropouts}"
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
    parser.add_argument("--arm-attempts", type=int, default=3)
    parser.add_argument("--prearm-heartbeat-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--control-mode",
        choices=["visual-servo", "policy", "attitude-rates", "visual-servo-attitude"],
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
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=DEFAULT_CAMERA_UDP_PORT)
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
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
    parser.add_argument("--max-approach-diagnostic-samples", type=int, default=12)
    parser.add_argument("--target-gate-count", type=int, default=1)
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
    if args.min_gate_passes is not None and args.min_gate_passes <= 0:
        raise ValueError("min_gate_passes must be positive")

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
