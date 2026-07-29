#!/usr/bin/env python3
"""Minimal camera-only course controller for AI-GP Simulator v3379.

The controller deliberately avoids global position, dead reckoning, landmark
maps, and learned policies.  It follows the cyan course-guidance structure,
uses the red gate pose for the precision approach, and uses official race
status as the only authoritative gate-pass signal.

State order:

    FOLLOW_GUIDANCE -> ALIGN_GATE -> APPROACH_GATE -> COMMIT_GATE
             ^                                      |
             +--------------- BRAKE ----------------+

All outputs are desired absolute attitude angles plus normalized thrust.  The
SITL runner closes those desired angles with the gyro because v3379 treats the
SET_ATTITUDE_TARGET quaternion as a rate demand.
"""

from __future__ import annotations

import dataclasses
import enum
import math

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime fallback
    cv2 = None
    np = None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclasses.dataclass(frozen=True)
class GuidanceDetection:
    """Image-space center of the visible cyan course corridor."""

    center_x_norm: float
    heading_error_rad: float
    confidence: float
    supporting_rows: int
    pixel_count: int


@dataclasses.dataclass
class GuidanceDetectorMetrics:
    frames_seen: int = 0
    detections: int = 0
    decode_failures: int = 0
    insufficient_pixels: int = 0


class CyanGuidanceDetector:
    """Detect the centerline between the simulator's cyan guidance rails.

    The implementation intentionally uses only a color mask and robust row
    medians.  It does not reconstruct a world-space course or fit a fragile
    geometric model.  A detection is useful when both rails are visible in a
    meaningful fraction of the lower two-thirds of the image.
    """

    def __init__(
        self,
        *,
        hue_low: int = 75,
        hue_high: int = 105,
        saturation_min: int = 80,
        value_min: int = 80,
        roi_top_fraction: float = 0.28,
        min_pixels: int = 80,
        min_supporting_rows: int = 8,
        focal_length_px: float = 320.0,
    ) -> None:
        if not 0 <= hue_low <= hue_high <= 179:
            raise ValueError("cyan hue bounds must satisfy 0 <= low <= high <= 179")
        if not 0.0 <= roi_top_fraction < 1.0:
            raise ValueError("roi_top_fraction must be in [0, 1)")
        if min_pixels <= 0 or min_supporting_rows <= 0:
            raise ValueError("guidance support thresholds must be positive")
        self.hue_low = int(hue_low)
        self.hue_high = int(hue_high)
        self.saturation_min = int(saturation_min)
        self.value_min = int(value_min)
        self.roi_top_fraction = float(roi_top_fraction)
        self.min_pixels = int(min_pixels)
        self.min_supporting_rows = int(min_supporting_rows)
        self.focal_length_px = float(focal_length_px)
        self.metrics = GuidanceDetectorMetrics()
        self.available = cv2 is not None and np is not None

    def detect_jpeg(self, jpeg: bytes) -> GuidanceDetection | None:
        self.metrics.frames_seen += 1
        if not self.available:
            self.metrics.decode_failures += 1
            return None
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.metrics.decode_failures += 1
            return None
        return self.detect_bgr(image, count_frame=False)

    def detect_bgr(self, image, *, count_frame: bool = True) -> GuidanceDetection | None:
        if count_frame:
            self.metrics.frames_seen += 1
        if not self.available or image is None or len(image.shape) != 3:
            self.metrics.decode_failures += 1
            return None

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [self.hue_low, self.saturation_min, self.value_min], dtype=np.uint8
        )
        upper = np.array([self.hue_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        roi_top = int(round(height * self.roi_top_fraction))
        roi = mask[roi_top:, :]
        pixel_count = int(cv2.countNonZero(roi))
        if pixel_count < self.min_pixels:
            self.metrics.insufficient_pixels += 1
            return None

        image_center = 0.5 * (width - 1)
        row_centers: list[tuple[float, float]] = []
        # Sampling every other row reduces work and suppresses single-row JPEG
        # artifacts. Use robust outer percentiles instead of requiring the two
        # rails to straddle the optical center: during a turn, the complete
        # corridor can legitimately sit on one side of the image.
        for y in range(roi_top, height, 2):
            xs = np.flatnonzero(mask[y])
            if xs.size < 2:
                continue
            left_x = float(np.percentile(xs, 10))
            right_x = float(np.percentile(xs, 90))
            if right_x - left_x < max(8.0, width * 0.025):
                continue
            row_centers.append((float(y), 0.5 * (left_x + right_x)))

        if len(row_centers) < self.min_supporting_rows:
            self.metrics.insufficient_pixels += 1
            return None

        # Rows nearer the horizon describe upcoming turns; lower rows describe
        # immediate corridor centering.  Blend both rather than fitting lines.
        centers = np.asarray([center for _y, center in row_centers], dtype=float)
        ys = np.asarray([y for y, _center in row_centers], dtype=float)
        upper_cut = np.percentile(ys, 45)
        lower_cut = np.percentile(ys, 70)
        upper_centers = centers[ys <= upper_cut]
        lower_centers = centers[ys >= lower_cut]
        upper_center = float(np.median(upper_centers))
        lower_center = float(np.median(lower_centers))
        target_x = 0.65 * upper_center + 0.35 * lower_center

        offset_px = target_x - image_center
        center_x_norm = clamp(offset_px / max(image_center, 1.0), -1.0, 1.0)
        heading_error = math.atan2(offset_px, max(self.focal_length_px, 1.0))
        row_fraction = len(row_centers) / max(1.0, (height - roi_top) / 2.0)
        pixel_fraction = pixel_count / max(1.0, roi.size)
        confidence = clamp(0.75 * row_fraction + 6.0 * pixel_fraction, 0.0, 1.0)

        self.metrics.detections += 1
        return GuidanceDetection(
            center_x_norm=center_x_norm,
            heading_error_rad=heading_error,
            confidence=confidence,
            supporting_rows=len(row_centers),
            pixel_count=pixel_count,
        )


class CoursePhase(str, enum.Enum):
    FOLLOW_GUIDANCE = "follow_guidance"
    ALIGN_GATE = "align_gate"
    APPROACH_GATE = "approach_gate"
    COMMIT_GATE = "commit_gate"
    BRAKE = "brake"


@dataclasses.dataclass(frozen=True)
class CourseCommand:
    roll: float
    pitch: float
    yaw: float
    thrust: float
    phase: str


@dataclasses.dataclass
class CourseControllerConfig:
    # Gate-1-proven vertical band.  Keep this conservative until a clean
    # post-pass hover is demonstrated; then tune only through config values.
    hover_thrust: float = 0.27
    min_thrust: float = 0.18
    max_thrust: float = 0.42
    approach_thrust: float | None = None
    vertical_image_kp: float = 0.0
    later_gate_vertical_image_kp: float | None = None
    vertical_image_control_range_m: float = 0.0
    far_gate_thrust: float | None = None
    vertical_kp: float = 0.035
    max_vertical_error_m: float = 2.0
    vertical_filter_alpha: float = 0.15

    guidance_min_confidence: float = 0.12
    guidance_pitch_rad: float = -0.025
    guidance_yaw_gain: float = 1.0
    guidance_max_yaw_step_rad: float = 0.20
    guidance_lost_search_rate_rad_s: float = 0.25
    guidance_lost_pitch_rad: float = 0.0

    gate_min_confidence: float = 0.25
    gate_max_acquire_range_m: float = 35.0
    first_gate_max_acquire_range_m: float | None = None
    gate_yaw_gain: float = 0.9
    first_gate_yaw_gain: float | None = None
    later_gate_roll_gain: float = 0.0
    later_gate_align_pitch_rad: float = 0.0
    gate_max_yaw_step_rad: float = 0.18
    align_yaw_tolerance_rad: float = 0.12
    later_gate_align_yaw_tolerance_rad: float | None = None
    align_vertical_tolerance_m: float = 0.65
    align_consecutive_ticks: int = 8
    gate_lost_timeout_s: float = 0.5

    approach_pitch_rad: float = -0.045
    first_gate_approach_pitch_rad: float | None = None
    approach_max_pitch_rad: float = 0.06
    commit_range_m: float = 4.0
    commit_pitch_rad: float = -0.06
    commit_duration_s: float = 1.2

    brake_pitch_rad: float = 0.08
    brake_duration_s: float = 1.5
    brake_settle_s: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_thrust <= self.hover_thrust <= self.max_thrust <= 1.0:
            raise ValueError("course thrust must satisfy 0 <= min <= hover <= max <= 1")
        if self.approach_thrust is not None and not (
            self.min_thrust <= self.approach_thrust <= self.max_thrust
        ):
            raise ValueError("approach_thrust must be within the configured thrust limits")
        if self.far_gate_thrust is not None and not (
            self.min_thrust <= self.far_gate_thrust <= self.max_thrust
        ):
            raise ValueError("far_gate_thrust must be within the configured thrust limits")
        if self.vertical_image_kp < 0.0:
            raise ValueError("vertical_image_kp must be non-negative")
        if self.later_gate_vertical_image_kp is not None and self.later_gate_vertical_image_kp < 0.0:
            raise ValueError("later_gate_vertical_image_kp must be non-negative")
        if self.align_consecutive_ticks <= 0:
            raise ValueError("align_consecutive_ticks must be positive")
        if not 0.0 < self.vertical_filter_alpha <= 1.0:
            raise ValueError("vertical_filter_alpha must be in (0, 1]")
        if self.commit_range_m <= 0.0:
            raise ValueError("commit_range_m must be positive")
        if self.commit_duration_s <= 0.0:
            raise ValueError("commit_duration_s must be positive")
        if self.brake_duration_s < 0.0 or self.brake_settle_s < 0.0:
            raise ValueError("brake durations must be non-negative")


@dataclasses.dataclass
class CourseController:
    config: CourseControllerConfig = dataclasses.field(default_factory=CourseControllerConfig)
    phase: CoursePhase = CoursePhase.FOLLOW_GUIDANCE
    phase_started_s: float | None = None
    last_official_gate_index: int | None = None
    last_gate_seen_s: float | None = None
    last_guidance_seen_s: float | None = None
    aligned_ticks: int = 0
    hold_yaw_rad: float = 0.0
    last_update_s: float | None = None
    filtered_vertical_error_m: float | None = None
    transitions: list[dict] = dataclasses.field(default_factory=list)
    command_counts: dict[str, int] = dataclasses.field(default_factory=dict)

    def _transition(self, phase: CoursePhase, now_s: float, reason: str) -> None:
        if phase == self.phase:
            return
        self.transitions.append(
            {
                "time_s": round(float(now_s), 3),
                "from": self.phase.value,
                "to": phase.value,
                "reason": reason,
            }
        )
        self.phase = phase
        self.phase_started_s = float(now_s)
        self.aligned_ticks = 0

    def _record(self, command: CourseCommand) -> CourseCommand:
        self.command_counts[command.phase] = self.command_counts.get(command.phase, 0) + 1
        return command

    def _thrust_for_gate(self, gate_pose) -> float:
        if self.config.approach_thrust is not None:
            if self.phase in {CoursePhase.ALIGN_GATE, CoursePhase.APPROACH_GATE}:
                base_thrust = self.config.approach_thrust
            else:
                base_thrust = self.config.hover_thrust
            image_center = getattr(gate_pose, "image_center_px", None)
            image_control_in_range = bool(
                gate_pose is not None
                and (
                    self.config.vertical_image_control_range_m <= 0.0
                    or float(gate_pose.range_camera_m)
                    <= self.config.vertical_image_control_range_m
                )
            )
            if not image_control_in_range and self.config.far_gate_thrust is not None:
                base_thrust = self.config.far_gate_thrust
            if (
                image_center is not None
                and self._active_vertical_image_kp() > 0.0
                and image_control_in_range
            ):
                vertical_norm = clamp((float(image_center[1]) - 180.0) / 180.0, -1.0, 1.0)
                base_thrust -= self._active_vertical_image_kp() * vertical_norm
            return clamp(base_thrust, self.config.min_thrust, self.config.max_thrust)
        if gate_pose is None:
            return self.config.hover_thrust
        _forward, _right, down = gate_pose.body_vector_ned_m
        raw_error = clamp(
            float(down),
            -self.config.max_vertical_error_m,
            self.config.max_vertical_error_m,
        )
        if self.filtered_vertical_error_m is None:
            self.filtered_vertical_error_m = raw_error
        else:
            alpha = self.config.vertical_filter_alpha
            self.filtered_vertical_error_m += alpha * (
                raw_error - self.filtered_vertical_error_m
            )
        correction = -self.config.vertical_kp * self.filtered_vertical_error_m
        return clamp(
            self.config.hover_thrust + correction,
            self.config.min_thrust,
            self.config.max_thrust,
        )

    def _gate_is_usable(self, gate_pose, gate_confidence: float) -> bool:
        max_range_m = self.config.gate_max_acquire_range_m
        if (
            self.config.first_gate_max_acquire_range_m is not None
            and (self.last_official_gate_index is None or self.last_official_gate_index <= 0)
        ):
            max_range_m = self.config.first_gate_max_acquire_range_m
        return bool(
            gate_pose is not None
            and float(gate_confidence) >= self.config.gate_min_confidence
            and 0.0 < float(gate_pose.range_camera_m) <= max_range_m
        )

    def _active_gate_yaw_gain(self) -> float:
        if (
            self.config.first_gate_yaw_gain is not None
            and (self.last_official_gate_index is None or self.last_official_gate_index <= 0)
        ):
            return self.config.first_gate_yaw_gain
        return self.config.gate_yaw_gain

    def _active_approach_pitch_rad(self) -> float:
        if (
            self.config.first_gate_approach_pitch_rad is not None
            and (self.last_official_gate_index is None or self.last_official_gate_index <= 0)
        ):
            return self.config.first_gate_approach_pitch_rad
        return self.config.approach_pitch_rad

    def _active_align_yaw_tolerance_rad(self) -> float:
        if (
            self.config.later_gate_align_yaw_tolerance_rad is not None
            and self.last_official_gate_index is not None
            and self.last_official_gate_index > 0
        ):
            return self.config.later_gate_align_yaw_tolerance_rad
        return self.config.align_yaw_tolerance_rad

    def _active_vertical_image_kp(self) -> float:
        if (
            self.config.later_gate_vertical_image_kp is not None
            and self.last_official_gate_index is not None
            and self.last_official_gate_index > 0
        ):
            return self.config.later_gate_vertical_image_kp
        return self.config.vertical_image_kp

    def _active_gate_roll_rad(self, gate_pose) -> float:
        if (
            self.config.later_gate_roll_gain == 0.0
            or self.last_official_gate_index is None
            or self.last_official_gate_index <= 0
            or gate_pose is None
        ):
            return 0.0
        image_center = getattr(gate_pose, "image_center_px", None)
        if image_center is None:
            return 0.0
        horizontal_norm = clamp((float(image_center[0]) - 320.0) / 320.0, -1.0, 1.0)
        return self.config.later_gate_roll_gain * horizontal_norm

    def _active_align_pitch_rad(self) -> float:
        if self.last_official_gate_index is not None and self.last_official_gate_index > 0:
            return self.config.later_gate_align_pitch_rad
        return 0.0

    def update(
        self,
        *,
        now_s: float,
        measured_yaw_rad: float,
        official_gate_index: int | None,
        gate_pose=None,
        gate_confidence: float = 0.0,
        guidance: GuidanceDetection | None = None,
    ) -> CourseCommand:
        cfg = self.config
        now_s = float(now_s)
        measured_yaw_rad = float(measured_yaw_rad)
        dt = 0.02 if self.last_update_s is None else clamp(now_s - self.last_update_s, 0.0, 0.2)
        self.last_update_s = now_s
        if self.phase_started_s is None:
            self.phase_started_s = now_s
            self.hold_yaw_rad = measured_yaw_rad

        if official_gate_index is not None:
            idx = int(official_gate_index)
            if self.last_official_gate_index is None:
                self.last_official_gate_index = idx
            elif idx > self.last_official_gate_index:
                self.last_official_gate_index = idx
                self.filtered_vertical_error_m = None
                self.hold_yaw_rad = measured_yaw_rad
                self._transition(CoursePhase.BRAKE, now_s, "official_gate_advanced")

        gate_usable = self._gate_is_usable(gate_pose, gate_confidence)
        if gate_usable:
            self.last_gate_seen_s = now_s
        if guidance is not None and guidance.confidence >= cfg.guidance_min_confidence:
            self.last_guidance_seen_s = now_s

        elapsed = now_s - float(self.phase_started_s)
        if self.phase == CoursePhase.BRAKE:
            pitch = cfg.brake_pitch_rad if elapsed < cfg.brake_duration_s else 0.0
            if elapsed >= cfg.brake_duration_s + cfg.brake_settle_s:
                self._transition(CoursePhase.FOLLOW_GUIDANCE, now_s, "brake_complete")
            else:
                return self._record(
                    CourseCommand(0.0, pitch, self.hold_yaw_rad, cfg.hover_thrust, "brake")
                )

        if self.phase == CoursePhase.COMMIT_GATE:
            desired_roll = self._active_gate_roll_rad(gate_pose) if gate_usable else 0.0
            if gate_usable:
                yaw_step = clamp(
                    self._active_gate_yaw_gain() * float(gate_pose.yaw_error_rad),
                    -cfg.gate_max_yaw_step_rad,
                    cfg.gate_max_yaw_step_rad,
                )
                self.hold_yaw_rad = wrap_angle(measured_yaw_rad - yaw_step)
            if elapsed >= cfg.commit_duration_s:
                self.hold_yaw_rad = measured_yaw_rad
                self._transition(CoursePhase.BRAKE, now_s, "commit_timeout")
                return self._record(
                    CourseCommand(
                        0.0,
                        cfg.brake_pitch_rad,
                        self.hold_yaw_rad,
                        cfg.hover_thrust,
                        "brake",
                    )
                )
            return self._record(
                CourseCommand(
                    desired_roll,
                    cfg.commit_pitch_rad,
                    self.hold_yaw_rad,
                    self._thrust_for_gate(gate_pose),
                    "commit_gate",
                )
            )

        if self.phase in {CoursePhase.ALIGN_GATE, CoursePhase.APPROACH_GATE}:
            if not gate_usable:
                if self.last_gate_seen_s is None or now_s - self.last_gate_seen_s > cfg.gate_lost_timeout_s:
                    self._transition(CoursePhase.FOLLOW_GUIDANCE, now_s, "gate_lost")
                else:
                    return self._record(
                        CourseCommand(0.0, 0.0, measured_yaw_rad, cfg.hover_thrust, "gate_coast")
                    )

        if self.phase == CoursePhase.FOLLOW_GUIDANCE:
            if gate_usable:
                self._transition(CoursePhase.ALIGN_GATE, now_s, "gate_acquired")
            else:
                if guidance is not None and guidance.confidence >= cfg.guidance_min_confidence:
                    yaw_step = clamp(
                        cfg.guidance_yaw_gain * guidance.heading_error_rad,
                        -cfg.guidance_max_yaw_step_rad,
                        cfg.guidance_max_yaw_step_rad,
                    )
                    # The rendered camera yaw axis is opposite the gyro/AHRS
                    # yaw axis: a camera-right target requires negative AHRS yaw.
                    desired_yaw = wrap_angle(measured_yaw_rad - yaw_step)
                    self.hold_yaw_rad = desired_yaw
                    return self._record(
                        CourseCommand(
                            0.0,
                            cfg.guidance_pitch_rad,
                            desired_yaw,
                            cfg.hover_thrust,
                            "follow_guidance",
                        )
                    )
                if cfg.guidance_lost_pitch_rad != 0.0:
                    return self._record(
                        CourseCommand(
                            0.0,
                            cfg.guidance_lost_pitch_rad,
                            self.hold_yaw_rad,
                            cfg.hover_thrust,
                            "guidance_coast",
                        )
                    )
                self.hold_yaw_rad = wrap_angle(
                    self.hold_yaw_rad + cfg.guidance_lost_search_rate_rad_s * dt
                )
                return self._record(
                    CourseCommand(
                        0.0,
                        0.0,
                        self.hold_yaw_rad,
                        cfg.hover_thrust,
                        "guidance_search",
                    )
                )

        if self.phase in {CoursePhase.ALIGN_GATE, CoursePhase.APPROACH_GATE} and gate_usable:
            _forward, _right, down = gate_pose.body_vector_ned_m
            yaw_error = float(gate_pose.yaw_error_rad)
            yaw_step = clamp(
                self._active_gate_yaw_gain() * yaw_error,
                -cfg.gate_max_yaw_step_rad,
                cfg.gate_max_yaw_step_rad,
            )
            desired_yaw = wrap_angle(measured_yaw_rad - yaw_step)
            desired_roll = self._active_gate_roll_rad(gate_pose)
            self.hold_yaw_rad = desired_yaw
            centered = bool(
                abs(yaw_error) <= self._active_align_yaw_tolerance_rad()
                and abs(float(down)) <= cfg.align_vertical_tolerance_m
            )

            if self.phase == CoursePhase.ALIGN_GATE:
                self.aligned_ticks = self.aligned_ticks + 1 if centered else 0
                if self.aligned_ticks >= cfg.align_consecutive_ticks:
                    self._transition(CoursePhase.APPROACH_GATE, now_s, "gate_centered")
                else:
                    return self._record(
                        CourseCommand(
                            desired_roll,
                            self._active_align_pitch_rad(),
                            desired_yaw,
                            self._thrust_for_gate(gate_pose),
                            "align_gate",
                        )
                    )

            if not centered:
                self._transition(CoursePhase.ALIGN_GATE, now_s, "alignment_lost")
                return self._record(
                    CourseCommand(
                        desired_roll,
                        self._active_align_pitch_rad(),
                        desired_yaw,
                        self._thrust_for_gate(gate_pose),
                        "align_gate",
                    )
                )

            if float(gate_pose.range_camera_m) <= cfg.commit_range_m:
                self.hold_yaw_rad = desired_yaw
                self._transition(CoursePhase.COMMIT_GATE, now_s, "commit_range_reached")
                return self._record(
                    CourseCommand(
                        desired_roll,
                        cfg.commit_pitch_rad,
                        self.hold_yaw_rad,
                        self._thrust_for_gate(gate_pose),
                        "commit_gate",
                    )
                )

            range_scale = clamp(float(gate_pose.range_camera_m) / 10.0, 0.45, 1.0)
            pitch = -min(
                cfg.approach_max_pitch_rad,
                abs(self._active_approach_pitch_rad()) * range_scale,
            )
            return self._record(
                CourseCommand(
                    desired_roll,
                    pitch,
                    desired_yaw,
                    self._thrust_for_gate(gate_pose),
                    "approach_gate",
                )
            )

        # Defensive hover; normal state paths return above.
        return self._record(
            CourseCommand(0.0, 0.0, measured_yaw_rad, cfg.hover_thrust, "safe_hover")
        )

    def to_summary(self) -> dict:
        return {
            "phase": self.phase.value,
            "last_official_gate_index": self.last_official_gate_index,
            "aligned_ticks": self.aligned_ticks,
            "filtered_vertical_error_m": self.filtered_vertical_error_m,
            "transitions": list(self.transitions),
            "command_counts": dict(self.command_counts),
            "config": dataclasses.asdict(self.config),
        }
