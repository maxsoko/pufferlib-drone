#!/usr/bin/env python3
"""State estimator for AI-GP Simulator v3379: gyro attitude + visual odometry.

Measured sensor truth (July 2026, camera-witnessed probes):

- HIGHRES_IMU **gyro** reports genuine body rates while the airframe rotates.
- HIGHRES_IMU **accel** is fiction: during clean flight it reads exactly the
  calibrated rest vector (constant magnitude 9.81, constant direction in the
  body frame) regardless of attitude, translation, or thrust, and only
  deviates during collisions (multi-hundred-g spikes). It carries zero
  translational and zero attitude information.

Consequences baked into this design:

1. Attitude comes from gyro integration alone, initialized level on the pad.
   There is no useful gravity reference in flight, so there is no accel blend
   (`ahrs_accel_gain` defaults to 0) and no automatic ZUPT attitude snap.
2. Velocity/position dead reckoning from the accelerometer is impossible.
   Position is corrected by gate-landmark fixes; velocity is differentiated
   from consecutive fixes and coasted (with leak) between them.
3. Accel spikes are still useful as a collision detector (`collision_events`).

Landmark model: gates are world-fixed. A detection gives the gate vector in
body-NED axes; rotating by the gyro attitude yields a world-frame relative
position. The first sighting maps the gate (SLAM-lite with data association
via active_gate_index); later sightings are absolute position fixes.

The accelerometer rest direction is tilted ~17.8 deg in pitch and must be
calibrated on the pad. The gyro, by contrast, reports directly in the
command/body frame (verified live: a pure yaw maneuver reads as pure zgyro),
so gyro rates are never mount-corrected. Note the rendered camera is level
(probe_camera_tilt.py) despite TS-002's documented 20 deg uptilt.
"""

from __future__ import annotations

import dataclasses
import math


GRAVITY_M_S2 = 9.80665


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


Vec3 = tuple[float, float, float]


def _rotation_matrix_zyx(roll: float, pitch: float, yaw: float) -> tuple[Vec3, Vec3, Vec3]:
    """Body-to-world rotation for aerospace ZYX (yaw-pitch-roll) Euler angles."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rotate_body_to_world(vec: Vec3, roll: float, pitch: float, yaw: float) -> Vec3:
    r = _rotation_matrix_zyx(roll, pitch, yaw)
    return (
        r[0][0] * vec[0] + r[0][1] * vec[1] + r[0][2] * vec[2],
        r[1][0] * vec[0] + r[1][1] * vec[1] + r[1][2] * vec[2],
        r[2][0] * vec[0] + r[2][1] * vec[1] + r[2][2] * vec[2],
    )


def rotate_world_to_body(vec: Vec3, roll: float, pitch: float, yaw: float) -> Vec3:
    r = _rotation_matrix_zyx(roll, pitch, yaw)
    return (
        r[0][0] * vec[0] + r[1][0] * vec[1] + r[2][0] * vec[2],
        r[0][1] * vec[0] + r[1][1] * vec[1] + r[2][1] * vec[2],
        r[0][2] * vec[0] + r[1][2] * vec[1] + r[2][2] * vec[2],
    )


@dataclasses.dataclass
class EstimatorConfig:
    gravity_m_s2: float = GRAVITY_M_S2
    # Velocity is derived from consecutive landmark fixes and *coasted*
    # between sightings; the leak keeps a stale vision velocity from steering
    # guidance forever after the gate leaves the FOV.
    velocity_leak_per_s: float = 0.3
    # Complementary correction gain for landmark position fixes.
    position_correction_gain: float = 0.3
    # Blend gain for the fix-differenced velocity measurement.
    fix_velocity_gain: float = 0.35
    # Velocity is differenced between raw fixes at least this far apart in
    # time (averages detector range jitter over a usable baseline) ...
    min_fix_baseline_s: float = 0.15
    # ... and two fixes further apart than this are not differentiated.
    max_fix_gap_s: float = 0.5
    # A racing quad cannot exceed ~25 m/s; faster fix-differenced velocity is
    # detector noise, not motion.
    max_speed_m_s: float = 25.0
    # Landmark map smoothing gain (first observation is taken as-is).
    landmark_smoothing_gain: float = 0.3
    # Innovation gate: the detector flips between aperture and frame
    # interpretations of the same gate (2x range jumps), so a fix that
    # disagrees with the current state by more than this is an outlier and
    # must not correct the state.
    max_innovation_m: float = 8.0
    min_calibration_samples: int = 20
    max_dt_s: float = 0.2
    # Reject non-gravity samples during calibration (spawn drops and reset
    # transients produce spikes of hundreds of m/s^2).
    calibration_magnitude_tolerance: float = 0.15
    # The sim accel departs from the rest vector only during collisions, so a
    # large magnitude excursion IS a collision event.
    collision_accel_m_s2: float = 3.0 * GRAVITY_M_S2
    collision_debounce_s: float = 0.5
    # Optional gravity blend for the gyro attitude. Default 0: the v3379
    # accelerometer always reads the calibrated rest vector in flight (it
    # carries no attitude information), so blending would drag pitch/roll
    # toward a fictitious "level" and destabilize the closed attitude loop.
    ahrs_accel_gain: float = 0.0
    ahrs_accel_trust_band: float = 0.05
    # Only blend when the body is quasi-static (low rotation rate).
    ahrs_accel_max_gyro_rad_s: float = 0.2


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclasses.dataclass
class AttitudeLoopConfig:
    """Closed-loop attitude control over the v3379 SET_ATTITUDE_TARGET plant.

    Measured plant behavior (July 2026): a constant commanded quaternion angle
    produces a constant body rate of ~2.4x the commanded angle on each axis —
    the sim FC has no attitude feedback and treats the command as a rate
    demand. We close the loop with the gyro AHRS: command an "angle" equal to
    desired_rate / plant_rate_gain where desired_rate = kp * attitude_error.
    """

    # Measured per-axis with a live calibrated gyro (July 2026):
    # roll +0.30 -> +0.761 rad/s, pitch +0.30 -> -0.700 rad/s (sign-inverted
    # plant), yaw +0.30 -> +0.616 rad/s; gains linear across 0.10-0.30 rad.
    plant_rate_gain_roll: float = 2.5
    plant_rate_gain_pitch: float = -2.35
    plant_rate_gain_yaw: float = 2.05
    # v3385 official body-rate fields are also a scaled plant rather than
    # literal gyro rates. Reset-isolated +0.10 rad/s pulses measured
    # (-0.2416, -0.2418, -0.2189) rad/s on (roll, pitch, yaw).
    body_rate_gain_roll: float = -2.416
    body_rate_gain_pitch: float = -2.418
    body_rate_gain_yaw: float = -2.189
    body_rate_min_command_rad_s: float = 0.05
    attitude_error_deadband_rad: float = 0.004
    # kp = 3.0 oscillated in flight (est pitch overshot a -0.06 command to
    # +0.37): the physical airframe has rate lag the pure-gain plant model
    # ignores. kp = 1.5 gives ~0.7 s convergence with margin for that lag.
    kp_roll: float = 1.5
    kp_pitch: float = 1.5
    kp_yaw: float = 1.5
    max_cmd_rad: float = 0.4
    max_body_rate_rad_s: float = 0.4


def attitude_loop_command(
    desired_rpy: Vec3, measured_rpy: Vec3, config: AttitudeLoopConfig | None = None
) -> Vec3:
    """Angle command to send so the sim FC drives measured attitude to desired."""
    cfg = config or AttitudeLoopConfig()
    err_roll = wrap_angle(desired_rpy[0] - measured_rpy[0])
    err_pitch = wrap_angle(desired_rpy[1] - measured_rpy[1])
    err_yaw = wrap_angle(desired_rpy[2] - measured_rpy[2])
    limit = cfg.max_cmd_rad
    return (
        clamp(cfg.kp_roll * err_roll / cfg.plant_rate_gain_roll, -limit, limit),
        clamp(cfg.kp_pitch * err_pitch / cfg.plant_rate_gain_pitch, -limit, limit),
        clamp(cfg.kp_yaw * err_yaw / cfg.plant_rate_gain_yaw, -limit, limit),
    )


def attitude_body_rate_command(
    desired_rpy: Vec3,
    measured_rpy: Vec3,
    config: AttitudeLoopConfig | None = None,
) -> Vec3:
    """Scaled command for the measured official body-rate plant."""
    cfg = config or AttitudeLoopConfig()
    limit = cfg.max_body_rate_rad_s

    def axis_command(error: float, kp: float, plant_gain: float) -> float:
        if abs(error) <= cfg.attitude_error_deadband_rad:
            return 0.0
        raw = kp * error / plant_gain
        if 0.0 < abs(raw) < cfg.body_rate_min_command_rad_s:
            raw = math.copysign(cfg.body_rate_min_command_rad_s, raw)
        return clamp(raw, -limit, limit)

    return tuple(
        axis_command(wrap_angle(desired - measured), kp, gain)
        for desired, measured, kp, gain in zip(
            desired_rpy,
            measured_rpy,
            (cfg.kp_roll, cfg.kp_pitch, cfg.kp_yaw),
            (
                cfg.body_rate_gain_roll,
                cfg.body_rate_gain_pitch,
                cfg.body_rate_gain_yaw,
            ),
        )
    )


@dataclasses.dataclass
class EstimatorState:
    position_ned_m: Vec3 = (0.0, 0.0, 0.0)
    velocity_ned_m_s: Vec3 = (0.0, 0.0, 0.0)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    imu_samples: int = 0
    position_corrections: int = 0
    collision_events: int = 0
    last_collision_time_s: float | None = None
    last_imu_time_s: float | None = None
    calibrated: bool = False


class DeadReckoningEstimator:
    """Gyro attitude + gate-landmark visual odometry.

    All world-frame quantities are NED with the origin at the calibration
    (takeoff) position and yaw 0 along the drone's initial heading — the same
    convention as the attitude-command yaw integrator.
    """

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.config = config or EstimatorConfig()
        self.state = EstimatorState()
        self._rest_samples: list[Vec3] = []
        # Mount correction: rotation that maps measured body-sensor axes into
        # the attitude-command body frame (identity until calibrated).
        self._mount_roll = 0.0
        self._mount_pitch = 0.0
        self._ahrs_active = False
        self._last_fix: tuple[Vec3, float] | None = None
        self.landmarks: dict[int, Vec3] = {}
        self.landmark_observations: dict[int, int] = {}
        self.rejected_observations = 0

    # ------------------------------------------------------------------
    # Calibration (drone landed or hovering with commanded level attitude)
    # ------------------------------------------------------------------
    def add_calibration_sample(self, accel_body: Vec3, gyro_body: Vec3 | None = None) -> bool:
        """Add a rest sample; returns False if rejected as a transient.

        Requires ~1g magnitude and (when gyro is provided) a still airframe,
        so spawn drops, spool-up shake, and in-flight samples never pollute
        the gravity/mount estimate.
        """
        sample = tuple(float(v) for v in accel_body)
        magnitude = math.sqrt(sum(v * v for v in sample))
        tol = self.config.calibration_magnitude_tolerance
        if abs(magnitude - self.config.gravity_m_s2) > tol * self.config.gravity_m_s2:
            self._rest_samples.clear()
            return False
        if gyro_body is not None:
            gyro_mag = math.sqrt(sum(float(v) * float(v) for v in gyro_body))
            if gyro_mag > 0.05:
                self._rest_samples.clear()
                return False
        self._rest_samples.append(sample)
        return True

    @property
    def calibration_samples(self) -> int:
        return len(self._rest_samples)

    def finish_calibration(self) -> dict:
        n = len(self._rest_samples)
        if n < self.config.min_calibration_samples:
            raise ValueError(
                f"need >= {self.config.min_calibration_samples} calibration samples, got {n}"
            )
        fx = sum(s[0] for s in self._rest_samples) / n
        fy = sum(s[1] for s in self._rest_samples) / n
        fz = sum(s[2] for s in self._rest_samples) / n
        magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
        if magnitude < 1e-6:
            raise ValueError("calibration accel magnitude is degenerate")
        # At rest with level commanded attitude the specific force should be
        # (0, 0, -g) in the command frame. The measured direction reveals the
        # fixed mount tilt (yaw component unobservable from gravity; assume 0).
        self._mount_pitch = math.atan2(-fx, -fz)
        horizontal = math.sqrt(fx * fx + fz * fz)
        self._mount_roll = math.atan2(fy, horizontal)
        self.state.calibrated = True
        self._rest_samples.clear()
        return {
            "samples": n,
            "mean_accel": (round(fx, 6), round(fy, 6), round(fz, 6)),
            "magnitude": round(magnitude, 6),
            "mount_pitch_rad": round(self._mount_pitch, 6),
            "mount_roll_rad": round(self._mount_roll, 6),
        }

    def _mount_correct(self, accel_body: Vec3) -> Vec3:
        # f_imu = R(mount) @ f_cmd, so recovering the command frame applies the
        # inverse rotation of the calibrated mount tilt.
        return rotate_world_to_body(accel_body, self._mount_roll, self._mount_pitch, 0.0)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def set_commanded_attitude(self, roll: float, pitch: float, yaw: float) -> None:
        """Commanded-attitude fallback; ignored once gyro AHRS is running.

        The sim FC tracks commands in free flight, but after a collision or on
        the ground the true attitude diverges from the command, so the gyro
        (when supplied to update_imu) is the authoritative source.
        """
        if self._ahrs_active:
            return
        self.state.roll = float(roll)
        self.state.pitch = float(pitch)
        self.state.yaw = float(yaw)

    def _update_ahrs(self, gyro_cmd_frame: Vec3, f_cmd_frame: Vec3, dt: float) -> None:
        st = self.state
        p, q, r = gyro_cmd_frame
        sr, cr = math.sin(st.roll), math.cos(st.roll)
        tp = math.tan(clamp(st.pitch, -1.45, 1.45))
        cp = max(math.cos(st.pitch), 1e-3)
        st.roll += (p + sr * tp * q + cr * tp * r) * dt
        st.pitch += (cr * q - sr * r) * dt
        st.yaw += ((sr / cp) * q + (cr / cp) * r) * dt

        fx, fy, fz = f_cmd_frame
        magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
        band = self.config.ahrs_accel_trust_band
        gyro_mag = math.sqrt(p * p + q * q + r * r)
        if (
            abs(magnitude - self.config.gravity_m_s2) <= band * self.config.gravity_m_s2
            and gyro_mag <= self.config.ahrs_accel_max_gyro_rad_s
        ):
            roll_meas = math.atan2(-fy, -fz)
            pitch_meas = math.atan2(fx, math.sqrt(fy * fy + fz * fz))
            gain = self.config.ahrs_accel_gain
            st.roll += gain * wrap_angle(roll_meas - st.roll)
            st.pitch += gain * wrap_angle(pitch_meas - st.pitch)

    def update_imu(self, time_s: float, accel_body: Vec3, gyro_body: Vec3 | None = None) -> None:
        """Fold one IMU sample: gyro -> attitude, accel -> collision detector.

        The v3379 accelerometer carries no translational information (it
        reads the calibrated rest vector throughout clean flight), so it is
        never integrated. Position advances by coasting the vision-derived
        velocity. time_s must be monotonic (e.g. time_usec * 1e-6).
        """
        st = self.state
        last = st.last_imu_time_s
        st.last_imu_time_s = float(time_s)
        st.imu_samples += 1
        sample = tuple(float(v) for v in accel_body)
        magnitude = math.sqrt(sum(v * v for v in sample))
        if magnitude > self.config.collision_accel_m_s2:
            if (
                st.last_collision_time_s is None
                or float(time_s) - st.last_collision_time_s > self.config.collision_debounce_s
            ):
                st.collision_events += 1
            st.last_collision_time_s = float(time_s)
        if last is None:
            return
        dt = float(time_s) - last
        if dt <= 0.0 or dt > self.config.max_dt_s:
            return

        if gyro_body is not None:
            self._ahrs_active = True
            # Measured (July 2026): a pure yaw maneuver produces a pure
            # z-axis gyro reading, so the gyro is already in the command/body
            # frame — only the accelerometer rest direction is tilted. Do NOT
            # mount-correct the gyro or a yaw injects phantom roll.
            gyro_cmd_frame = tuple(float(v) for v in gyro_body)
            self._update_ahrs(gyro_cmd_frame, self._mount_correct(sample), dt)

        leak = clamp(1.0 - self.config.velocity_leak_per_s * dt, 0.0, 1.0)
        vx = st.velocity_ned_m_s[0] * leak
        vy = st.velocity_ned_m_s[1] * leak
        vz = st.velocity_ned_m_s[2] * leak
        st.velocity_ned_m_s = (vx, vy, vz)
        px, py, pz = st.position_ned_m
        st.position_ned_m = (px + vx * dt, py + vy * dt, pz + vz * dt)

    def zero_velocity_update(self, gain: float = 1.0) -> None:
        """Apply when the drone is known stationary (pre-arm, landed)."""
        g = clamp(gain, 0.0, 1.0)
        vx, vy, vz = self.state.velocity_ned_m_s
        self.state.velocity_ned_m_s = (vx * (1.0 - g), vy * (1.0 - g), vz * (1.0 - g))

    # ------------------------------------------------------------------
    # Landmark map + corrections
    # ------------------------------------------------------------------
    def gate_world_vector(self, body_vector_ned: Vec3) -> Vec3:
        """Rotate a detector body-NED gate vector into the world frame."""
        return rotate_body_to_world(
            tuple(float(v) for v in body_vector_ned),
            self.state.roll,
            self.state.pitch,
            self.state.yaw,
        )

    def observe_gate(
        self, gate_id: int, body_vector_ned: Vec3, time_s: float | None = None
    ) -> Vec3:
        """Fold one gate detection into the map and correct the state.

        Returns the gate's current mapped world position. The first sighting
        maps the gate from the current state estimate; later sightings are
        absolute position fixes, and consecutive fixes are differentiated
        into a velocity measurement (the only velocity source available:
        v3379's accelerometer is uninformative).
        """
        rel_world = self.gate_world_vector(body_vector_ned)
        px, py, pz = self.state.position_ned_m
        observed_gate = (px + rel_world[0], py + rel_world[1], pz + rel_world[2])

        if gate_id not in self.landmarks:
            self.landmarks[gate_id] = observed_gate
            self.landmark_observations[gate_id] = 1
            if time_s is not None:
                self._last_fix = (self.state.position_ned_m, float(time_s))
            return observed_gate

        mapped = self.landmarks[gate_id]
        self.landmark_observations[gate_id] += 1
        # Position fix: where the mapped gate says we are.
        fix = (
            mapped[0] - rel_world[0],
            mapped[1] - rel_world[1],
            mapped[2] - rel_world[2],
        )
        innovation = math.sqrt(
            (fix[0] - px) ** 2 + (fix[1] - py) ** 2 + (fix[2] - pz) ** 2
        )
        if innovation > self.config.max_innovation_m:
            self.rejected_observations += 1
            return mapped
        kp = self.config.position_correction_gain
        corrected = (
            px + kp * (fix[0] - px),
            py + kp * (fix[1] - py),
            pz + kp * (fix[2] - pz),
        )
        self.state.position_ned_m = corrected
        if time_s is not None:
            if self._last_fix is not None:
                (lx, ly, lz), lt = self._last_fix
                gap = float(time_s) - lt
                if gap > self.config.max_fix_gap_s:
                    # Stale baseline; restart differencing from this fix.
                    self._last_fix = (fix, float(time_s))
                elif gap >= self.config.min_fix_baseline_s:
                    v_meas = (
                        (fix[0] - lx) / gap,
                        (fix[1] - ly) / gap,
                        (fix[2] - lz) / gap,
                    )
                    v_mag = math.sqrt(sum(v * v for v in v_meas))
                    if v_mag <= self.config.max_speed_m_s:
                        kv = self.config.fix_velocity_gain
                        vx, vy, vz = self.state.velocity_ned_m_s
                        self.state.velocity_ned_m_s = (
                            vx + kv * (v_meas[0] - vx),
                            vy + kv * (v_meas[1] - vy),
                            vz + kv * (v_meas[2] - vz),
                        )
                    self._last_fix = (fix, float(time_s))
            else:
                self._last_fix = (fix, float(time_s))
        self.state.position_corrections += 1
        # Refine the map with the residual so early mapping error washes out.
        km = self.config.landmark_smoothing_gain / self.landmark_observations[gate_id]
        self.landmarks[gate_id] = (
            mapped[0] + km * (observed_gate[0] - mapped[0]),
            mapped[1] + km * (observed_gate[1] - mapped[1]),
            mapped[2] + km * (observed_gate[2] - mapped[2]),
        )
        return self.landmarks[gate_id]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_summary(self) -> dict:
        st = self.state
        return {
            "calibrated": st.calibrated,
            "mount_pitch_rad": round(self._mount_pitch, 6),
            "mount_roll_rad": round(self._mount_roll, 6),
            "position_ned_m": tuple(round(v, 6) for v in st.position_ned_m),
            "velocity_ned_m_s": tuple(round(v, 6) for v in st.velocity_ned_m_s),
            "yaw_rad": round(st.yaw, 6),
            "imu_samples": st.imu_samples,
            "position_corrections": st.position_corrections,
            "rejected_observations": self.rejected_observations,
            "collision_events": st.collision_events,
            "roll_rad": round(st.roll, 6),
            "pitch_rad": round(st.pitch, 6),
            "landmarks": {
                str(gate_id): tuple(round(v, 6) for v in pos)
                for gate_id, pos in sorted(self.landmarks.items())
            },
            "landmark_observations": {
                str(gate_id): count
                for gate_id, count in sorted(self.landmark_observations.items())
            },
        }
