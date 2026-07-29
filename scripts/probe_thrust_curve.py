#!/usr/bin/env python3
"""Characterize the v3379 thrust plant and check IMU accel plausibility.

Flies a scripted sequence of constant (level, thrust) commands right after a
sim reset and logs, per phase:
  - raw HIGHRES_IMU accel/gyro statistics (mount-corrected),
  - the detector's gate range / image center-y track (independent witness of
    actual vertical motion: the gate is fixed in the world, so cy rising in
    the image means the drone is climbing).

This settles whether the estimator's "climbing at 17 m/s with thrust below
hover" is a real sim behavior or an accel-interpretation bug.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_camera_receiver import UdpCameraReceiver  # noqa: E402
from drone_gate_detector import SquareGateDetector  # noqa: E402
from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter  # noqa: E402
from drone_state_estimator import DeadReckoningEstimator  # noqa: E402
from drone_visual_servo import estimate_gate_pose_from_corners  # noqa: E402


# One fresh reset per phase; constant thrust held for the duration. The gyro
# attitude disambiguates the gate's image row into true relative height:
#   elev_body = -(cy - cy0)/fy + pitch, height = range * tan(elev_body)
PHASES = [
    (0.20, 5.0),
    (0.30, 5.0),
    (0.40, 5.0),
    (0.50, 5.0),
    (0.60, 5.0),
]


def main() -> int:
    adapter = MavlinkSitlAdapter("udpin:0.0.0.0:14550", dropout_after_s=5.0)
    receiver = UdpCameraReceiver(host="0.0.0.0", port=5600)
    detector = SquareGateDetector(
        min_area_px=300.0, max_aspect_error=0.8, min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )
    estimator = DeadReckoningEstimator()
    last_imu_us = None

    def run_phase(thrust: float, duration_s: float) -> None:
        nonlocal last_imu_us
        cmd = AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.0, thrust=thrust)
        start = time.monotonic()
        deadline = start + duration_s
        next_hb = 0.0
        next_cmd = 0.0
        next_print = 0.0
        latest = None
        print(f"--- thrust {thrust:.2f} ---")
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_hb:
                adapter.send_heartbeat()
                next_hb = now + 0.5
            if now >= next_cmd:
                adapter.send_attitude_setpoint(cmd, mode="attitude")
                next_cmd = now + 1.0 / 30.0
            adapter.poll_telemetry(timeout_s=0.005)
            st = adapter.telemetry.state
            imu_us = st.imu_time_usec
            if imu_us is not None and imu_us != last_imu_us and st.xacc is not None:
                last_imu_us = imu_us
                estimator.update_imu(
                    imu_us * 1e-6,
                    (st.xacc, st.yacc, st.zacc),
                    (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                )
            frames = receiver.poll_frames(max_packets=256)
            if frames:
                det = detector.detect_jpeg(frames[-1].jpeg)
                if det is not None:
                    width_m = 1.5 if getattr(det, "source", "frame") == "aperture" else 3.0
                    pose = estimate_gate_pose_from_corners(det.corners, gate_inner_width_m=width_m)
                    # Elevation of the gate above the drone, gyro-corrected
                    # for the drone's own pitch.
                    elev_cam = -(pose.image_center_px[1] - 180.0) / 320.0
                    elev_body = elev_cam + estimator.state.pitch
                    height_m = pose.range_camera_m * math.tan(elev_body)
                    latest = {
                        "t": round(now - start, 2),
                        "cy": round(pose.image_center_px[1], 0),
                        "range": round(pose.range_camera_m, 1),
                        "gate_above_m": round(height_m, 1),
                        "src": det.source,
                    }
            if now >= next_print:
                es = estimator.state
                print(
                    f"t={now - start:4.1f} rpy=({es.roll:+.2f},{es.pitch:+.2f},{es.yaw:+.2f}) "
                    f"gate={latest}"
                )
                latest = None
                next_print = now + 0.5

    def reset_calibrate_arm() -> bool:
        nonlocal last_imu_us, estimator
        estimator = DeadReckoningEstimator()
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
        time.sleep(4.0)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and estimator.calibration_samples < 40:
            adapter.send_heartbeat()
            adapter.poll_telemetry(timeout_s=0.05)
            st = adapter.telemetry.state
            imu_us = st.imu_time_usec
            if imu_us is not None and imu_us != last_imu_us and st.xacc is not None:
                last_imu_us = imu_us
                estimator.add_calibration_sample(
                    (st.xacc, st.yacc, st.zacc),
                    (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                )
        if estimator.calibration_samples < 30:
            print("calibration failed")
            return False
        estimator.finish_calibration()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            adapter.send_heartbeat()
            adapter.poll_telemetry(timeout_s=0.05)
            rs = adapter.telemetry.state.race_status
            if rs is not None and 0 <= rs.race_start_boot_time_ms <= rs.sim_boot_time_ms:
                break
        for _ in range(5):
            adapter.send_heartbeat()
            adapter.send_arm_command()
            time.sleep(0.05)
        receiver.poll_frames(max_packets=4096)
        return True

    try:
        t = time.monotonic() + 6
        while time.monotonic() < t and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        if adapter.telemetry.state.race_status is None:
            print("simulator not reachable")
            return 1
        for thrust, duration in PHASES:
            if not reset_calibrate_arm():
                return 1
            run_phase(thrust, duration)
        return 0
    finally:
        receiver.close()
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
