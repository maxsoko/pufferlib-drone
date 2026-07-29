#!/usr/bin/env python3
"""Ground-truth probe for the v3379 attitude plant sign conventions.

The closed attitude loop went unstable in flight, which smells like a sign
error. Every prior plant characterization used the sim gyro as the reference,
so a sign-flipped gyro axis would corrupt both the AHRS and the plant model
in a self-consistent way. This probe uses the camera as an independent
witness:

1. reset the sim, calibrate the IMU mount, arm, climb briefly;
2. command a small constant quaternion pitch (open loop) for ~1 s;
3. record the integrated (mount-corrected) gyro pitch AND the detector's
   range/center-pixel track of gate 1.

If the commanded nose-down maneuver shrinks the range while the gyro claims
nose-up (or vice versa), the gyro axis is inverted and the estimator must
negate it. The same run also re-measures the plant rate gain against the
image-derived truth. A second phase does the equivalent yaw check using the
horizontal image shift of the gate.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_camera_receiver import UdpCameraReceiver  # noqa: E402
from drone_gate_detector import SquareGateDetector  # noqa: E402
from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter  # noqa: E402
from drone_state_estimator import DeadReckoningEstimator  # noqa: E402


def dataclasses_asdict_safe(obj):
    import dataclasses as _dc

    try:
        return _dc.asdict(obj)
    except TypeError:
        return repr(obj)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--pitch-cmd-rad", type=float, default=-0.12)
    parser.add_argument("--pitch-hold-s", type=float, default=1.0)
    parser.add_argument("--yaw-cmd-rad", type=float, default=0.12)
    parser.add_argument("--yaw-hold-s", type=float, default=1.0)
    parser.add_argument("--climb-thrust", type=float, default=0.66)
    parser.add_argument("--climb-s", type=float, default=1.2)
    parser.add_argument("--hover-thrust", type=float, default=0.58)
    args = parser.parse_args()

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=5.0)
    receiver = UdpCameraReceiver(host="0.0.0.0", port=args.camera_port)
    detector = SquareGateDetector(
        min_area_px=300.0, max_aspect_error=0.8, min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )
    estimator = DeadReckoningEstimator()
    last_imu_us = None

    def poll(
        duration_s: float,
        *,
        cmd: AttitudeSetpoint | None,
        collect: list | None,
        tag: str = "",
    ):
        nonlocal last_imu_us
        deadline = time.monotonic() + duration_s
        next_hb = 0.0
        next_cmd = 0.0
        next_dump = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_hb:
                adapter.send_heartbeat()
                next_hb = now + 0.5
            if cmd is not None and now >= next_cmd:
                adapter.send_attitude_setpoint(cmd, mode="attitude")
                next_cmd = now + 1.0 / 30.0
            adapter.poll_telemetry(timeout_s=0.005)
            st = adapter.telemetry.state
            imu_us = st.imu_time_usec
            if imu_us is not None and imu_us != last_imu_us and st.xacc is not None:
                last_imu_us = imu_us
                if estimator.state.calibrated:
                    estimator.update_imu(
                        imu_us * 1e-6,
                        (st.xacc, st.yacc, st.zacc),
                        (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                    )
                else:
                    estimator.add_calibration_sample(
                        (st.xacc, st.yacc, st.zacc),
                        (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                    )
                if tag and now >= next_dump:
                    next_dump = now + 0.25
                    print(
                        f"[{tag}] accel=({st.xacc:+.2f},{st.yacc:+.2f},{st.zacc:+.2f}) "
                        f"gyro=({st.xgyro or 0.0:+.3f},{st.ygyro or 0.0:+.3f},{st.zgyro or 0.0:+.3f})"
                    )
            frames = receiver.poll_frames(max_packets=256)
            if frames and collect is not None:
                det = detector.detect_jpeg(frames[-1].jpeg)
                if det is not None:
                    from drone_visual_servo import estimate_gate_pose_from_corners

                    width_m = 1.5 if getattr(det, "source", "frame") == "aperture" else 3.0
                    pose = estimate_gate_pose_from_corners(
                        det.corners, gate_inner_width_m=width_m
                    )
                    collect.append(
                        {
                            "t": round(time.monotonic(), 3),
                            "range_m": round(pose.range_camera_m, 2),
                            "cx_px": round(pose.image_center_px[0], 1),
                            "cy_px": round(pose.image_center_px[1], 1),
                        }
                    )

    def summarize(tag: str, track: list, est_before: tuple, est_after: tuple):
        print(f"--- {tag} ---")
        print(f"gyro-integrated rpy before: {tuple(round(v, 3) for v in est_before)}")
        print(f"gyro-integrated rpy after:  {tuple(round(v, 3) for v in est_after)}")
        if track:
            first, last = track[0], track[-1]
            print(f"gate first: {first}")
            print(f"gate last:  {last}")
            print(f"range delta: {round(last['range_m'] - first['range_m'], 2)} m, "
                  f"cx delta: {round(last['cx_px'] - first['cx_px'], 1)} px")
        else:
            print("no gate detections in window")

    try:
        # Wait for the sim, reset it, wait for the auto-start.
        t = time.monotonic() + 6
        while time.monotonic() < t and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        if adapter.telemetry.state.race_status is None:
            print("simulator not reachable")
            return 1
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
        print("reset sent; waiting for respawn/auto-start")
        time.sleep(6.0)

        deadline = time.monotonic() + 8.0
        imu_seen = 0
        while time.monotonic() < deadline and estimator.calibration_samples < 60:
            adapter.send_heartbeat()
            adapter.poll_telemetry(timeout_s=0.05)
            st = adapter.telemetry.state
            imu_us = st.imu_time_usec
            if imu_us is not None and imu_us != last_imu_us and st.xacc is not None:
                last_imu_us = imu_us
                imu_seen += 1
                estimator.add_calibration_sample(
                    (st.xacc, st.yacc, st.zacc),
                    (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0),
                )
        print(f"imu samples seen: {imu_seen}, accepted: {estimator.calibration_samples}")
        st = adapter.telemetry.state
        print("last accel:", st.xacc, st.yacc, st.zacc, "gyro:", st.xgyro, st.ygyro, st.zgyro)
        if estimator.calibration_samples >= 30:
            print("calibration:", estimator.finish_calibration())
        else:
            print("calibration failed")
            return 1

        # Arming only sticks once the race is live (sim boot clock past the
        # scheduled auto-start), so wait for that before the arm burst.
        start_deadline = time.monotonic() + 15.0
        while time.monotonic() < start_deadline:
            adapter.send_heartbeat()
            adapter.poll_telemetry(timeout_s=0.05)
            rs = adapter.telemetry.state.race_status
            if (
                rs is not None
                and rs.race_start_boot_time_ms >= 0
                and rs.sim_boot_time_ms >= rs.race_start_boot_time_ms
            ):
                break
        rs = adapter.telemetry.state.race_status
        print("race_status pre-arm:", None if rs is None else dataclasses_asdict_safe(rs))
        for _ in range(5):
            adapter.send_heartbeat()
            adapter.send_arm_command()
            adapter.poll_telemetry(timeout_s=0.0)
            time.sleep(0.05)

        # Verify liftoff via the accelerometer: on the pad the specific force
        # stays pinned at the rest vector; any deviation means live motors.
        level = AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.0, thrust=args.climb_thrust)
        arm_deadline = time.monotonic() + 6.0
        lifted = False
        while time.monotonic() < arm_deadline and not lifted:
            adapter.send_heartbeat()
            adapter.send_arm_command()
            adapter.poll_telemetry(timeout_s=0.0)
            poll(0.3, cmd=level, collect=None)
            st = adapter.telemetry.state
            if st.xacc is not None:
                dev = math.sqrt(
                    (st.xacc + 2.999) ** 2 + (st.yacc - 0.0) ** 2 + (st.zacc + 9.3403) ** 2
                )
                if dev > 0.05:
                    lifted = True
        rs = adapter.telemetry.state.race_status
        print("race_status post-arm:", None if rs is None else dataclasses_asdict_safe(rs))
        print("lifted:", lifted)
        if not lifted:
            print("drone never spooled up; aborting probe")
            return 1
        poll(args.climb_s, cmd=level, collect=None)

        hover = AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.0, thrust=args.hover_thrust)
        baseline: list = []
        poll(0.6, cmd=hover, collect=baseline)

        # Phase 1: pitch.
        est_before = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        pitch_track: list = []
        pitch_cmd = AttitudeSetpoint(
            roll=0.0, pitch=args.pitch_cmd_rad, yaw=0.0, thrust=args.hover_thrust
        )
        poll(args.pitch_hold_s, cmd=pitch_cmd, collect=pitch_track, tag="pitch")
        est_after = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        summarize(
            f"PITCH cmd {args.pitch_cmd_rad:+.2f} rad for {args.pitch_hold_s}s",
            baseline + pitch_track,
            est_before,
            est_after,
        )

        # Re-level briefly.
        poll(1.0, cmd=hover, collect=None)

        # Phase 2: yaw.
        est_before = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        yaw_track: list = []
        yaw_cmd = AttitudeSetpoint(
            roll=0.0, pitch=0.0, yaw=args.yaw_cmd_rad, thrust=args.hover_thrust
        )
        poll(args.yaw_hold_s, cmd=yaw_cmd, collect=yaw_track, tag="yaw")
        est_after = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        summarize(
            f"YAW cmd {args.yaw_cmd_rad:+.2f} rad for {args.yaw_hold_s}s",
            yaw_track,
            est_before,
            est_after,
        )

        poll(0.5, cmd=hover, collect=None)
        return 0
    finally:
        receiver.close()
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
