#!/usr/bin/env python3
"""Measure the open-loop rate response per axis with live gyro integration.

Commands a constant SET_ATTITUDE_TARGET quaternion angle on one axis at a
time (open loop) while integrating the mount-corrected gyro, printing the
observed rate. This re-derives the plant gains with a working gyro feed;
the original pitch gain (-2.47) was inferred while the gyro read zero.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter  # noqa: E402
from drone_state_estimator import DeadReckoningEstimator  # noqa: E402


def main() -> int:
    adapter = MavlinkSitlAdapter("udpin:0.0.0.0:14550", dropout_after_s=5.0)
    estimator = DeadReckoningEstimator()
    last_imu_us = None

    def pump(duration_s: float, cmd: AttitudeSetpoint | None) -> dict:
        nonlocal last_imu_us
        n = 0
        gyro_min = [0.0, 0.0, 0.0]
        gyro_max = [0.0, 0.0, 0.0]
        deadline = time.monotonic() + duration_s
        next_hb = 0.0
        next_cmd = 0.0
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
                n += 1
                gyro = (st.xgyro or 0.0, st.ygyro or 0.0, st.zgyro or 0.0)
                for i in range(3):
                    gyro_min[i] = min(gyro_min[i], gyro[i])
                    gyro_max[i] = max(gyro_max[i], gyro[i])
                if estimator.state.calibrated:
                    estimator.update_imu(
                        imu_us * 1e-6,
                        (st.xacc, st.yacc, st.zacc),
                        gyro,
                    )
                else:
                    estimator.add_calibration_sample(
                        (st.xacc, st.yacc, st.zacc),
                        gyro,
                    )
        return {
            "n": n,
            "gyro_min": tuple(round(v, 3) for v in gyro_min),
            "gyro_max": tuple(round(v, 3) for v in gyro_max),
        }

    def phase(tag: str, cmd: AttitudeSetpoint, hold_s: float):
        r0 = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        t0 = time.monotonic()
        stats = pump(hold_s, cmd)
        dt = time.monotonic() - t0
        r1 = (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)
        rates = tuple((b - a) / dt for a, b in zip(r0, r1))
        acts = adapter.telemetry.state.actuator_outputs
        acts4 = tuple(round(a, 3) for a in (acts or ())[:4])
        print(
            f"{tag}: d_rpy=({r1[0]-r0[0]:+.3f},{r1[1]-r0[1]:+.3f},{r1[2]-r0[2]:+.3f}) rad "
            f"over {dt:.2f}s rates=({rates[0]:+.3f},{rates[1]:+.3f},{rates[2]:+.3f}) "
            f"gyro_minmax={stats['gyro_min']}..{stats['gyro_max']} imu_n={stats['n']} acts={acts4}"
        )

    try:
        t = time.monotonic() + 6
        while time.monotonic() < t and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
        print("reset sent")
        time.sleep(5.0)

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and estimator.calibration_samples < 60:
            pump(0.1, None)
        if estimator.calibration_samples < 30:
            print("calibration failed")
            return 1
        print("calibration:", estimator.finish_calibration())

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            adapter.send_heartbeat()
            adapter.poll_telemetry(timeout_s=0.05)
            rs = adapter.telemetry.state.race_status
            if rs is not None and 0 <= rs.race_start_boot_time_ms <= rs.sim_boot_time_ms:
                break
        # Persistent arming interleaved with commands: single bursts are
        # sometimes ignored. Verify spool-up via actuator outputs before the
        # measurement phases; idle actuators mean every phase reads zero.
        climb = AttitudeSetpoint(0.0, 0.0, 0.0, thrust=0.66)
        spooled = False
        for _ in range(25):
            adapter.send_heartbeat()
            adapter.send_arm_command()
            pump(0.2, climb)
            acts = adapter.telemetry.state.actuator_outputs
            if acts and max(acts) > 0.2:
                spooled = True
                break
        print("actuators:", adapter.telemetry.state.actuator_outputs, "spooled:", spooled)
        if not spooled:
            print("motors never spooled; aborting")
            return 1
        pump(1.2, climb)

        hover = 0.58
        phase("yaw   +0.30", AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.30, thrust=hover), 1.0)
        phase("yaw   +0.15", AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.15, thrust=hover), 1.0)
        phase("yaw   +0.10", AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.10, thrust=hover), 1.0)
        phase("pitch -0.30", AttitudeSetpoint(roll=0.0, pitch=-0.30, yaw=0.0, thrust=hover), 1.0)
        phase("pitch +0.30", AttitudeSetpoint(roll=0.0, pitch=0.30, yaw=0.0, thrust=hover), 1.0)
        phase("roll  +0.30", AttitudeSetpoint(roll=0.30, pitch=0.0, yaw=0.0, thrust=hover), 1.0)
        phase("level hold", AttitudeSetpoint(0.0, 0.0, 0.0, thrust=hover), 1.0)
        print("final rpy:", tuple(round(v, 3) for v in (estimator.state.roll, estimator.state.pitch, estimator.state.yaw)))
        print("collisions:", estimator.state.collision_events)
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
