#!/usr/bin/env python3
"""Find the real hover thrust of v3379 using the gate as an altitude witness.

Steps through candidate thrust values while tracking the active gate's
image center-y (gate fixed in world; cy decreasing = drone climbing above
it, cy increasing = drone sinking). Commands level attitude throughout so
the vertical channel is isolated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_camera_receiver import UdpCameraReceiver  # noqa: E402
from drone_gate_detector import SquareGateDetector  # noqa: E402
from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter  # noqa: E402
from drone_visual_servo import estimate_gate_pose_from_corners  # noqa: E402


# Liftoff sits between 0.10 (stays on pad) and 0.20 (flies); bisect it.
# Each value gets a fresh reset so climbs never accumulate across phases.
PHASES = [
    (0.12, 3.0),
    (0.14, 3.0),
    (0.16, 3.0),
    (0.18, 3.0),
    (0.20, 3.0),
]
RESET_EACH_PHASE = True


def main() -> int:
    adapter = MavlinkSitlAdapter("udpin:0.0.0.0:14550", dropout_after_s=5.0)
    receiver = UdpCameraReceiver(host="0.0.0.0", port=5600)
    detector = SquareGateDetector(
        min_area_px=300.0, max_aspect_error=0.8, min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )

    def run_phase(thrust: float, duration_s: float) -> None:
        cmd = AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.0, thrust=thrust)
        track: list[dict] = []
        deadline = time.monotonic() + duration_s
        next_hb = 0.0
        next_cmd = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_hb:
                adapter.send_heartbeat()
                next_hb = now + 0.5
            if now >= next_cmd:
                adapter.send_attitude_setpoint(cmd, mode="attitude")
                next_cmd = now + 1.0 / 30.0
            adapter.poll_telemetry(timeout_s=0.005)
            frames = receiver.poll_frames(max_packets=256)
            if frames:
                det = detector.detect_jpeg(frames[-1].jpeg)
                if det is not None:
                    width_m = 1.5 if getattr(det, "source", "frame") == "aperture" else 3.0
                    pose = estimate_gate_pose_from_corners(det.corners, gate_inner_width_m=width_m)
                    track.append(
                        {
                            "range": round(pose.range_camera_m, 1),
                            "cy": round(pose.image_center_px[1], 0),
                        }
                    )
        if track:
            cys = [p["cy"] for p in track]
            print(
                f"thrust {thrust:.2f}: n={len(track)} cy {cys[0]:.0f} -> {cys[-1]:.0f} "
                f"(min {min(cys):.0f}, max {max(cys):.0f}) range {track[0]['range']} -> {track[-1]['range']}"
            )
        else:
            print(f"thrust {thrust:.2f}: no detections")

    def reset_and_arm() -> None:
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
        time.sleep(5.0)
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

    try:
        t = time.monotonic() + 6
        while time.monotonic() < t and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        reset_each = bool(globals().get("RESET_EACH_PHASE", False))
        if not reset_each:
            reset_and_arm()
            print("reset sent")
        for thrust, duration in PHASES:
            if reset_each:
                reset_and_arm()
                # Drain stale frames buffered during the reset.
                receiver.poll_frames(max_packets=4096)
            run_phase(thrust, duration)
        return 0
    finally:
        receiver.close()
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
