#!/usr/bin/env python3
"""Check every MAVLink message for usable body-rate data during a maneuver.

The sign probe showed HIGHRES_IMU gyro pinned at zero while the drone was
visibly rotating, so this dumps raw message fields (not the adapter's parsed
state) to find out whether rates exist anywhere in the v3379 stream.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter  # noqa: E402


def main() -> int:
    adapter = MavlinkSitlAdapter("udpin:0.0.0.0:14550", dropout_after_s=5.0)
    master = adapter.master
    counts: Counter = Counter()
    gyro_nonzero: Counter = Counter()
    samples: dict[str, str] = {}

    def pump(duration_s: float, cmd: AttitudeSetpoint | None):
        next_hb = 0.0
        next_cmd = 0.0
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_hb:
                adapter.send_heartbeat()
                next_hb = now + 0.5
            if cmd is not None and now >= next_cmd:
                adapter.send_attitude_setpoint(cmd, mode="attitude")
                next_cmd = now + 1.0 / 30.0
            msg = master.recv_match(blocking=True, timeout=0.02)
            if msg is None:
                continue
            mtype = msg.get_type()
            counts[mtype] += 1
            d = msg.to_dict()
            rate_fields = {
                k: v
                for k, v in d.items()
                if any(t in k.lower() for t in ("gyro", "speed", "rate", "roll", "pitch", "yaw"))
                and isinstance(v, (int, float))
            }
            if rate_fields and any(abs(v) > 1e-6 for v in rate_fields.values()):
                gyro_nonzero[mtype] += 1
                samples[mtype] = str(rate_fields)
            elif mtype not in samples:
                samples[mtype] = str(rate_fields) if rate_fields else "(no rate-ish fields)"

    try:
        t = time.monotonic() + 6
        while time.monotonic() < t and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
        print("reset sent")
        time.sleep(5.0)

        # Wait for race start, then arm.
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

        print("climb...")
        pump(1.5, AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.0, thrust=0.66))
        print("yaw maneuver (expect sustained rotation)...")
        pump(3.0, AttitudeSetpoint(roll=0.0, pitch=0.0, yaw=0.3, thrust=0.58))

        print("\nmessage counts:", dict(counts))
        print("\nmessages with nonzero rate-ish fields:", dict(gyro_nonzero))
        for mtype, sample in sorted(samples.items()):
            marker = "*" if gyro_nonzero.get(mtype) else " "
            print(f"{marker} {mtype}: {sample}")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
