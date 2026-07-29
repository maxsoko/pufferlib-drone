#!/usr/bin/env python3
"""Send a short heartbeat/disarm burst without issuing any flight controls."""

from __future__ import annotations

import argparse
import time

from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument(
        "--zero-thrust-setpoint",
        action="store_true",
        help="Also clear a stale SET_ATTITUDE_TARGET throttle while disarmed.",
    )
    parser.add_argument(
        "--clear-throttle-inputs",
        action="store_true",
        help="Also publish minimum MANUAL_CONTROL and RC-channel-3 throttle.",
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")

    adapter = MavlinkSitlAdapter(args.endpoint)
    sent = 0
    deadline = time.monotonic() + args.duration_s
    try:
        while time.monotonic() < deadline:
            adapter.master.recv_match(blocking=True, timeout=0.05)
            adapter.send_heartbeat()
            adapter.send_disarm_command()
            if args.zero_thrust_setpoint:
                adapter.send_attitude_setpoint(
                    AttitudeSetpoint(thrust=0.0), mode="body_rates"
                )
            if args.clear_throttle_inputs:
                target_system, target_component = adapter._target_ids()
                adapter.master.mav.manual_control_send(
                    target_system, 0, 0, -1000, 0, 0
                )
                adapter.master.mav.rc_channels_override_send(
                    target_system,
                    target_component,
                    65535,
                    65535,
                    1000,
                    65535,
                    65535,
                    65535,
                    65535,
                    65535,
                )
            sent += 1
            time.sleep(0.05)
    finally:
        adapter.close()
    print(f"disarm_commands_sent={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
