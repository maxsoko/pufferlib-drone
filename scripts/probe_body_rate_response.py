#!/usr/bin/env python3
"""Measure v3385 body-rate command signs with reset-isolated short pulses."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from drone_sitl_adapter import AttitudeSetpoint, MavlinkSitlAdapter


def wait_for_heartbeat(adapter: MavlinkSitlAdapter, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while adapter.telemetry.metrics.heartbeats <= 0 and time.monotonic() < deadline:
        adapter.poll_telemetry(timeout_s=0.05)
    if adapter.telemetry.metrics.heartbeats <= 0:
        raise RuntimeError("no simulator heartbeat")


def safe_reset(adapter: MavlinkSitlAdapter, settle_s: float) -> None:
    wait_for_heartbeat(adapter)
    adapter.send_heartbeat()
    adapter.send_disarm_command()
    time.sleep(0.1)
    adapter.send_sim_reset_command()
    time.sleep(0.5)
    for _ in range(3):
        adapter.send_heartbeat()
        adapter.send_disarm_command()
        time.sleep(0.1)
    time.sleep(settle_s)


def pump(
    adapter: MavlinkSitlAdapter,
    duration_s: float,
    command: AttitudeSetpoint,
    *,
    arm: bool,
) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    last_imu_us = None
    deadline = time.monotonic() + duration_s
    next_heartbeat = 0.0
    next_command = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            adapter.send_heartbeat()
            if arm:
                adapter.send_arm_command()
            next_heartbeat = now + 0.2
        if now >= next_command:
            adapter.send_attitude_setpoint(command, mode="body_rates")
            next_command = now + 0.02
        adapter.poll_telemetry(timeout_s=0.005)
        state = adapter.telemetry.state
        if state.imu_time_usec is not None and state.imu_time_usec != last_imu_us:
            last_imu_us = state.imu_time_usec
            samples.append(
                (
                    float(state.xgyro or 0.0),
                    float(state.ygyro or 0.0),
                    float(state.zgyro or 0.0),
                )
            )
    return samples


def summarize(samples: list[tuple[float, float, float]]) -> dict:
    if not samples:
        raise RuntimeError("no gyro samples")
    tail = samples[len(samples) // 2 :]
    names = ("xgyro", "ygyro", "zgyro")
    return {
        name: {
            "median": round(statistics.median(row[index] for row in tail), 6),
            "min": round(min(row[index] for row in tail), 6),
            "max": round(max(row[index] for row in tail), 6),
        }
        for index, name in enumerate(names)
    }


def run_case(args, axis: str) -> dict:
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=3.0)
    try:
        safe_reset(adapter, args.reset_settle_s)
        wait_for_heartbeat(adapter)
        rates = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        level = AttitudeSetpoint(thrust=args.thrust)
        pump(adapter, args.spool_s, level, arm=True)
        baseline = pump(adapter, args.baseline_s, level, arm=True)
        rates[axis] = args.command_rate_rad_s
        pulse = AttitudeSetpoint(
            body_roll_rate=rates["roll"],
            body_pitch_rate=rates["pitch"],
            body_yaw_rate=rates["yaw"],
            thrust=args.thrust,
        )
        response = pump(adapter, args.pulse_s, pulse, arm=True)
        return {
            "axis": axis,
            "command_rate_rad_s": args.command_rate_rad_s,
            "baseline": summarize(baseline),
            "response": summarize(response),
            "actuator_outputs": list(adapter.telemetry.state.actuator_outputs or ())[:4],
            "collisions": adapter.telemetry.metrics.collisions,
        }
    finally:
        try:
            adapter.send_heartbeat()
            adapter.send_disarm_command()
        finally:
            adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--command-rate-rad-s", type=float, default=0.10)
    parser.add_argument("--thrust", type=float, default=0.27)
    parser.add_argument("--spool-s", type=float, default=0.8)
    parser.add_argument("--baseline-s", type=float, default=0.3)
    parser.add_argument("--pulse-s", type=float, default=0.4)
    parser.add_argument("--reset-settle-s", type=float, default=4.0)
    parser.add_argument("--json-path", default="logs/sitl/body_rate_response.json")
    args = parser.parse_args()

    payload = {
        "endpoint": args.endpoint,
        "cases": [run_case(args, axis) for axis in ("roll", "pitch", "yaw")],
    }
    path = Path(args.json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
