#!/usr/bin/env python3
"""Benchmark the fixed-rate policy publisher under deterministic GIL load."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time

try:
    from drone_sitl_competition_smoke import (
        AttitudeSetpoint,
        FixedRateAttitudePublisher,
        calculate_effective_command_rate,
    )
except ModuleNotFoundError:
    from scripts.drone_sitl_competition_smoke import (
        AttitudeSetpoint,
        FixedRateAttitudePublisher,
        calculate_effective_command_rate,
    )


class _EncodingAdapter:
    """Exercise Python-side setpoint encoding without opening a socket."""

    def __init__(self, encode_repeats: int) -> None:
        self.encode_repeats = encode_repeats
        self.payloads = 0

    def send_attitude_setpoint(self, target, *, mode: str = "body_rates") -> None:
        if mode != "body_rates":
            raise ValueError("benchmark requires body_rates")
        values = (
            float(target.body_roll_rate),
            float(target.body_pitch_rate),
            float(target.body_yaw_rate),
            float(target.thrust),
        )
        for _ in range(self.encode_repeats):
            struct.pack("<Iffff", self.payloads, *values)
        self.payloads += 1


def _run_once(
    *,
    switch_interval_s: float,
    command_hz: float,
    duration_s: float,
    main_work_iterations: int,
    encode_repeats: int,
) -> dict:
    sys.setswitchinterval(switch_interval_s)
    adapter = _EncodingAdapter(encode_repeats)
    publisher = FixedRateAttitudePublisher(
        adapter,
        command_hz=command_hz,
        initial_target=AttitudeSetpoint(
            body_roll_rate=0.1,
            body_pitch_rate=-0.1,
            body_yaw_rate=0.01,
            thrust=0.27,
        ),
        send_lock=threading.Lock(),
    )
    deadline = time.monotonic() + duration_s
    checksum = 0
    publisher.start()
    try:
        while time.monotonic() < deadline:
            for index in range(main_work_iterations):
                checksum = ((checksum * 1_664_525) + index + 1_013_904_223) & 0xFFFFFFFF
    finally:
        publisher.stop()
    rate_hz, interval_s = calculate_effective_command_rate(
        commands_sent=publisher.commands_sent,
        first_command_sent_s=publisher.first_command_sent_s,
        last_command_sent_s=publisher.last_command_sent_s,
        fallback_duration_s=duration_s,
    )
    return {
        "switch_interval_s": switch_interval_s,
        "requested_command_hz": command_hz,
        "commands_sent": publisher.commands_sent,
        "measurement_duration_s": interval_s,
        "effective_command_hz": rate_hz,
        "command_rate_violations": publisher.command_rate_violations,
        "adapter_payloads": adapter.payloads,
        "checksum": checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-interval-ms", nargs="+", type=float, default=[1.0, 0.5, 0.25])
    parser.add_argument("--command-hz", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--main-work-iterations", type=int, default=20_000)
    parser.add_argument("--encode-repeats", type=int, default=4)
    parser.add_argument("--json-path")
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.switch_interval_ms):
        parser.error("switch intervals must be positive")
    if args.command_hz <= 0.0 or args.duration <= 0.0 or args.repeats <= 0:
        parser.error("command rate, duration, and repeats must be positive")
    runs = []
    for switch_ms in args.switch_interval_ms:
        for repeat in range(args.repeats):
            run = _run_once(
                switch_interval_s=switch_ms * 1e-3,
                command_hz=args.command_hz,
                duration_s=args.duration,
                main_work_iterations=args.main_work_iterations,
                encode_repeats=args.encode_repeats,
            )
            run["repeat"] = repeat + 1
            runs.append(run)
    groups = {}
    for switch_ms in args.switch_interval_ms:
        selected = [run for run in runs if run["switch_interval_s"] == switch_ms * 1e-3]
        rates = [run["effective_command_hz"] for run in selected]
        groups[str(switch_ms)] = {
            "minimum_effective_command_hz": min(rates),
            "average_effective_command_hz": sum(rates) / len(rates),
            "maximum_effective_command_hz": max(rates),
            "total_command_rate_violations": sum(
                run["command_rate_violations"] for run in selected
            ),
        }
    result = {
        "platform": sys.platform,
        "command_hz": args.command_hz,
        "duration_s": args.duration,
        "repeats": args.repeats,
        "main_work_iterations": args.main_work_iterations,
        "encode_repeats": args.encode_repeats,
        "groups": groups,
        "runs": runs,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
