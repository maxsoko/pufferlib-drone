#!/usr/bin/env python3
"""Wait for the official simulator race-start flag before running control."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Callable

from drone_sitl_adapter import MavlinkSitlAdapter


def _to_dict(value) -> dict | None:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return dict(value)


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def wait_for_race_start(
    args,
    *,
    adapter_factory: Callable[..., MavlinkSitlAdapter] = MavlinkSitlAdapter,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.poll_timeout_s < 0.0:
        raise ValueError("poll-timeout-s must be non-negative")

    adapter = adapter_factory(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    started_s = monotonic()
    deadline_s = started_s + args.duration
    try:
        while monotonic() < deadline_s:
            timeout_s = min(args.poll_timeout_s, max(0.0, deadline_s - monotonic()))
            adapter.poll_telemetry(timeout_s=timeout_s)
            race_status = adapter.telemetry.state.race_status
            if (
                race_status is not None
                and int(race_status.race_start_boot_time_ms) >= 0
                and adapter.telemetry.state.base_mode is not None
                and adapter.telemetry.state.system_status is not None
            ):
                break
    finally:
        adapter.close()

    elapsed_s = round(monotonic() - started_s, 6)
    state = adapter.telemetry.state
    race_status = state.race_status
    race_started = None
    if race_status is not None:
        race_started = int(race_status.race_start_boot_time_ms) >= 0

    return {
        "endpoint": args.endpoint,
        "duration_s": float(args.duration),
        "elapsed_s": elapsed_s,
        "race_started": race_started,
        "race_status": _to_dict(race_status),
        "base_mode": state.base_mode,
        "system_status": state.system_status,
        "messages_seen": int(adapter.telemetry.metrics.messages_seen),
        "heartbeats_seen": int(adapter.telemetry.metrics.heartbeats),
        "race_statuses_seen": int(adapter.telemetry.metrics.race_statuses),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for official simulator RACE_STATUS to become active")
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--poll-timeout-s", type=float, default=0.05)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--require-started", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = wait_for_race_start(args)
    _write_json(args.json_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_started and report["race_started"] is not True:
        raise SystemExit("race did not start before timeout")


if __name__ == "__main__":
    main()
