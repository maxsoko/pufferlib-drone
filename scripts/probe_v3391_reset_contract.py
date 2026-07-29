#!/usr/bin/env python3
"""Prove the VQ1 v3391 MAVLink reset contract without flying.

The probe sends one normal disarm, waits 100 ms, and sends exactly one
simulator-specific command 31000.  It never arms and never publishes a flight
setpoint.  Acceptance requires a simulator boot rollback greater than 1000 ms
and a fresh official race epoch at gate index zero with no finish time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from drone_sitl_adapter import MavlinkSitlAdapter, RaceStatus


def accepted_reset_status(pre_reset_boot_ms: int, status: RaceStatus | None) -> bool:
    if status is None:
        return False
    return (
        int(status.sim_boot_time_ms) + 1000 < int(pre_reset_boot_ms)
        and int(status.race_start_boot_time_ms) >= 0
        and int(status.active_gate_index) == 0
        and int(status.race_finish_time_ns) < 0
    )


def _wait_for_preflight(adapter: MavlinkSitlAdapter, timeout_s: float) -> RaceStatus:
    deadline_s = time.monotonic() + timeout_s
    next_heartbeat_s = 0.0
    while time.monotonic() < deadline_s:
        now_s = time.monotonic()
        if now_s >= next_heartbeat_s:
            adapter.send_heartbeat()
            next_heartbeat_s = now_s + 0.5
        adapter.drain_telemetry(timeout_s=0.05)
        status = adapter.telemetry.state.race_status
        if (
            status is not None
            and adapter.telemetry.state.base_mode is not None
            and adapter.telemetry.state.system_status is not None
        ):
            return status
    raise RuntimeError("v3391 reset preflight did not receive heartbeat and race status")


def run_probe(endpoint: str, timeout_s: float) -> dict:
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")

    adapter = MavlinkSitlAdapter(endpoint, dropout_after_s=2.0)
    reset_command_s = None
    try:
        pre_status = _wait_for_preflight(adapter, min(timeout_s, 3.0))
        pre_state = dataclasses.asdict(adapter.telemetry.state)
        pre_reset_boot_ms = int(pre_status.sim_boot_time_ms)
        target_system, target_component = adapter._target_ids()

        adapter.send_disarm_command()
        time.sleep(0.1)
        adapter.send_sim_reset_command()
        reset_command_s = time.monotonic()

        deadline_s = reset_command_s + timeout_s
        next_heartbeat_s = reset_command_s
        accepted_s = None
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if now_s >= next_heartbeat_s:
                adapter.send_heartbeat()
                next_heartbeat_s = now_s + 0.5
            adapter.drain_telemetry(timeout_s=0.05)
            if accepted_reset_status(pre_reset_boot_ms, adapter.telemetry.state.race_status):
                accepted_s = time.monotonic()
                break

        adapter.telemetry.update_ages()
        post_status = adapter.telemetry.state.race_status
        report = {
            "contract": {
                "disarm_commands_sent": 1,
                "disarm_to_reset_wait_s": 0.1,
                "reset_command": 31000,
                "reset_commands_sent": 1,
                "arm_commands_sent": 0,
                "flight_setpoints_sent": 0,
            },
            "endpoint": endpoint,
            "learned_target_system": int(target_system),
            "learned_target_component": int(target_component),
            "accepted": accepted_s is not None,
            "reset_detection_s": (
                None if accepted_s is None else round(accepted_s - reset_command_s, 6)
            ),
            "pre_reset": pre_state,
            "post_reset": dataclasses.asdict(adapter.telemetry.state),
            "post_reset_race_status": (
                None if post_status is None else dataclasses.asdict(post_status)
            ),
            "telemetry": dataclasses.asdict(adapter.telemetry.metrics),
        }
        return report
    finally:
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove the telemetry-enabled VQ1 v3391 reset contract without flight"
    )
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--json-path", required=True)
    args = parser.parse_args()

    report = run_probe(args.endpoint, args.timeout_s)
    output_path = Path(args.json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit("v3391 reset did not satisfy the rollback/fresh-race contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
