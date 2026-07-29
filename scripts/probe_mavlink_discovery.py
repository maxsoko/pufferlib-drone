#!/usr/bin/env python3
"""Probe non-flight MAVLink discovery services on the AI-GP simulator.

The probe deliberately sends no reset, arm/disarm, or control setpoint.  Its
default message requests are limited to protocol metadata and simulator-to-
client messages named by VADR-TS-002 issue 00.02.  It never requests position,
odometry, HIL, simulator-truth, GPS, or track geometry.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from drone_sitl_adapter import MavlinkSitlAdapter


DEFAULT_ONE_SHOT_MESSAGES = (
    "AUTOPILOT_VERSION",
    "PROTOCOL_VERSION",
    "ATTITUDE",
)
DEFAULT_INTERVAL_MESSAGES = (
    "ATTITUDE",
    "HIGHRES_IMU",
    "TIMESYNC",
)
STANDARD_MESSAGE_IDS = {
    # Keep discovery usable with older generated pymavlink dialects. These are
    # wire-stable message IDs from common.xml, not simulator-private IDs.
    "ATTITUDE": 30,
    "HIGHRES_IMU": 105,
    "TIMESYNC": 111,
    "AUTOPILOT_VERSION": 148,
    "MESSAGE_INTERVAL": 244,
    "PROTOCOL_VERSION": 300,
}


def mavlink_message_id(mavlink: Any, name: str) -> int:
    normalized = name.upper()
    constant = f"MAVLINK_MSG_ID_{normalized}"
    if hasattr(mavlink, constant):
        return int(getattr(mavlink, constant))
    if normalized in STANDARD_MESSAGE_IDS:
        return STANDARD_MESSAGE_IDS[normalized]
    raise ValueError(f"pymavlink dialect has no {constant}")


def command_result_name(mavlink: Any, result: int) -> str:
    entries = getattr(mavlink, "enums", {}).get("MAV_RESULT", {})
    entry = entries.get(int(result))
    return str(getattr(entry, "name", f"MAV_RESULT_{int(result)}"))


def send_request_message(
    adapter: MavlinkSitlAdapter,
    message_id: int,
    *,
    request_param1: float = 0.0,
    response_target: int = 1,
) -> None:
    """Send one MAV_CMD_REQUEST_MESSAGE with a response addressed to us."""
    target_system, target_component = adapter._target_ids()
    adapter.master.mav.command_long_send(
        target_system,
        target_component,
        adapter.mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0,
        float(message_id),
        float(request_param1),
        0.0,
        0.0,
        0.0,
        0.0,
        float(response_target),
    )


def send_message_interval_query(
    adapter: MavlinkSitlAdapter,
    message_id: int,
    *,
    response_target: int = 1,
) -> None:
    mavlink = adapter.mavutil.mavlink
    send_request_message(
        adapter,
        mavlink_message_id(mavlink, "MESSAGE_INTERVAL"),
        request_param1=float(message_id),
        response_target=response_target,
    )


def message_record(message: Any, mavlink: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": message.get_type(),
        "source_system": int(message.get_srcSystem()),
        "source_component": int(message.get_srcComponent()),
        "sequence": int(message.get_seq()),
    }
    if message.get_type() == "COMMAND_ACK":
        result = int(message.result)
        record.update(
            {
                "command": int(message.command),
                "result": result,
                "result_name": command_result_name(mavlink, result),
            }
        )
    elif message.get_type() == "MESSAGE_INTERVAL":
        record.update(
            {
                "message_id": int(message.message_id),
                "interval_us": int(message.interval_us),
            }
        )
    elif message.get_type() == "PING":
        record.update(
            {
                "time_usec": int(message.time_usec),
                "ping_sequence": int(message.seq),
                "target_system": int(message.target_system),
                "target_component": int(message.target_component),
            }
        )
    elif message.get_type() == "TIMESYNC":
        record.update({"tc1": int(message.tc1), "ts1": int(message.ts1)})
    elif message.get_type() == "ATTITUDE":
        record.update(
            {
                "time_boot_ms": int(message.time_boot_ms),
                "roll": float(message.roll),
                "pitch": float(message.pitch),
                "yaw": float(message.yaw),
                "rollspeed": float(message.rollspeed),
                "pitchspeed": float(message.pitchspeed),
                "yawspeed": float(message.yawspeed),
            }
        )
    elif message.get_type() in {"AUTOPILOT_VERSION", "PROTOCOL_VERSION"}:
        record["payload"] = message.to_dict()
    return record


def collect_messages(
    adapter: MavlinkSitlAdapter,
    *,
    duration_s: float,
    heartbeat_hz: float,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if heartbeat_hz <= 0.0:
        raise ValueError("heartbeat_hz must be positive")

    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    heartbeat_period_s = 1.0 / heartbeat_hz
    next_heartbeat_s = 0.0
    deadline_s = time.monotonic() + duration_s
    while time.monotonic() < deadline_s:
        now_s = time.monotonic()
        if now_s >= next_heartbeat_s:
            adapter.send_heartbeat()
            next_heartbeat_s = now_s + heartbeat_period_s
        message = adapter.master.recv_match(blocking=True, timeout=0.02)
        if message is None:
            continue
        msg_type = message.get_type()
        counts[msg_type] += 1
        if msg_type in {
            "COMMAND_ACK",
            "MESSAGE_INTERVAL",
            "PING",
            "TIMESYNC",
            "ATTITUDE",
            "AUTOPILOT_VERSION",
            "PROTOCOL_VERSION",
        }:
            records.append(message_record(message, adapter.mavutil.mavlink))
    return counts, records


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=5.0)
    mavlink = adapter.mavutil.mavlink
    report: dict[str, Any] = {
        "contract": {
            "flight_commands_sent": 0,
            "reset_commands_sent": 0,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "persistent_stream_changes_sent": 0,
            "position_or_sim_truth_requests_sent": 0,
        },
        "endpoint": args.endpoint,
        "baseline": {},
        "interval_queries": [],
        "one_shot_requests": [],
        "timesync": {},
        "ping": {},
    }
    try:
        baseline_counts, baseline_records = collect_messages(
            adapter,
            duration_s=args.baseline_s,
            heartbeat_hz=args.heartbeat_hz,
        )
        if baseline_counts.get("HEARTBEAT", 0) <= 0:
            raise RuntimeError("no simulator HEARTBEAT received during baseline")
        report["baseline"] = {
            "message_counts": dict(sorted(baseline_counts.items())),
            "diagnostic_records": baseline_records,
            "learned_target_system": adapter._target_ids()[0],
            "learned_target_component": adapter._target_ids()[1],
        }

        for name in args.interval_message:
            message_id = mavlink_message_id(mavlink, name)
            send_message_interval_query(adapter, message_id)
            counts, records = collect_messages(
                adapter,
                duration_s=args.response_s,
                heartbeat_hz=args.heartbeat_hz,
            )
            report["interval_queries"].append(
                {
                    "requested_name": name,
                    "requested_id": message_id,
                    "message_counts": dict(sorted(counts.items())),
                    "diagnostic_records": records,
                }
            )

        for name in args.one_shot_message:
            message_id = mavlink_message_id(mavlink, name)
            send_request_message(adapter, message_id)
            counts, records = collect_messages(
                adapter,
                duration_s=args.response_s,
                heartbeat_hz=args.heartbeat_hz,
            )
            report["one_shot_requests"].append(
                {
                    "requested_name": name,
                    "requested_id": message_id,
                    "message_counts": dict(sorted(counts.items())),
                    "diagnostic_records": records,
                }
            )

        timesync_ts1 = time.time_ns()
        adapter.master.mav.timesync_send(0, timesync_ts1)
        counts, records = collect_messages(
            adapter,
            duration_s=args.response_s,
            heartbeat_hz=args.heartbeat_hz,
        )
        report["timesync"] = {
            "request_tc1": 0,
            "request_ts1": timesync_ts1,
            "message_counts": dict(sorted(counts.items())),
            "diagnostic_records": records,
        }

        ping_sequence = 1
        ping_time_usec = time.monotonic_ns() // 1000
        adapter.master.mav.ping_send(ping_time_usec, ping_sequence, 0, 0)
        counts, records = collect_messages(
            adapter,
            duration_s=args.response_s,
            heartbeat_hz=args.heartbeat_hz,
        )
        report["ping"] = {
            "request_time_usec": ping_time_usec,
            "request_sequence": ping_sequence,
            "message_counts": dict(sorted(counts.items())),
            "diagnostic_records": records,
        }
        return report
    finally:
        adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe MAVLink discovery without reset, arming, or flight control"
    )
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--baseline-s", type=float, default=2.0)
    parser.add_argument("--response-s", type=float, default=1.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument(
        "--interval-message",
        action="append",
        default=None,
        help="TS-002 message name whose current interval is queried",
    )
    parser.add_argument(
        "--one-shot-message",
        action="append",
        default=None,
        help="metadata or TS-002 message name to request once",
    )
    parser.add_argument("--json-path", required=True)
    args = parser.parse_args()
    if args.interval_message is None:
        args.interval_message = list(DEFAULT_INTERVAL_MESSAGES)
    if args.one_shot_message is None:
        args.one_shot_message = list(DEFAULT_ONE_SHOT_MESSAGES)
    return args


def main() -> int:
    args = parse_args()
    report = run_probe(args)
    output_path = Path(args.json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
