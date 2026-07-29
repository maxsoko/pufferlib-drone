#!/usr/bin/env python3
"""Inventory VQ2 MAVLink and camera streams without sending any packet."""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

from drone_camera_receiver import UdpCameraReceiver
from drone_sitl_adapter import MavlinkSitlAdapter


def json_safe(value, *, path: str = "", nonfinite_paths: list[str] | None = None):
    if nonfinite_paths is None:
        nonfinite_paths = []
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(
                item,
                path=f"{path}.{key}" if path else str(key),
                nonfinite_paths=nonfinite_paths,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_safe(
                item,
                path=f"{path}[{index}]",
                nonfinite_paths=nonfinite_paths,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite_paths.append(path)
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def finite_vector(value, length: int) -> bool:
    return bool(
        value is not None
        and len(value) == length
        and all(item is not None and math.isfinite(float(item)) for item in value)
    )


def compact_message(message) -> dict[str, object]:
    payload = message.to_dict()
    if "data" in payload:
        data = list(payload["data"])
        payload["data"] = {
            "length": len(data),
            "prefix": data[:16],
        }
    if "actuator" in payload:
        actuator = list(payload["actuator"])
        payload["actuator"] = {
            "length": len(actuator),
            "values": actuator,
        }
    return payload


def jpeg_dimensions(jpeg: bytes) -> list[int] | None:
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        return [int(width), int(height)]
    except Exception:
        return None


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.dropout_after_s)
    camera = UdpCameraReceiver(
        host=args.camera_host,
        port=args.camera_port,
        socket_timeout_s=0.0,
        receive_buffer_bytes=args.camera_receive_buffer_bytes,
    )
    message_counts: Counter[str] = Counter()
    first_samples: dict[str, object] = {}
    frame_records: list[dict[str, object]] = []
    first_frame_jpeg: bytes | None = None
    started_s = time.monotonic()
    deadline_s = started_s + args.duration_s
    try:
        while time.monotonic() < deadline_s:
            consumed = 0
            while consumed < args.max_mavlink_messages_per_loop:
                message = adapter.master.recv_match(blocking=False, timeout=0.0)
                if message is None:
                    break
                message_type = message.get_type()
                message_counts[message_type] += 1
                if message_type not in first_samples:
                    first_samples[message_type] = compact_message(message)
                # Deliberately call the parser directly: the adapter helper can
                # reply to TIMESYNC requests, which a passive probe must not do.
                adapter.telemetry.ingest(message)
                consumed += 1
            frames = camera.poll_frames(max_packets=args.max_camera_packets_per_loop)
            for frame in frames:
                if first_frame_jpeg is None:
                    first_frame_jpeg = frame.jpeg
                if len(frame_records) < args.max_frame_records:
                    frame_records.append(
                        {
                            "frame_id": int(frame.frame_id),
                            "sim_time_ns": int(frame.sim_time_ns),
                            "jpeg_bytes": len(frame.jpeg),
                            "jpeg_sha256": hashlib.sha256(frame.jpeg).hexdigest(),
                            "received_chunks": int(frame.received_chunks),
                            "total_chunks": int(frame.total_chunks),
                        }
                    )
            if consumed == 0 and not frames:
                time.sleep(args.idle_sleep_s)
    finally:
        camera.close()
        adapter.close()

    elapsed_s = time.monotonic() - started_s
    if first_frame_jpeg is not None and args.frame_path:
        frame_path = Path(args.frame_path)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(first_frame_jpeg)
    state = adapter.telemetry.state
    metrics = adapter.telemetry.metrics
    nonfinite_paths: list[str] = []
    safe_state = json_safe(state, path="latest_state", nonfinite_paths=nonfinite_paths)
    safe_samples = json_safe(
        first_samples, path="first_samples", nonfinite_paths=nonfinite_paths
    )
    camera_metrics = dataclasses.asdict(camera.reassembler.metrics)
    availability = {
        "heartbeat": metrics.heartbeats > 0,
        "attitude": bool(
            metrics.attitudes > 0
            and finite_vector((state.roll, state.pitch, state.yaw), 3)
            and finite_vector((state.rollspeed, state.pitchspeed, state.yawspeed), 3)
        ),
        "highres_imu_accel_gyro": bool(
            metrics.highres_imus > 0
            and finite_vector((state.xacc, state.yacc, state.zacc), 3)
            and finite_vector((state.xgyro, state.ygyro, state.zgyro), 3)
        ),
        "local_position_ned": bool(
            metrics.local_positions > 0
            and finite_vector(state.local_position_ned_m, 3)
            and finite_vector(state.local_velocity_ned_m_s, 3)
        ),
        "odometry": bool(
            metrics.odometries > 0
            and finite_vector(state.odometry_position_ned_m, 3)
            and finite_vector(state.odometry_quaternion_wxyz, 4)
            and finite_vector(state.odometry_velocity_ned_m_s, 3)
        ),
        "actuator_output_status": bool(
            metrics.actuator_outputs > 0 and state.actuator_outputs is not None
        ),
        "collision_stream": metrics.collisions > 0,
        "race_status": bool(metrics.race_statuses > 0 and state.race_status is not None),
        "track_transfer": bool(metrics.track_infos > 0 and state.track_gates),
        "camera": camera_metrics["completed_frames"] > 0,
    }
    return {
        "contract": {
            "kind": "command_free_vq2_passive_interface_inventory",
            "heartbeats_sent": 0,
            "timesync_replies_sent": 0,
            "metadata_requests_sent": 0,
            "reset_commands_sent": 0,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "setpoints_sent": 0,
        },
        "endpoint": args.endpoint,
        "camera_endpoint": f"{args.camera_host}:{args.camera_port}",
        "duration_s": elapsed_s,
        "message_counts": dict(sorted(message_counts.items())),
        "message_rates_hz": {
            key: value / elapsed_s for key, value in sorted(message_counts.items())
        },
        "first_samples": safe_samples,
        "telemetry_metrics": dataclasses.asdict(metrics),
        "availability": availability,
        "latest_state": safe_state,
        "nonfinite_paths": sorted(set(nonfinite_paths)),
        "camera": {
            "metrics": camera_metrics,
            "frame_records": frame_records,
            "first_frame_dimensions": (
                jpeg_dimensions(first_frame_jpeg) if first_frame_jpeg is not None else None
            ),
            "first_frame_path": args.frame_path if first_frame_jpeg is not None else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--dropout-after-s", type=float, default=2.0)
    parser.add_argument("--max-mavlink-messages-per-loop", type=int, default=512)
    parser.add_argument("--max-camera-packets-per-loop", type=int, default=4096)
    parser.add_argument("--camera-receive-buffer-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-frame-records", type=int, default=12)
    parser.add_argument("--idle-sleep-s", type=float, default=0.0005)
    parser.add_argument("--frame-path", default="")
    parser.add_argument("--json-path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(args)
    output = Path(args.json_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
