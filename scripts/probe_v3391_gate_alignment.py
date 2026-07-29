#!/usr/bin/env python3
"""Passively compare the visible Gate 1 aperture with v3391 telemetry geometry.

The probe intentionally sends no MAVLink messages. It records camera frames,
detected gate corners, visual pose estimates, and contemporaneous vehicle/race
state so simulator geometry can be audited before an actuated policy run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

from drone_camera_receiver import UdpCameraReceiver
from drone_gate_detector import SquareGateDetector
from drone_sitl_adapter import MavlinkSitlAdapter
from drone_visual_servo import estimate_gate_pose_from_corners


def _finite(values) -> bool:
    return values is not None and all(math.isfinite(float(value)) for value in values)


def _object_dict(value) -> dict | None:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    names = (
        "sim_boot_time_ms",
        "race_start_boot_time_ms",
        "race_finish_time_ns",
        "active_gate_index",
        "last_gate_race_time",
    )
    return {name: getattr(value, name) for name in names if hasattr(value, name)}


def run_probe(args) -> dict:
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    receiver = UdpCameraReceiver(host=args.camera_host, port=args.camera_port, socket_timeout_s=0.0)
    detector = SquareGateDetector(
        min_area_px=args.detector_min_area_px,
        max_aspect_error=args.detector_max_aspect_error,
        min_fill_ratio=args.detector_min_fill_ratio,
    )
    deadline_s = time.monotonic() + args.duration_s
    samples = []
    frames_seen = 0
    detections_seen = 0
    try:
        while time.monotonic() < deadline_s:
            adapter.poll_telemetry(timeout_s=0.0)
            frames = receiver.poll_frames(max_packets=args.camera_max_packets_per_loop)
            frames_seen += len(frames)
            if frames:
                frame = frames[-1]
                detection = detector.detect_jpeg(frame.jpeg)
                if detection is not None:
                    detections_seen += 1
                    pose = estimate_gate_pose_from_corners(
                        detection.corners,
                        gate_inner_width_m=(
                            args.gate_inner_width_m
                            if detection.source == "aperture"
                            else args.gate_outer_width_m
                        ),
                    )
                    state = adapter.telemetry.state
                    if _finite(state.local_position_ned_m):
                        frame_path = output_dir / f"frame_{frame.frame_id}.jpg"
                        frame_path.write_bytes(frame.jpeg)
                        samples.append(
                            {
                                "frame_id": frame.frame_id,
                                "sim_time_ns": frame.sim_time_ns,
                                "frame_path": str(frame_path),
                                "frame_sha256": hashlib.sha256(frame.jpeg).hexdigest(),
                                "detection": dataclasses.asdict(detection),
                                "visual_pose": dataclasses.asdict(pose),
                                "vehicle_position_ned_m": list(state.local_position_ned_m),
                                "vehicle_velocity_ned_m_s": list(state.local_velocity_ned_m_s),
                                "attitude_rpy_rad": [state.roll, state.pitch, state.yaw],
                                "race_status": _object_dict(state.race_status),
                            }
                        )
                        if len(samples) >= args.max_samples:
                            break
            if args.idle_sleep_s > 0.0:
                time.sleep(args.idle_sleep_s)
    finally:
        receiver.close()
        adapter.close()

    payload = {
        "contract": {
            "kind": "command_free_v3391_gate_alignment_probe",
            "heartbeats_sent": 0,
            "reset_commands_sent": 0,
            "arm_commands_sent": 0,
            "disarm_commands_sent": 0,
            "setpoints_sent": 0,
        },
        "accepted": bool(samples),
        "duration_s": args.duration_s,
        "frames_seen": frames_seen,
        "detections_seen": detections_seen,
        "samples": samples,
        "telemetry": dataclasses.asdict(adapter.telemetry.metrics),
        "camera_stream_metrics": dataclasses.asdict(receiver.reassembler.metrics),
        "detector_metrics": dataclasses.asdict(detector.metrics),
    }
    output_path = output_dir / "alignment_probe.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--gate-inner-width-m", type=float, default=1.5)
    parser.add_argument("--gate-outer-width-m", type=float, default=2.72)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", default="logs/sitl/n254_v3391_gate_alignment")
    return parser


def main() -> None:
    payload = run_probe(build_parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
