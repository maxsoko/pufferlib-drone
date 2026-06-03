#!/usr/bin/env python3
"""Local TS-002 mock stream generator (MAVLink telemetry + camera UDP)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import socket
import struct
import time

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None
    np = None


TS002_CAMERA_HEADER_FORMAT = "<IHHIIQ"
TS002_CAMERA_HEADER_SIZE = struct.calcsize(TS002_CAMERA_HEADER_FORMAT)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360


@dataclasses.dataclass
class MockStreamMetrics:
    duration_s: float
    heartbeat_sent: int = 0
    attitude_sent: int = 0
    highres_imu_sent: int = 0
    timesync_sent: int = 0
    camera_frames_sent: int = 0
    camera_packets_sent: int = 0
    camera_bytes_sent: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def scheduled_gate_size_for_phase(phase: float) -> tuple[int, int]:
    phase = max(0.0, min(1.0, float(phase)))
    if phase <= 0.25:
        return (120, 120)
    if phase >= 0.65:
        return (520, 340)
    t = (phase - 0.25) / (0.65 - 0.25)
    width = int(round(120 + t * (520 - 120)))
    height = int(round(120 + t * (340 - 120)))
    return (width, height)


def scheduled_gate_size(elapsed_s: float, total_s: float) -> tuple[int, int]:
    """Start small (far), then grow to trigger pass-range in smoke tracker."""
    if total_s <= 0.0:
        return (120, 120)
    progress = max(0.0, min(1.0, elapsed_s / total_s))
    return scheduled_gate_size_for_phase(progress)


def build_gate_image(
    gate_width_px: int,
    gate_height_px: int,
    *,
    border_px: int = 10,
    center_offset_x_px: int = 0,
    center_offset_y_px: int = 0,
    rotate_deg: float = 0.0,
) -> np.ndarray:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV (cv2) and numpy are required for camera-frame generation")
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    gate_width_px = max(20, min(IMAGE_WIDTH - 4, int(gate_width_px)))
    gate_height_px = max(20, min(IMAGE_HEIGHT - 4, int(gate_height_px)))
    cx = (IMAGE_WIDTH // 2) + int(center_offset_x_px)
    cy = (IMAGE_HEIGHT // 2) + int(center_offset_y_px)
    cx = max(gate_width_px // 2 + 2, min(IMAGE_WIDTH - gate_width_px // 2 - 2, cx))
    cy = max(gate_height_px // 2 + 2, min(IMAGE_HEIGHT - gate_height_px // 2 - 2, cy))
    x0 = cx - gate_width_px // 2
    y0 = cy - gate_height_px // 2
    x1 = x0 + gate_width_px
    y1 = y0 + gate_height_px
    # Filled outer rectangle + dark interior gives robust contours across scales.
    cv2.rectangle(image, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=-1)
    inner_x0 = min(x1 - 1, x0 + max(2, border_px))
    inner_y0 = min(y1 - 1, y0 + max(2, border_px))
    inner_x1 = max(x0 + 1, x1 - max(2, border_px))
    inner_y1 = max(y0 + 1, y1 - max(2, border_px))
    if inner_x1 > inner_x0 and inner_y1 > inner_y0:
        cv2.rectangle(image, (inner_x0, inner_y0), (inner_x1, inner_y1), color=(0, 0, 0), thickness=-1)
    if abs(float(rotate_deg)) > 1e-6:
        matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rotate_deg), 1.0)
        image = cv2.warpAffine(image, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return image


def encode_jpeg(image: np.ndarray, *, quality: int = 90) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for JPEG encoding")
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to encode JPEG")
    return encoded.tobytes()


def make_camera_packets(
    frame_id: int,
    jpeg: bytes,
    sim_time_ns: int,
    *,
    max_payload_bytes: int = 1200,
) -> list[bytes]:
    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    total_chunks = int(math.ceil(len(jpeg) / max_payload_bytes))
    total_chunks = max(1, total_chunks)
    if total_chunks > 0xFFFF:
        raise ValueError("total_chunks exceeds TS-002 uint16 header capacity")

    packets: list[bytes] = []
    for chunk_id in range(total_chunks):
        start = chunk_id * max_payload_bytes
        end = min(len(jpeg), start + max_payload_bytes)
        payload = jpeg[start:end]
        header = struct.pack(
            TS002_CAMERA_HEADER_FORMAT,
            int(frame_id) & 0xFFFFFFFF,
            int(chunk_id) & 0xFFFF,
            int(total_chunks) & 0xFFFF,
            len(jpeg),
            len(payload),
            int(sim_time_ns) & 0xFFFFFFFFFFFFFFFF,
        )
        packets.append(header + payload)
    return packets


def run_stream(args) -> MockStreamMetrics:
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("pymavlink is required for mock MAVLink telemetry streaming") from exc

    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.heartbeat_hz <= 0.0 or args.telemetry_hz <= 0.0 or args.camera_fps <= 0.0:
        raise ValueError("heartbeat_hz, telemetry_hz, and camera_fps must be positive")
    if args.num_gates <= 0:
        raise ValueError("num_gates must be positive")

    mav = mavutil.mavlink_connection(
        f"udpout:{args.mavlink_host}:{args.mavlink_port}",
        source_system=args.source_system,
        source_component=args.source_component,
        dialect="common",
    )
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    metrics = MockStreamMetrics(duration_s=float(args.duration))
    started_s = time.monotonic()
    deadline_s = started_s + float(args.duration)
    next_heartbeat = started_s
    next_telemetry = started_s
    next_camera = started_s
    frame_id = 0

    heartbeat_period = 1.0 / float(args.heartbeat_hz)
    telemetry_period = 1.0 / float(args.telemetry_hz)
    camera_period = 1.0 / float(args.camera_fps)

    try:
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            elapsed_s = now_s - started_s

            if now_s >= next_heartbeat:
                mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                    0,
                    0,
                    mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                metrics.heartbeat_sent += 1
                next_heartbeat = now_s + heartbeat_period

            if now_s >= next_telemetry:
                roll = 0.0
                pitch = 0.0
                yaw = 0.0
                rollspeed = 0.0
                pitchspeed = 0.0
                yawspeed = 0.0
                now_ns = time.time_ns()
                time_boot_ms = int(elapsed_s * 1000.0) & 0xFFFFFFFF

                mav.mav.attitude_send(
                    time_boot_ms,
                    roll,
                    pitch,
                    yaw,
                    rollspeed,
                    pitchspeed,
                    yawspeed,
                )
                metrics.attitude_sent += 1

                mav.mav.highres_imu_send(
                    now_ns & 0xFFFFFFFFFFFFFFFF,
                    0.0, 0.0, -9.81,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    1013.25,
                    0.0,
                    0.0,
                    25.0,
                    0xFFFF,
                )
                metrics.highres_imu_sent += 1

                # Send request-style TIMESYNC so the adapter may reply.
                sim_time_ns = now_ns & 0x7FFFFFFFFFFFFFFF
                mav.mav.timesync_send(0, sim_time_ns)
                metrics.timesync_sent += 1
                next_telemetry = now_s + telemetry_period

            if now_s >= next_camera:
                progress = max(0.0, min(1.0, elapsed_s / float(args.duration)))
                gate_progress = progress * max(1, int(args.num_gates))
                local_phase = gate_progress - math.floor(gate_progress)
                width_px, height_px = scheduled_gate_size_for_phase(local_phase)
                drift_phase = 2.0 * math.pi * progress
                drift_x = int(round(float(args.viewpoint_drift_px) * math.sin(drift_phase)))
                drift_y = int(round(0.4 * float(args.viewpoint_drift_px) * math.sin(drift_phase + 0.5 * math.pi)))
                rotate_deg = float(args.yaw_drift_deg) * math.sin(drift_phase)
                image = build_gate_image(
                    width_px,
                    height_px,
                    border_px=args.gate_border_px,
                    center_offset_x_px=drift_x,
                    center_offset_y_px=drift_y,
                    rotate_deg=rotate_deg,
                )
                jpeg = encode_jpeg(image, quality=args.jpeg_quality)
                sim_time_ns = int(time.time_ns()) & 0x7FFFFFFFFFFFFFFF
                packets = make_camera_packets(
                    frame_id,
                    jpeg,
                    sim_time_ns,
                    max_payload_bytes=args.max_camera_payload_bytes,
                )
                for packet in packets:
                    cam_sock.sendto(packet, (args.camera_host, int(args.camera_port)))
                    metrics.camera_packets_sent += 1
                    metrics.camera_bytes_sent += len(packet)
                metrics.camera_frames_sent += 1
                frame_id += 1
                next_camera = now_s + camera_period

            time.sleep(0.0005)
    finally:
        cam_sock.close()

    return metrics


def write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock TS-002 stream generator for local SITL pipeline tests")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--mavlink-host", default="127.0.0.1")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--telemetry-hz", type=float, default=30.0)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--source-system", type=int, default=1)
    parser.add_argument("--source-component", type=int, default=1)
    parser.add_argument("--gate-border-px", type=int, default=10)
    parser.add_argument("--num-gates", type=int, default=1)
    parser.add_argument("--viewpoint-drift-px", type=int, default=0)
    parser.add_argument("--yaw-drift-deg", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-camera-payload-bytes", type=int, default=1200)
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    metrics = run_stream(args)
    payload = {"metrics": metrics.to_dict()}
    write_json(args.json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
