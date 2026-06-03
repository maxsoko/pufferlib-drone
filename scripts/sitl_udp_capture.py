#!/usr/bin/env python3
"""Capture MAVLink/camera UDP traffic into deterministic replay events."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import select
import socket
import time
import uuid


STREAM_MAVLINK = "mavlink"
STREAM_CAMERA = "camera"


@dataclasses.dataclass(frozen=True)
class UdpEvent:
    dt_ns: int
    stream: str
    payload_b64: str

    @staticmethod
    def from_payload(*, dt_ns: int, stream: str, payload: bytes) -> "UdpEvent":
        return UdpEvent(
            dt_ns=int(dt_ns),
            stream=stream,
            payload_b64=base64.b64encode(payload).decode("ascii"),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CaptureMetrics:
    duration_s: float
    events_written: int = 0
    mavlink_packets: int = 0
    camera_packets: int = 0
    mavlink_bytes: int = 0
    camera_bytes: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CaptureMetadata:
    capture_id: str
    output_path: str
    output_sha256: str
    created_unix_s: float
    duration_s: float
    host: str
    mavlink_port: int
    camera_port: int
    timing_profile: str
    generator_params: dict
    metrics: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_event_line(file_obj, event: UdpEvent) -> None:
    file_obj.write(json.dumps(event.to_dict(), sort_keys=True))
    file_obj.write("\n")


def capture_udp_events(
    *,
    output_path: str,
    duration_s: float,
    host: str,
    mavlink_port: int,
    camera_port: int,
) -> CaptureMetrics:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    mav_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mav_sock.bind((host, int(mavlink_port)))
        cam_sock.bind((host, int(camera_port)))
        mav_sock.setblocking(False)
        cam_sock.setblocking(False)
        sockets = [mav_sock, cam_sock]

        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(duration_s * 1e9)
        metrics = CaptureMetrics(duration_s=duration_s)

        with open(output_path, "w") as f:
            while time.monotonic_ns() < deadline_ns:
                timeout = max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9)
                timeout = min(0.05, timeout)
                readable, _w, _x = select.select(sockets, [], [], timeout)
                now_ns = time.monotonic_ns()
                for sock_ in readable:
                    payload, _addr = sock_.recvfrom(65535)
                    stream = STREAM_MAVLINK if sock_ is mav_sock else STREAM_CAMERA
                    event = UdpEvent.from_payload(
                        dt_ns=now_ns - started_ns,
                        stream=stream,
                        payload=payload,
                    )
                    write_event_line(f, event)
                    metrics.events_written += 1
                    if stream == STREAM_MAVLINK:
                        metrics.mavlink_packets += 1
                        metrics.mavlink_bytes += len(payload)
                    else:
                        metrics.camera_packets += 1
                        metrics.camera_bytes += len(payload)
        return metrics
    finally:
        mav_sock.close()
        cam_sock.close()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_capture_metadata(
    *,
    output_path: str,
    metrics: CaptureMetrics,
    host: str,
    mavlink_port: int,
    camera_port: int,
    timing_profile: str,
    generator_params: dict,
) -> CaptureMetadata:
    output_sha256 = sha256_file(output_path)
    capture_id = f"{uuid.uuid5(uuid.NAMESPACE_URL, output_sha256)}"
    return CaptureMetadata(
        capture_id=capture_id,
        output_path=output_path,
        output_sha256=output_sha256,
        created_unix_s=time.time(),
        duration_s=float(metrics.duration_s),
        host=host,
        mavlink_port=int(mavlink_port),
        camera_port=int(camera_port),
        timing_profile=str(timing_profile),
        generator_params=dict(generator_params),
        metrics=metrics.to_dict(),
    )


def write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture UDP MAVLink/camera traffic into replay events")
    parser.add_argument("--output-path", required=True, help="JSONL output path")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--timing-profile", default="normal")
    parser.add_argument(
        "--generator-params-json",
        default="{}",
        help="Optional JSON object describing the traffic generator used for this capture",
    )
    parser.add_argument("--require-mavlink", action="store_true")
    parser.add_argument("--require-camera", action="store_true")
    args = parser.parse_args()

    metrics = capture_udp_events(
        output_path=args.output_path,
        duration_s=args.duration,
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
    )
    generator_params = json.loads(args.generator_params_json)
    if not isinstance(generator_params, dict):
        raise ValueError("--generator-params-json must decode to a JSON object")
    metadata = build_capture_metadata(
        output_path=args.output_path,
        metrics=metrics,
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        timing_profile=args.timing_profile,
        generator_params=generator_params,
    )
    metadata_path = args.metadata_path or f"{args.output_path}.meta.json"
    write_json(metadata_path, metadata.to_dict())
    blockers: list[str] = []
    if args.require_mavlink and metrics.mavlink_packets <= 0:
        blockers.append("no_mavlink_packets")
    if args.require_camera and metrics.camera_packets <= 0:
        blockers.append("no_camera_packets")
    payload = {
        "output_path": args.output_path,
        "metadata_path": metadata_path,
        "capture_id": metadata.capture_id,
        "output_sha256": metadata.output_sha256,
        "metrics": metrics.to_dict(),
        "blockers": blockers,
        "requirements_met": len(blockers) == 0,
    }
    write_json(args.json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit("capture requirements failed: " + ", ".join(blockers))


if __name__ == "__main__":
    main()
