#!/usr/bin/env python3
"""Quick UDP probe for TS-002 SITL MAVLink + camera stream presence."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import select
import socket
import struct
import time


TS002_CAMERA_HEADER_FORMAT = "<IHHIIQ"
TS002_CAMERA_HEADER_SIZE = struct.calcsize(TS002_CAMERA_HEADER_FORMAT)


@dataclasses.dataclass
class PortStats:
    packets_seen: int = 0
    bytes_seen: int = 0
    first_packet_s: float | None = None
    last_packet_s: float | None = None


@dataclasses.dataclass
class MavlinkProbeStats(PortStats):
    mavlink_v2_packets: int = 0
    mavlink_v1_packets: int = 0


@dataclasses.dataclass
class CameraProbeStats(PortStats):
    ts002_header_packets: int = 0
    ts002_parse_failures: int = 0


@dataclasses.dataclass
class ProbeReport:
    duration_s: float
    host: str
    mavlink_port: int
    camera_port: int
    mavlink: MavlinkProbeStats
    camera: CameraProbeStats
    requirements_met: bool
    blockers: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _mark_packet(stats: PortStats, now_s: float, size: int) -> None:
    stats.packets_seen += 1
    stats.bytes_seen += size
    if stats.first_packet_s is None:
        stats.first_packet_s = now_s
    stats.last_packet_s = now_s


def classify_mavlink_datagram(packet: bytes) -> str | None:
    if not packet:
        return None
    marker = packet[0]
    if marker == 0xFD:  # MAVLink v2
        return "v2"
    if marker == 0xFE:  # MAVLink v1
        return "v1"
    return None


def looks_like_ts002_camera_packet(packet: bytes) -> bool:
    if len(packet) < TS002_CAMERA_HEADER_SIZE:
        return False
    try:
        _frame_id, chunk_id, total_chunks, jpeg_size, payload_size, _sim_time_ns = struct.unpack(
            TS002_CAMERA_HEADER_FORMAT,
            packet[:TS002_CAMERA_HEADER_SIZE],
        )
    except struct.error:
        return False
    payload_len = len(packet) - TS002_CAMERA_HEADER_SIZE
    if total_chunks <= 0:
        return False
    if chunk_id >= total_chunks:
        return False
    if jpeg_size <= 0:
        return False
    if payload_size != payload_len:
        return False
    if payload_size > jpeg_size:
        return False
    return True


def run_probe(
    *,
    host: str,
    mavlink_port: int,
    camera_port: int,
    duration_s: float,
    poll_timeout_s: float = 0.05,
) -> ProbeReport:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    mav_stats = MavlinkProbeStats()
    cam_stats = CameraProbeStats()

    mav_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mav_sock.bind((host, mavlink_port))
        cam_sock.bind((host, camera_port))
        mav_sock.setblocking(False)
        cam_sock.setblocking(False)
        sockets = [mav_sock, cam_sock]

        started_s = time.monotonic()
        deadline_s = started_s + duration_s
        while time.monotonic() < deadline_s:
            timeout = min(poll_timeout_s, max(0.0, deadline_s - time.monotonic()))
            readable, _w, _x = select.select(sockets, [], [], timeout)
            now_s = time.monotonic()
            for sock_ in readable:
                packet, _addr = sock_.recvfrom(65535)
                if sock_ is mav_sock:
                    _mark_packet(mav_stats, now_s, len(packet))
                    version = classify_mavlink_datagram(packet)
                    if version == "v2":
                        mav_stats.mavlink_v2_packets += 1
                    elif version == "v1":
                        mav_stats.mavlink_v1_packets += 1
                else:
                    _mark_packet(cam_stats, now_s, len(packet))
                    if looks_like_ts002_camera_packet(packet):
                        cam_stats.ts002_header_packets += 1
                    else:
                        cam_stats.ts002_parse_failures += 1
    finally:
        mav_sock.close()
        cam_sock.close()

    report = ProbeReport(
        duration_s=duration_s,
        host=host,
        mavlink_port=mavlink_port,
        camera_port=camera_port,
        mavlink=mav_stats,
        camera=cam_stats,
        requirements_met=True,
        blockers=[],
    )
    return report


def evaluate_probe_requirements(
    report: ProbeReport,
    *,
    require_mavlink: bool,
    require_camera: bool,
    require_ts002_header: bool,
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if require_mavlink and report.mavlink.packets_seen <= 0:
        blockers.append("no_mavlink_packets")
    if require_camera and report.camera.packets_seen <= 0:
        blockers.append("no_camera_packets")
    if require_ts002_header and report.camera.ts002_header_packets <= 0:
        blockers.append("no_ts002_camera_headers")
    return len(blockers) == 0, blockers


def write_json(path: str, report: ProbeReport) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe TS-002 SITL UDP streams (MAVLink + camera)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--require-mavlink", action="store_true")
    parser.add_argument("--require-camera", action="store_true")
    parser.add_argument("--require-ts002-header", action="store_true")
    args = parser.parse_args()

    report = run_probe(
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        duration_s=args.duration,
    )
    met, blockers = evaluate_probe_requirements(
        report,
        require_mavlink=args.require_mavlink,
        require_camera=args.require_camera,
        require_ts002_header=args.require_ts002_header,
    )
    report.requirements_met = met
    report.blockers = blockers

    write_json(args.json_path, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.requirements_met:
        raise SystemExit("probe requirements failed: " + ", ".join(report.blockers))


if __name__ == "__main__":
    main()
