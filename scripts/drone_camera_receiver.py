#!/usr/bin/env python3
"""TS-002 JPEG-over-UDP camera packet parsing and frame reassembly.

The PRD-derived TS-002 camera header is 24 little-endian bytes with fields:

    frame_id:uint32, chunk_id:uint16, total_chunks:uint16,
    jpeg_size:uint32, payload_size:uint32, sim_time_ns:uint64

The PDF is not present in this workspace, so this module keeps the implied field
order and sizes explicit and covered by synthetic packet tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import socket
import struct
import time
from collections.abc import Iterator


DEFAULT_CAMERA_UDP_PORT = 5600
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30.0
EXPECTED_FRAME_INTERVAL_NS = int(round(1_000_000_000 / CAMERA_FPS))

TS002_CAMERA_HEADER_FORMAT = "<IHHIIQ"
TS002_CAMERA_HEADER_SIZE = struct.calcsize(TS002_CAMERA_HEADER_FORMAT)


@dataclasses.dataclass(frozen=True)
class CameraIntrinsics:
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    vertical_fov_deg: float = 90.0
    distortion: tuple[float, ...] = ()


@dataclasses.dataclass(frozen=True)
class CameraExtrinsics:
    body_to_camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_uptilt_deg: float = 20.0


@dataclasses.dataclass(frozen=True)
class CameraCalibration:
    intrinsics: CameraIntrinsics = dataclasses.field(default_factory=CameraIntrinsics)
    extrinsics: CameraExtrinsics = dataclasses.field(default_factory=CameraExtrinsics)


@dataclasses.dataclass(frozen=True)
class CameraChunkHeader:
    frame_id: int
    chunk_id: int
    total_chunks: int
    jpeg_size: int
    payload_size: int
    sim_time_ns: int

    @classmethod
    def parse(cls, packet: bytes) -> tuple["CameraChunkHeader", bytes]:
        if len(packet) < TS002_CAMERA_HEADER_SIZE:
            raise ValueError(
                f"camera packet too short: {len(packet)} bytes, "
                f"expected at least {TS002_CAMERA_HEADER_SIZE}"
            )

        fields = struct.unpack(
            TS002_CAMERA_HEADER_FORMAT,
            packet[:TS002_CAMERA_HEADER_SIZE],
        )
        header = cls(*fields)
        payload = packet[TS002_CAMERA_HEADER_SIZE:]
        header.validate(payload_size=len(payload))
        return header, payload

    def to_bytes(self) -> bytes:
        self.validate(payload_size=self.payload_size)
        return struct.pack(
            TS002_CAMERA_HEADER_FORMAT,
            self.frame_id,
            self.chunk_id,
            self.total_chunks,
            self.jpeg_size,
            self.payload_size,
            self.sim_time_ns,
        )

    def validate(self, payload_size: int) -> None:
        if self.total_chunks <= 0:
            raise ValueError("total_chunks must be positive")
        if self.chunk_id >= self.total_chunks:
            raise ValueError(
                f"chunk_id {self.chunk_id} must be less than total_chunks {self.total_chunks}"
            )
        if self.jpeg_size <= 0:
            raise ValueError("jpeg_size must be positive")
        if self.payload_size != payload_size:
            raise ValueError(
                f"payload_size header={self.payload_size} does not match packet={payload_size}"
            )
        if self.payload_size > self.jpeg_size:
            raise ValueError("payload_size cannot exceed jpeg_size")


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    sim_time_ns: int
    jpeg: bytes
    received_chunks: int
    total_chunks: int
    calibration: CameraCalibration = dataclasses.field(default_factory=CameraCalibration)


@dataclasses.dataclass
class CameraStreamMetrics:
    packets_seen: int = 0
    chunks_accepted: int = 0
    malformed_packets: int = 0
    duplicate_chunks: int = 0
    completed_frames: int = 0
    reconstruction_failures: int = 0
    chunk_loss: int = 0
    incomplete_frames_evicted: int = 0
    out_of_order_frames: int = 0
    frame_stalls: int = 0
    last_frame_fps: float = 0.0
    average_frame_fps: float = 0.0
    average_jitter_ns: float = 0.0
    max_frame_gap_ns: int = 0


@dataclasses.dataclass
class _PendingFrame:
    header: CameraChunkHeader
    first_arrival_ns: int
    chunks: dict[int, bytes] = dataclasses.field(default_factory=dict)

    def missing_chunks(self) -> int:
        return self.header.total_chunks - len(self.chunks)


class JpegFrameReassembler:
    """Reassemble TS-002 camera frames and track stream health metrics."""

    def __init__(
        self,
        *,
        expected_fps: float = CAMERA_FPS,
        stall_after_intervals: float = 2.0,
        calibration: CameraCalibration | None = None,
    ) -> None:
        if expected_fps <= 0.0:
            raise ValueError("expected_fps must be positive")
        if stall_after_intervals <= 1.0:
            raise ValueError("stall_after_intervals must be greater than 1.0")

        self.expected_frame_interval_ns = int(round(1_000_000_000 / expected_fps))
        self.stall_after_intervals = stall_after_intervals
        self.calibration = calibration or CameraCalibration()
        self.metrics = CameraStreamMetrics()
        self._pending: dict[tuple[int, int], _PendingFrame] = {}
        self._last_completed_sim_time_ns: int | None = None
        self._fps_samples = 0

    @property
    def pending_frames(self) -> int:
        return len(self._pending)

    def ingest_packet(self, packet: bytes, *, arrival_time_ns: int | None = None) -> CameraFrame | None:
        self.metrics.packets_seen += 1
        try:
            header, payload = CameraChunkHeader.parse(packet)
        except ValueError:
            self.metrics.malformed_packets += 1
            return None
        return self.ingest_chunk(header, payload, arrival_time_ns=arrival_time_ns)

    def ingest_chunk(
        self,
        header: CameraChunkHeader,
        payload: bytes,
        *,
        arrival_time_ns: int | None = None,
    ) -> CameraFrame | None:
        now_ns = arrival_time_ns if arrival_time_ns is not None else time.monotonic_ns()
        try:
            header.validate(payload_size=len(payload))
        except ValueError:
            self.metrics.malformed_packets += 1
            return None

        key = (header.frame_id, header.sim_time_ns)
        pending = self._pending.get(key)
        if pending is None:
            pending = _PendingFrame(header=header, first_arrival_ns=now_ns)
            self._pending[key] = pending
        elif (
            pending.header.total_chunks != header.total_chunks
            or pending.header.jpeg_size != header.jpeg_size
        ):
            self.metrics.reconstruction_failures += 1
            self._pending.pop(key, None)
            return None

        if header.chunk_id in pending.chunks:
            self.metrics.duplicate_chunks += 1
            return None

        pending.chunks[header.chunk_id] = payload
        self.metrics.chunks_accepted += 1

        if len(pending.chunks) != header.total_chunks:
            return None

        jpeg = b"".join(pending.chunks[i] for i in range(header.total_chunks))
        self._pending.pop(key, None)
        if len(jpeg) != header.jpeg_size:
            self.metrics.reconstruction_failures += 1
            return None

        frame = CameraFrame(
            frame_id=header.frame_id,
            sim_time_ns=header.sim_time_ns,
            jpeg=jpeg,
            received_chunks=len(pending.chunks),
            total_chunks=header.total_chunks,
            calibration=self.calibration,
        )
        self._record_completed_frame(frame)
        return frame

    def evict_stale(self, *, now_ns: int | None = None, stale_after_ns: int) -> int:
        if stale_after_ns <= 0:
            raise ValueError("stale_after_ns must be positive")
        now = now_ns if now_ns is not None else time.monotonic_ns()
        stale_keys = [
            key for key, pending in self._pending.items()
            if now - pending.first_arrival_ns >= stale_after_ns
        ]
        for key in stale_keys:
            pending = self._pending.pop(key)
            missing = pending.missing_chunks()
            self.metrics.chunk_loss += max(0, missing)
            self.metrics.reconstruction_failures += 1
            self.metrics.incomplete_frames_evicted += 1
        return len(stale_keys)

    def _record_completed_frame(self, frame: CameraFrame) -> None:
        self.metrics.completed_frames += 1
        last_sim_time_ns = self._last_completed_sim_time_ns
        self._last_completed_sim_time_ns = frame.sim_time_ns
        if last_sim_time_ns is None:
            return

        delta_ns = frame.sim_time_ns - last_sim_time_ns
        if delta_ns <= 0:
            self.metrics.out_of_order_frames += 1
            return

        self.metrics.last_frame_fps = 1_000_000_000.0 / delta_ns
        self._fps_samples += 1
        sample_count = self._fps_samples
        self.metrics.average_frame_fps += (
            self.metrics.last_frame_fps - self.metrics.average_frame_fps
        ) / sample_count

        jitter_ns = abs(delta_ns - self.expected_frame_interval_ns)
        self.metrics.average_jitter_ns += (
            jitter_ns - self.metrics.average_jitter_ns
        ) / sample_count
        self.metrics.max_frame_gap_ns = max(self.metrics.max_frame_gap_ns, delta_ns)
        if delta_ns >= int(self.expected_frame_interval_ns * self.stall_after_intervals):
            self.metrics.frame_stalls += 1


class UdpCameraReceiver:
    """Thin UDP wrapper around the testable TS-002 frame reassembler."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_CAMERA_UDP_PORT,
        socket_timeout_s: float = 0.5,
        reassembler: JpegFrameReassembler | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.reassembler = reassembler or JpegFrameReassembler()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(socket_timeout_s)

    def close(self) -> None:
        self._socket.close()

    def _poll_packet(self) -> bytes | None:
        try:
            packet, _addr = self._socket.recvfrom(65535)
        except (socket.timeout, BlockingIOError):
            return None
        return packet

    def poll_frame(self) -> CameraFrame | None:
        """Receive at most one UDP packet and return a completed frame when available."""
        packet = self._poll_packet()
        if packet is None:
            return None
        return self.reassembler.ingest_packet(packet)

    def poll_frames(self, *, max_packets: int) -> list[CameraFrame]:
        """Drain up to max_packets UDP packets and return completed frames."""
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        frames = []
        for _ in range(max_packets):
            packet = self._poll_packet()
            if packet is None:
                break
            frame = self.reassembler.ingest_packet(packet)
            if frame is not None:
                frames.append(frame)
        return frames

    def frames(self, *, max_packets: int | None = None) -> Iterator[CameraFrame]:
        packets = 0
        while max_packets is None or packets < max_packets:
            packet = self._poll_packet()
            if packet is None:
                continue
            packets += 1
            frame = self.reassembler.ingest_packet(packet)
            if frame is not None:
                yield frame


def main() -> None:
    parser = argparse.ArgumentParser(description="TS-002 JPEG-over-UDP camera receiver scaffold")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_CAMERA_UDP_PORT)
    parser.add_argument("--max-packets", type=int, default=0)
    args = parser.parse_args()

    max_packets = args.max_packets or None
    receiver = UdpCameraReceiver(host=args.host, port=args.port)
    try:
        for frame in receiver.frames(max_packets=max_packets):
            metrics = receiver.reassembler.metrics
            print(
                f"frame_id={frame.frame_id} sim_time_ns={frame.sim_time_ns} "
                f"bytes={len(frame.jpeg)} completed={metrics.completed_frames} "
                f"fps={metrics.average_frame_fps:.2f} jitter_ns={metrics.average_jitter_ns:.0f}"
            )
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
