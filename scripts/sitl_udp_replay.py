#!/usr/bin/env python3
"""Replay captured MAVLink/camera UDP events with deterministic timing."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import random
import socket
import time


STREAM_MAVLINK = "mavlink"
STREAM_CAMERA = "camera"


@dataclasses.dataclass(frozen=True)
class ReplayEvent:
    dt_ns: int
    stream: str
    payload: bytes


@dataclasses.dataclass
class ReplayMetrics:
    loops: int
    speed: float
    loss_rate: float = 0.0
    reorder_rate: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    seed: int = 0
    events_loaded: int = 0
    events_sent: int = 0
    events_dropped: int = 0
    reorder_swaps: int = 0
    mavlink_packets: int = 0
    camera_packets: int = 0
    mavlink_bytes: int = 0
    camera_bytes: int = 0
    metadata_loaded: bool = False
    metadata_hash_match: bool | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def parse_event_line(line: str) -> ReplayEvent:
    obj = json.loads(line)
    dt_ns = int(obj["dt_ns"])
    stream = str(obj["stream"])
    payload = base64.b64decode(obj["payload_b64"].encode("ascii"))
    if stream not in {STREAM_MAVLINK, STREAM_CAMERA}:
        raise ValueError(f"unknown stream {stream!r}")
    return ReplayEvent(dt_ns=dt_ns, stream=stream, payload=payload)


def load_events(path: str) -> list[ReplayEvent]:
    events: list[ReplayEvent] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(parse_event_line(line))
    events.sort(key=lambda e: e.dt_ns)
    return events


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("metadata must decode to a JSON object")
    return payload


def materialize_impaired_schedule(
    events: list[ReplayEvent],
    *,
    speed: float,
    loss_rate: float,
    reorder_rate: float,
    latency_ms: float,
    jitter_ms: float,
    seed: int,
) -> tuple[list[tuple[int, int, ReplayEvent]], int, int]:
    rng = random.Random(seed)
    schedule: list[tuple[int, int, ReplayEvent]] = []
    dropped = 0
    for index, event in enumerate(events):
        if loss_rate > 0.0 and rng.random() < loss_rate:
            dropped += 1
            continue
        base_dt_ns = int(event.dt_ns / speed)
        jitter = rng.uniform(-jitter_ms, jitter_ms) if jitter_ms > 0.0 else 0.0
        delay_ms = max(0.0, latency_ms + jitter)
        schedule.append((base_dt_ns + int(delay_ms * 1_000_000.0), index, event))

    reorder_swaps = 0
    if reorder_rate > 0.0 and len(schedule) > 1:
        i = 0
        while i < len(schedule) - 1:
            if rng.random() < reorder_rate:
                a = schedule[i]
                b = schedule[i + 1]
                ta = a[0]
                tb = b[0]
                if tb <= ta:
                    tb = ta + 1
                schedule[i] = (tb, a[1], a[2])
                schedule[i + 1] = (ta, b[1], b[2])
                reorder_swaps += 1
                i += 2
            else:
                i += 1
    schedule.sort(key=lambda item: (item[0], item[1]))
    return schedule, dropped, reorder_swaps


def replay_events(
    events: list[ReplayEvent],
    *,
    mavlink_target: tuple[str, int],
    camera_target: tuple[str, int],
    speed: float,
    loops: int,
    loss_rate: float = 0.0,
    reorder_rate: float = 0.0,
    latency_ms: float = 0.0,
    jitter_ms: float = 0.0,
    seed: int = 20260530,
) -> ReplayMetrics:
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if loops <= 0:
        raise ValueError("loops must be positive")

    if not 0.0 <= loss_rate < 1.0:
        raise ValueError("loss_rate must be in [0, 1)")
    if not 0.0 <= reorder_rate < 1.0:
        raise ValueError("reorder_rate must be in [0, 1)")
    if latency_ms < 0.0:
        raise ValueError("latency_ms must be non-negative")
    if jitter_ms < 0.0:
        raise ValueError("jitter_ms must be non-negative")

    metrics = ReplayMetrics(
        loops=loops,
        speed=speed,
        loss_rate=loss_rate,
        reorder_rate=reorder_rate,
        latency_ms=latency_ms,
        jitter_ms=jitter_ms,
        seed=seed,
        events_loaded=len(events),
    )
    if not events:
        return metrics

    mav_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for loop_idx in range(loops):
            schedule, dropped, reordered = materialize_impaired_schedule(
                events,
                speed=speed,
                loss_rate=loss_rate,
                reorder_rate=reorder_rate,
                latency_ms=latency_ms,
                jitter_ms=jitter_ms,
                seed=seed + loop_idx,
            )
            metrics.events_dropped += dropped
            metrics.reorder_swaps += reordered
            if not schedule:
                continue

            loop_start_ns = time.monotonic_ns()
            for target_dt_ns, _order, event in schedule:
                target_ns = loop_start_ns + target_dt_ns
                while True:
                    now_ns = time.monotonic_ns()
                    remaining_ns = target_ns - now_ns
                    if remaining_ns <= 0:
                        break
                    time.sleep(min(0.001, remaining_ns / 1e9))
                if event.stream == STREAM_MAVLINK:
                    mav_sock.sendto(event.payload, mavlink_target)
                    metrics.mavlink_packets += 1
                    metrics.mavlink_bytes += len(event.payload)
                else:
                    cam_sock.sendto(event.payload, camera_target)
                    metrics.camera_packets += 1
                    metrics.camera_bytes += len(event.payload)
                metrics.events_sent += 1
            # Keep loop cadence deterministic relative to capture duration at current speed.
            scaled_max_ns = schedule[-1][0]
            loop_end_target_ns = loop_start_ns + scaled_max_ns
            while time.monotonic_ns() < loop_end_target_ns:
                time.sleep(0.0005)
    finally:
        mav_sock.close()
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
    parser = argparse.ArgumentParser(description="Replay captured UDP MAVLink/camera events")
    parser.add_argument("--input-path", required=True, help="JSONL capture file")
    parser.add_argument("--mavlink-host", default="127.0.0.1")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--reorder-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--require-hash-match", action="store_true")
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    events = load_events(args.input_path)
    metadata = None
    hash_match = None
    if args.metadata_path:
        metadata = load_metadata(args.metadata_path)
        expected_hash = metadata.get("output_sha256")
        actual_hash = sha256_file(args.input_path)
        hash_match = (expected_hash == actual_hash) if expected_hash is not None else None
        if args.require_hash_match and hash_match is not True:
            raise SystemExit("replay metadata hash mismatch")
    metrics = replay_events(
        events,
        mavlink_target=(args.mavlink_host, int(args.mavlink_port)),
        camera_target=(args.camera_host, int(args.camera_port)),
        speed=float(args.speed),
        loops=int(args.loops),
        loss_rate=float(args.loss_rate),
        reorder_rate=float(args.reorder_rate),
        latency_ms=float(args.latency_ms),
        jitter_ms=float(args.jitter_ms),
        seed=int(args.seed),
    )
    metrics.metadata_loaded = metadata is not None
    metrics.metadata_hash_match = hash_match
    payload = {
        "input_path": args.input_path,
        "metadata_path": args.metadata_path,
        "metadata": metadata,
        "metrics": metrics.to_dict(),
    }
    write_json(args.json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
