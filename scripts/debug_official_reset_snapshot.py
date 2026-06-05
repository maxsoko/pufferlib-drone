#!/usr/bin/env python3
"""Capture official simulator reset/start diagnostics.

This script is intentionally diagnostic: it records camera frames, detector
results, MAVLink telemetry, and simulator race status around reset/start so we
can explain why a controller run does or does not advance the official gate
index.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

from drone_camera_receiver import UdpCameraReceiver
from drone_gate_detector import SquareGateDetector
from drone_sitl_adapter import MavlinkSitlAdapter

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _maybe_send_reset(args) -> bool:
    if not args.send_sim_reset:
        return False
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    try:
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
    finally:
        adapter.close()
    if args.post_reset_sleep_s > 0.0:
        time.sleep(args.post_reset_sleep_s)
    return True


def _annotate_jpeg(jpeg: bytes, detection, path: Path) -> bool:
    if cv2 is None or np is None:
        return False
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    if detection is not None:
        points = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"conf={detection.confidence:.3f}"
        cv2.putText(
            image,
            label,
            (max(0, int(points[:, 0, 0].min())), max(20, int(points[:, 0, 1].min()) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def _jpeg_stats(jpeg: bytes) -> dict:
    if cv2 is None or np is None:
        return {}
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"decode_failed": True}
    mean = float(image.mean())
    std = float(image.std())
    min_px = int(image.min())
    max_px = int(image.max())
    return {
        "decode_failed": False,
        "mean_luma": mean,
        "std_luma": std,
        "min_luma": min_px,
        "max_luma": max_px,
        "black_frame_likely": mean < 2.0 and std < 2.0 and max_px < 8,
    }


def _save_frame(output_dir: Path, frame, detection, label: str) -> dict:
    jpeg_path = output_dir / f"{label}.jpg"
    annotated_path = output_dir / f"{label}_annotated.jpg"
    jpeg_path.write_bytes(frame.jpeg)
    annotated_written = _annotate_jpeg(frame.jpeg, detection, annotated_path)
    return {
        "label": label,
        "frame_id": frame.frame_id,
        "sim_time_ns": frame.sim_time_ns,
        "jpeg_path": str(jpeg_path),
        "annotated_path": str(annotated_path) if annotated_written else "",
        "jpeg_bytes": len(frame.jpeg),
        "image_stats": _jpeg_stats(frame.jpeg),
        "detection": dataclasses.asdict(detection) if detection is not None else None,
    }


def run_capture(args) -> dict:
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    if args.max_saved_frames < 0:
        raise ValueError("--max-saved-frames must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_unix_s = time.time()
    reset_sent = _maybe_send_reset(args)

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    receiver = UdpCameraReceiver(
        host=args.camera_host,
        port=args.camera_port,
        socket_timeout_s=args.camera_timeout_s,
    )
    detector = SquareGateDetector(
        min_area_px=args.detector_min_area_px,
        max_aspect_error=args.detector_max_aspect_error,
        min_fill_ratio=args.detector_min_fill_ratio,
    )

    heartbeat_period_s = 1.0 / args.heartbeat_hz
    next_heartbeat_s = 0.0
    deadline_s = time.monotonic() + args.duration_s
    saved_frames: list[dict] = []
    frames_seen = 0
    detections_seen = 0
    first_frame_saved = False
    first_detection_saved = False
    last_frame = None
    last_detection = None

    try:
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if now_s >= next_heartbeat_s:
                adapter.send_heartbeat()
                next_heartbeat_s = now_s + heartbeat_period_s

            adapter.poll_telemetry(timeout_s=args.telemetry_timeout_s)
            frames = receiver.poll_frames(max_packets=args.camera_max_packets_per_loop)
            if frames:
                frame = frames[-1]
                frames_seen += len(frames)
                detection = detector.detect_jpeg(frame.jpeg)
                if detection is not None:
                    detections_seen += 1
                last_frame = frame
                last_detection = detection

                if not first_frame_saved and len(saved_frames) < args.max_saved_frames:
                    saved_frames.append(_save_frame(output_dir, frame, detection, "first_frame"))
                    first_frame_saved = True
                if (
                    detection is not None
                    and not first_detection_saved
                    and len(saved_frames) < args.max_saved_frames
                ):
                    saved_frames.append(_save_frame(output_dir, frame, detection, "first_detection"))
                    first_detection_saved = True
            if args.idle_sleep_s > 0.0:
                time.sleep(args.idle_sleep_s)
    finally:
        receiver.close()
        adapter.close()

    if last_frame is not None and len(saved_frames) < args.max_saved_frames:
        saved_frames.append(_save_frame(output_dir, last_frame, last_detection, "last_frame"))

    adapter.telemetry.update_ages()
    payload = {
        "started_unix_s": started_unix_s,
        "finished_unix_s": time.time(),
        "elapsed_s": round(time.time() - started_unix_s, 6),
        "reset_sent": reset_sent,
        "endpoint": args.endpoint,
        "camera_host": args.camera_host,
        "camera_port": args.camera_port,
        "duration_s": args.duration_s,
        "frames_seen": frames_seen,
        "detections_seen": detections_seen,
        "saved_frames": saved_frames,
        "telemetry": dataclasses.asdict(adapter.telemetry.metrics),
        "latest_telemetry": dataclasses.asdict(adapter.telemetry.state),
        "camera_stream_metrics": dataclasses.asdict(receiver.reassembler.metrics),
        "detector_metrics": dataclasses.asdict(detector.metrics),
    }
    _write_json(output_dir / "snapshot_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture official simulator reset/start diagnostics")
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--send-sim-reset", action="store_true")
    parser.add_argument("--post-reset-sleep-s", type=float, default=2.0)
    parser.add_argument("--max-saved-frames", type=int, default=3)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", default=os.path.join("logs", "sitl", "reset_snapshot"))
    args = parser.parse_args()

    payload = run_capture(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
