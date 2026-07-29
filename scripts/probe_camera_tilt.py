#!/usr/bin/env python3
"""Measure the real camera tilt of AI-GP Simulator v3379 from the launch pad.

The drone stays disarmed on the pad, so its attitude is the rest attitude.
Gate 1 sits straight ahead at a known-ish range with its center ~1.5-2.5 m
above the pad. The pixel row of the detected gate center then tells us the
camera boresight elevation directly:

    elevation_camera = -atan2(center_y - cy, fy)      (up positive)
    elevation_true   =  atan2(gate_center_z - eye_z, range)   (small, ~2-6 deg)
    camera_uptilt    =  elevation_true - elevation_camera

If the TS-002 20 deg uptilt is real, the gate must appear ~14-18 deg BELOW
image center (row ~290+ of 360). If the gate sits near image center, the
rendered camera is level and the 20 deg correction in the pose pipeline is
manufacturing a huge fictitious gate height.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_camera_receiver import UdpCameraReceiver
from drone_gate_detector import SquareGateDetector
from drone_sitl_adapter import MavlinkSitlAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=180.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--save-frame", default="logs/sitl/probe_camera_tilt.jpg")
    args = parser.parse_args()

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=5.0)
    camera = UdpCameraReceiver(host=args.camera_host, port=args.camera_port)
    detector = SquareGateDetector(
        min_area_px=300.0,
        max_aspect_error=0.8,
        min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )
    rows: list[float] = []
    cols: list[float] = []
    widths: list[float] = []
    sources: dict[str, int] = {}
    saved = False
    try:
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and adapter.telemetry.state.race_status is None:
            adapter.poll_telemetry(timeout_s=0.5)
        if args.reset:
            adapter.send_heartbeat()
            adapter.send_sim_reset_command()
            print("sim reset sent; sampling from the pad (never arming)")
            time.sleep(2.0)

        t_end = time.monotonic() + args.duration_s
        while time.monotonic() < t_end:
            adapter.poll_telemetry(timeout_s=0.01)
            frames = camera.poll_frames(max_packets=256)
            if not frames:
                time.sleep(0.005)
                continue
            det = detector.detect_jpeg(frames[-1].jpeg)
            if det is None:
                continue
            xs = [c[0] for c in det.corners]
            ys = [c[1] for c in det.corners]
            cx_px = sum(xs) / 4.0
            cy_px = sum(ys) / 4.0
            rows.append(cy_px)
            cols.append(cx_px)
            widths.append(float(det.bounding_width_px))
            sources[det.source] = sources.get(det.source, 0) + 1
            if not saved and args.save_frame:
                import cv2
                import numpy as np

                Path(args.save_frame).parent.mkdir(parents=True, exist_ok=True)
                img = cv2.imdecode(
                    np.frombuffer(frames[-1].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                cv2.drawMarker(img, (int(cx_px), int(cy_px)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.line(img, (0, int(args.cy)), (img.shape[1], int(args.cy)), (255, 255, 0), 1)
                cv2.imwrite(args.save_frame, img)
                saved = True
    finally:
        camera.close()
        adapter.close()

    if not rows:
        print("no gate detections from the pad")
        return 1

    row = statistics.median(rows)
    col = statistics.median(cols)
    width_px = statistics.median(widths)
    elev_cam_rad = -math.atan2(row - args.cy, args.fy)
    az_cam_rad = math.atan2(col - args.cx, args.fx)
    range_if_inner = 1.5 * args.fx / width_px
    range_if_outer = 2.7 * args.fx / width_px
    print(f"detections: {len(rows)}  sources: {sources}")
    print(f"gate center pixel: ({col:.1f}, {row:.1f})  width={width_px:.1f}px")
    print(f"camera-frame elevation of gate: {math.degrees(elev_cam_rad):+.2f} deg (up positive)")
    print(f"camera-frame azimuth  of gate: {math.degrees(az_cam_rad):+.2f} deg (right positive)")
    print(f"range if aperture (1.5 m): {range_if_inner:.1f} m; if frame (2.7 m): {range_if_outer:.1f} m")
    for uptilt in (0.0, 20.0):
        theta = math.radians(uptilt)
        elev_body = elev_cam_rad + theta
        height = range_if_inner * math.tan(elev_body)
        print(
            f"assumed uptilt {uptilt:4.1f} deg -> gate center {height:+.2f} m "
            f"relative to camera eye (aperture range)"
        )
    print("physical truth: gate center should be roughly +1 to +3 m above the pad camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
