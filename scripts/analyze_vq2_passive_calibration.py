#!/usr/bin/env python3
"""Measure VQ2 camera geometry from command-free stationary imagery.

The VQ2 HUD labels the camera as ``20 deg``, but that label does not establish
the optical transform used by the UDP stream.  This offline tool estimates the
world-horizontal vanishing point from low-saturation hangar structure, detects
the active gate, and reports the camera uptilt needed by the policy observation
frontend.  It never opens a socket or sends a simulator packet.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Sequence

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - reported by main
    cv2 = None
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drone_gate_detector import SquareGateDetector
from drone_visual_servo import estimate_gate_pose_from_corners


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_median(values: Sequence[tuple[float, float]]) -> float:
    """Return the deterministic weighted median of ``(value, weight)`` pairs."""

    usable = sorted(
        (float(value), float(weight))
        for value, weight in values
        if math.isfinite(float(value)) and float(weight) > 0.0
    )
    if not usable:
        raise ValueError("weighted median requires a positive finite weight")
    half = 0.5 * sum(weight for _value, weight in usable)
    cumulative = 0.0
    for value, weight in usable:
        cumulative += weight
        if cumulative >= half:
            return value
    return usable[-1][0]


def _line_from_segment(segment) -> tuple[float, float, float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in segment)
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("degenerate line segment")
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    return a / norm, b / norm, c / norm, length, 0.5 * (x1 + x2), math.degrees(
        math.atan2(dy, dx)
    )


def estimate_horizontal_vanishing_point(
    image,
    *,
    focal_y_px: float = 320.0,
    principal_y_px: float = 180.0,
) -> dict[str, object]:
    """Estimate the forward horizontal vanishing point from hangar linework."""

    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required")
    if image is None or len(image.shape) != 3:
        raise ValueError("expected a decoded BGR image")
    height, width = image.shape[:2]
    center_x = 0.5 * width
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    # Remove the saturated cyan corridor and red gate.  Their converging
    # graphics point at the gate, not necessarily the world horizon.
    edges[hsv[:, :, 1] > 80] = 0
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=25,
        minLineLength=25,
        maxLineGap=8,
    )
    if raw_lines is None:
        raise ValueError("no structural line segments found")

    # OpenCV has returned both (N, 1, 4) and (N, 4) layouts across its
    # supported Python builds.  Normalize either representation without
    # changing segment order.
    line_segments = np.asarray(raw_lines).reshape(-1, 4)
    left = []
    right = []
    for raw in line_segments:
        line = _line_from_segment(raw)
        _a, _b, _c, length, midpoint_x, angle_deg = line
        abs_angle = abs(angle_deg)
        if length < 25.0 or not 5.0 <= abs_angle <= 75.0:
            continue
        if midpoint_x < center_x and angle_deg > 0.0:
            left.append(line)
        elif midpoint_x > center_x and angle_deg < 0.0:
            right.append(line)

    intersections: list[tuple[float, float, float]] = []
    for line_left in left:
        for line_right in right:
            a, b, c, length_left, _midpoint_left, _angle_left = line_left
            d, e, f, length_right, _midpoint_right, _angle_right = line_right
            determinant = a * e - d * b
            if abs(determinant) < 0.05:
                continue
            x = (b * f - e * c) / determinant
            y = (c * d - f * a) / determinant
            if not (0.28 * width <= x <= 0.72 * width):
                continue
            if not (-0.15 * height <= y <= 0.70 * height):
                continue
            intersections.append((x, y, min(length_left, length_right)))
    if len(intersections) < 8:
        raise ValueError(
            f"insufficient structural intersections: {len(intersections)}"
        )

    initial_x = weighted_median([(x, weight) for x, _y, weight in intersections])
    initial_y = weighted_median([(y, weight) for _x, y, weight in intersections])
    residuals = [math.hypot(x - initial_x, y - initial_y) for x, y, _w in intersections]
    median_residual = statistics.median(residuals)
    inlier_radius_px = max(12.0, min(35.0, 2.5 * median_residual))
    inliers = [
        item
        for item, residual in zip(intersections, residuals)
        if residual <= inlier_radius_px
    ]
    if len(inliers) < 8:
        raise ValueError(f"insufficient vanishing-point inliers: {len(inliers)}")
    vanishing_x = weighted_median([(x, weight) for x, _y, weight in inliers])
    vanishing_y = weighted_median([(y, weight) for _x, y, weight in inliers])
    inlier_residuals = [
        math.hypot(x - vanishing_x, y - vanishing_y) for x, y, _weight in inliers
    ]
    camera_uptilt_deg = math.degrees(
        math.atan2(vanishing_y - principal_y_px, focal_y_px)
    )
    return {
        "vanishing_point_px": [vanishing_x, vanishing_y],
        "camera_uptilt_deg": camera_uptilt_deg,
        "left_segments": len(left),
        "right_segments": len(right),
        "candidate_intersections": len(intersections),
        "inlier_intersections": len(inliers),
        "inlier_radius_px": inlier_radius_px,
        "median_inlier_residual_px": statistics.median(inlier_residuals),
    }


def _read_inventory_accel(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    imu = payload.get("first_samples", {}).get("HIGHRES_IMU", {})
    accel = tuple(imu.get(axis) for axis in ("xacc", "yacc", "zacc"))
    if not all(value is not None and math.isfinite(float(value)) for value in accel):
        return None
    xacc, yacc, zacc = (float(value) for value in accel)
    magnitude = math.sqrt(xacc * xacc + yacc * yacc + zacc * zacc)
    return {
        "sample_m_s2": [xacc, yacc, zacc],
        "magnitude_m_s2": magnitude,
        "sensor_mount_pitch_deg": math.degrees(math.atan2(-xacc, -zacc)),
        "sensor_mount_roll_deg": math.degrees(
            math.atan2(yacc, math.sqrt(xacc * xacc + zacc * zacc))
        ),
        "interpretation": (
            "fixed IMU sensor-mount direction; not vehicle attitude or camera tilt"
        ),
    }


def analyze_image(path: Path) -> dict[str, object]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    horizon = estimate_horizontal_vanishing_point(image)
    detector = SquareGateDetector(
        min_area_px=300.0,
        max_aspect_error=0.8,
        min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )
    detection = detector.detect_bgr(image)
    gate = None
    if detection is not None:
        pose = estimate_gate_pose_from_corners(
            detection.corners,
            camera_uptilt_deg=float(horizon["camera_uptilt_deg"]),
        )
        gate = {
            "detection": dataclasses.asdict(detection),
            "pose_with_measured_uptilt": dataclasses.asdict(pose),
        }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "dimensions": [int(image.shape[1]), int(image.shape[0])],
        "horizon": horizon,
        "gate": gate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--inventory-json", type=Path)
    parser.add_argument("--json-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_reports = [analyze_image(path) for path in args.images]
    tilts = [float(report["horizon"]["camera_uptilt_deg"]) for report in image_reports]
    report = {
        "contract": {
            "kind": "offline_command_free_vq2_camera_calibration",
            "simulator_packets_sent": 0,
            "camera_intrinsics": {
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 180.0,
            },
        },
        "images": image_reports,
        "aggregate": {
            "image_count": len(image_reports),
            "camera_uptilt_deg_median": statistics.median(tilts),
            "camera_uptilt_deg_min": min(tilts),
            "camera_uptilt_deg_max": max(tilts),
        },
        "stationary_imu": _read_inventory_accel(args.inventory_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
