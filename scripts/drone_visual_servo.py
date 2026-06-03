#!/usr/bin/env python3
"""First-pass TS-002 gate pose and visual-servo helpers.

This module intentionally starts at the geometry boundary: it accepts detected
gate corner pixels and produces conservative local-NED velocity/yaw commands.
Image detection can feed these helpers later without changing the controller
contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections.abc import Sequence


TS002_GATE_INNER_WIDTH_M = 1.5
TS002_CAMERA_UPTILT_DEG = 20.0


@dataclasses.dataclass(frozen=True)
class CameraIntrinsics:
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0
    width: int = 640
    height: int = 360


@dataclasses.dataclass(frozen=True)
class GatePoseEstimate:
    image_center_px: tuple[float, float]
    image_width_px: float
    image_height_px: float
    range_camera_m: float
    body_vector_ned_m: tuple[float, float, float]
    yaw_error_rad: float
    confidence: float


@dataclasses.dataclass(frozen=True)
class LocalNedVelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_corners(corners: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    if len(corners) != 4:
        raise ValueError("expected exactly four gate corners")
    out = []
    for corner in corners:
        if len(corner) != 2:
            raise ValueError("each corner must be an [x, y] pair")
        out.append((float(corner[0]), float(corner[1])))
    return out


def estimate_gate_pose_from_corners(
    corners: Sequence[Sequence[float]],
    *,
    intrinsics: CameraIntrinsics | None = None,
    gate_inner_width_m: float = TS002_GATE_INNER_WIDTH_M,
    camera_uptilt_deg: float = TS002_CAMERA_UPTILT_DEG,
) -> GatePoseEstimate:
    intr = intrinsics or CameraIntrinsics()
    pts = normalize_corners(corners)
    if gate_inner_width_m <= 0.0:
        raise ValueError("gate_inner_width_m must be positive")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    center_px = (mean(xs), mean(ys))
    width_px = max(xs) - min(xs)
    height_px = max(ys) - min(ys)
    side_px = mean([
        distance(pts[0], pts[1]),
        distance(pts[1], pts[2]),
        distance(pts[2], pts[3]),
        distance(pts[3], pts[0]),
        width_px,
        height_px,
    ])
    if side_px <= 1e-6:
        raise ValueError("gate corners are degenerate")

    range_camera_m = gate_inner_width_m * intr.fx / side_px
    cam_x_m = (center_px[0] - intr.cx) * range_camera_m / intr.fx
    cam_y_m = (center_px[1] - intr.cy) * range_camera_m / intr.fy
    cam_z_m = range_camera_m
    body = camera_vector_to_body_ned(
        (cam_x_m, cam_y_m, cam_z_m),
        camera_uptilt_deg=camera_uptilt_deg,
    )
    yaw_error_rad = math.atan2(body[1], max(body[0], 1e-6))
    confidence = clamp(min(width_px, height_px) / max(intr.width, intr.height), 0.0, 1.0)

    return GatePoseEstimate(
        image_center_px=center_px,
        image_width_px=width_px,
        image_height_px=height_px,
        range_camera_m=range_camera_m,
        body_vector_ned_m=body,
        yaw_error_rad=yaw_error_rad,
        confidence=confidence,
    )


def camera_vector_to_body_ned(vector_camera_m, *, camera_uptilt_deg=TS002_CAMERA_UPTILT_DEG):
    """Convert image-library camera coordinates to body NED.

    Camera coordinates are x-right, y-down, z-forward. Body NED is x-forward,
    y-right, z-down. TS-002 states the camera is tilted 20 degrees upward
    relative to the body.
    """
    cam_x, cam_y, cam_z = vector_camera_m
    theta = math.radians(camera_uptilt_deg)
    body_x = math.cos(theta) * cam_z + math.sin(theta) * cam_y
    body_y = cam_x
    body_z = -math.sin(theta) * cam_z + math.cos(theta) * cam_y
    return (body_x, body_y, body_z)


def body_velocity_to_local_ned(v_body, *, yaw_rad):
    vx_body, vy_body, vz_body = v_body
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (
        c * vx_body - s * vy_body,
        s * vx_body + c * vy_body,
        vz_body,
    )


def visual_servo_command(
    pose: GatePoseEstimate,
    *,
    yaw_rad: float = 0.0,
    desired_standoff_m: float = 1.0,
    max_forward_m_s: float = 1.0,
    max_lateral_m_s: float = 0.5,
    max_vertical_m_s: float = 0.4,
    max_yaw_rate_rad_s: float = 0.6,
    k_forward: float = 0.45,
    k_lateral: float = 0.7,
    k_vertical: float = 0.7,
    k_yaw: float = 1.2,
) -> LocalNedVelocityCommand:
    if desired_standoff_m < 0.0:
        raise ValueError("desired_standoff_m must be non-negative")
    bx, by, bz = pose.body_vector_ned_m
    forward_error = max(0.0, bx - desired_standoff_m)
    vx_body = clamp(k_forward * forward_error, 0.0, max_forward_m_s)
    vy_body = clamp(k_lateral * by, -max_lateral_m_s, max_lateral_m_s)
    vz_body = clamp(k_vertical * bz, -max_vertical_m_s, max_vertical_m_s)
    vx, vy, vz = body_velocity_to_local_ned((vx_body, vy_body, vz_body), yaw_rad=yaw_rad)
    yaw_rate = clamp(k_yaw * pose.yaw_error_rad, -max_yaw_rate_rad_s, max_yaw_rate_rad_s)
    return LocalNedVelocityCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)


def main():
    parser = argparse.ArgumentParser(description="Estimate TS-002 gate pose and visual-servo command from corners")
    parser.add_argument("--corners-json", required=True, help="JSON list of four [x, y] gate corners")
    parser.add_argument("--yaw-rad", type=float, default=0.0)
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    corners = json.loads(args.corners_json)
    pose = estimate_gate_pose_from_corners(corners)
    command = visual_servo_command(pose, yaw_rad=args.yaw_rad)
    report = {
        "pose": dataclasses.asdict(pose),
        "command": dataclasses.asdict(command),
    }
    if args.json_path:
        directory = os.path.dirname(args.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
