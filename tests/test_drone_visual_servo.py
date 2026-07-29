import importlib.util
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "drone_visual_servo.py"
SPEC = importlib.util.spec_from_file_location("drone_visual_servo", MODULE_PATH)
visual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visual
SPEC.loader.exec_module(visual)


def test_estimates_centered_gate_range_from_ts002_intrinsics():
    corners = [
        [240.0, 100.0],
        [400.0, 100.0],
        [400.0, 260.0],
        [240.0, 260.0],
    ]

    pose = visual.estimate_gate_pose_from_corners(corners)

    # v3379's camera is level (probe_camera_tilt.py), so a centered gate is
    # straight ahead in the body frame with no vertical offset.
    assert pose.image_center_px == (320.0, 180.0)
    assert abs(pose.range_camera_m - 3.0) < 1e-6
    assert abs(pose.body_vector_ned_m[0] - 3.0) < 1e-6
    assert abs(pose.body_vector_ned_m[1]) < 1e-6
    assert abs(pose.body_vector_ned_m[2]) < 1e-6
    assert abs(pose.yaw_error_rad) < 1e-6


def test_shifted_gate_produces_lateral_error_and_yaw_command():
    # Gate right of center and above center: with the level v3379 camera the
    # command must translate right, climb (vz < 0 in NED), and yaw right.
    corners = [
        [280.0, 60.0],
        [440.0, 60.0],
        [440.0, 220.0],
        [280.0, 220.0],
    ]

    pose = visual.estimate_gate_pose_from_corners(corners)
    command = visual.visual_servo_command(pose, yaw_rad=0.0)

    assert pose.body_vector_ned_m[1] > 0.0
    assert pose.yaw_error_rad > 0.0
    assert command.vx > 0.0
    assert command.vy > 0.0
    assert command.vz < 0.0
    assert command.yaw_rate > 0.0


def test_body_velocity_rotates_into_local_ned_with_yaw():
    vx, vy, vz = visual.body_velocity_to_local_ned((1.0, 0.0, -0.2), yaw_rad=math.pi / 2.0)

    assert abs(vx) < 1e-6
    assert abs(vy - 1.0) < 1e-6
    assert vz == -0.2


def test_rejects_bad_corner_inputs():
    for corners in [[], [[1, 2]], [[1, 2, 3]] * 4]:
        try:
            visual.estimate_gate_pose_from_corners(corners)
        except ValueError:
            pass
        else:
            raise AssertionError("expected bad corners to fail")
