import numpy as np

from pufferlib.environments.drone_race.gate_progress import is_valid_gate_crossing


def test_valid_crossing_through_center_front_to_back():
    gate_center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gate_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prev_pos = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    curr_pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    assert is_valid_gate_crossing(
        prev_position=prev_pos,
        position=curr_pos,
        gate_center=gate_center,
        gate_normal=gate_normal,
        gate_radius=1.0,
        plane_cross_tolerance=1e-3,
        direction_min=0.05,
    )


def test_invalid_crossing_outside_gate_aperture():
    gate_center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gate_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prev_pos = np.array([-1.0, 1.5, 1.0], dtype=np.float32)
    curr_pos = np.array([1.0, 1.5, 1.0], dtype=np.float32)

    assert not is_valid_gate_crossing(
        prev_position=prev_pos,
        position=curr_pos,
        gate_center=gate_center,
        gate_normal=gate_normal,
        gate_radius=1.0,
        plane_cross_tolerance=1e-3,
        direction_min=0.05,
    )


def test_invalid_crossing_when_moving_backwards():
    gate_center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gate_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prev_pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    curr_pos = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    assert not is_valid_gate_crossing(
        prev_position=prev_pos,
        position=curr_pos,
        gate_center=gate_center,
        gate_normal=gate_normal,
        gate_radius=1.0,
        plane_cross_tolerance=1e-3,
        direction_min=0.05,
    )


def test_invalid_crossing_when_motion_too_small_for_direction():
    gate_center = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gate_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    prev_pos = np.array([-0.0001, 0.0, 1.0], dtype=np.float32)
    curr_pos = np.array([0.0001, 0.0, 1.0], dtype=np.float32)

    assert not is_valid_gate_crossing(
        prev_position=prev_pos,
        position=curr_pos,
        gate_center=gate_center,
        gate_normal=gate_normal,
        gate_radius=1.0,
        plane_cross_tolerance=1e-3,
        direction_min=0.99,
    )
