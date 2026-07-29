import numpy as np

from scripts.analyze_vq2_n483_corrected_geometry import (
    corrected_initial_gate_position_from_bearing,
    find_new_gate_association,
    rotate_vector_by_quaternion,
)


def _sample(elapsed_s, phase, body_ned, yaw):
    return {
        "elapsed_s": elapsed_s,
        "official_active_gate_index": phase,
        "gate_pose": {
            "body_vector_ned_m": body_ned,
            "yaw_error_rad": yaw,
        },
    }


def test_find_new_gate_association_skips_repeated_stale_pose():
    samples = [
        _sample(3.20, 0, [14.0, 8.0, -1.0], 0.52),
        _sample(3.28, 1, [14.0, 8.0, -1.0], 0.52),
        _sample(3.30, 1, [14.0, 8.0, -1.0], 0.52),
        _sample(4.26, 1, [30.0, 1.0, 0.2], 0.03),
    ]

    assert find_new_gate_association(samples) == 3


def test_bearing_correction_preserves_forward_range_and_displacement():
    correction = corrected_initial_gate_position_from_bearing(
        old_initial_world=[14.0, 8.0, 2.0],
        old_current_world=[10.0, 7.0, 1.0],
        old_current_body_ned=[10.0, 7.0, -1.0],
        raw_current_body_ned=[20.0, 2.0, 0.4],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        correction["corrected_body_ned"], [10.0, 1.0, 0.2])
    np.testing.assert_allclose(
        correction["corrected_current_world"], [10.0, 1.0, -0.2])
    np.testing.assert_allclose(
        correction["predictor_displacement_world"], [-4.0, -1.0, -1.0])
    np.testing.assert_allclose(
        correction["corrected_initial_world"], [14.0, 2.0, 0.8])


def test_rotate_vector_identity_quaternion():
    assert rotate_vector_by_quaternion(
        [1.0, -2.0, 3.0], [1.0, 0.0, 0.0, 0.0]
    ) == (1.0, -2.0, 3.0)
