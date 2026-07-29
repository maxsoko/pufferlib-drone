import numpy as np

from scripts.analyze_vq2_n399_gate2_failure import (
    decode_gate_vector,
    first_phase,
    unique_pose_jumps,
)


def _sample(elapsed_s, phase, pose):
    return {
        "elapsed_s": elapsed_s,
        "official_active_gate_index": phase,
        "gate_pose": {"body_vector_ned_m": pose},
    }


def test_decode_gate_vector_inverts_public_tanh_scales():
    observation = np.zeros(32, dtype=np.float32)
    expected = np.asarray([14.7, 8.6, -1.2], dtype=np.float32)
    observation[11:14] = np.tanh(expected / np.asarray([10.0, 5.0, 5.0]))

    np.testing.assert_allclose(decode_gate_vector(observation), expected, atol=2e-6)


def test_first_phase_finds_official_transition():
    samples = [
        _sample(0.0, 0, [9.0, 0.0, 0.0]),
        _sample(3.2, 0, [14.7, 8.6, -1.2]),
        _sample(3.3, 1, [8.5, 5.0, -0.4]),
    ]

    assert first_phase(samples, 1) == 2


def test_unique_pose_jumps_ignores_repeated_camera_pose():
    samples = [
        _sample(3.20, 0, [14.7, 8.6, -1.2]),
        _sample(3.22, 0, [14.7, 8.6, -1.2]),
        _sample(3.25, 1, [8.5, 5.0, -0.4]),
    ]

    jumps = unique_pose_jumps(samples)

    assert len(jumps) == 1
    assert abs(jumps[0]["dt_s"] - 0.05) < 1e-9
    assert jumps[0]["jump_m"] > 7.0
