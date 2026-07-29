import numpy as np

from scripts.expand_gate_phase_dataset import expand_records


def test_expand_records_matches_runtime_phase_adapter_layout():
    source_observations = 26
    source = np.zeros((2, source_observations + 5), dtype=np.float32)
    source[:, 1] = (0.2, 0.3)
    source[:, 2] = (-0.4, -0.5)
    source[:, 12] = (0.6, 0.7)
    source[:, 13] = (-0.8, -0.9)
    source[0, 24] = 1.0
    source[1, 25] = 1.0
    source[:, source_observations : source_observations + 4] = (
        (0.1, 0.2, 0.3, 0.4),
        (-0.1, -0.2, -0.3, -0.4),
    )
    source[:, -1] = (1.0, 0.0)

    expanded = expand_records(source, source_observations)

    np.testing.assert_allclose(
        expanded[0, 23:32],
        (0.0, 1.0, 0.0, 0.6, 0.2, -0.8, -0.4, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        expanded[1, 23:32],
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3),
    )
    np.testing.assert_allclose(expanded[:, 32:36], source[:, 26:30])
    np.testing.assert_allclose(expanded[:, -1], source[:, -1])
