import numpy as np
import pytest

from scripts.distill_assisted_trace_dataset import assisted_trace_records


def test_assisted_trace_shifts_public_executed_actions_and_marks_resets():
    observations = np.zeros((6, 2, 32), dtype=np.float32)
    terminals = np.zeros((6, 2), dtype=np.float32)
    terminals[4, 0] = 1.0
    terminals[5, 1] = 1.0
    for tick in range(1, 6):
        observations[tick, :, 19:23] = tick

    records, lengths = assisted_trace_records(observations, terminals)

    assert lengths == [3, 4]
    assert records.shape == (7, 37)
    np.testing.assert_array_equal(records[:3, 32:36], [[1] * 4, [2] * 4, [3] * 4])
    np.testing.assert_array_equal(records[3:, 32:36], [[1] * 4, [2] * 4, [3] * 4, [4] * 4])
    np.testing.assert_array_equal(records[:, 36], [1, 0, 0, 1, 0, 0, 0])


def test_assisted_trace_rejects_missing_terminal_marker():
    observations = np.zeros((4, 1, 32), dtype=np.float32)
    terminals = np.zeros((4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="no terminal"):
        assisted_trace_records(observations, terminals)
