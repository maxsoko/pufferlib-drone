import json

import numpy as np

from scripts.build_vq2_gate2_prefixed_dataset import (
    ACTIONS,
    OBSERVATIONS,
    RECORD_WIDTH,
    build_prefixed_records,
    load_prefix,
)


def test_load_prefix_uses_only_public_trace_fields(tmp_path):
    report = tmp_path / "report.json"
    samples = []
    for index in range(2):
        observation = [0.0] * OBSERVATIONS
        observation[24] = 1.0
        action = [0.1 + index, 0.2, 0.3, 0.4]
        samples.append({
            "observation": observation,
            "normalized_action": action,
            "official_active_gate_index": 0,
        })
    report.write_text(json.dumps({"policy_trace": {"samples": samples}}))

    prefix = load_prefix(report)

    assert prefix.shape == (2, RECORD_WIDTH)
    np.testing.assert_allclose(prefix[:, OBSERVATIONS:OBSERVATIONS + ACTIONS], [
        [0.1, 0.2, 0.3, 0.4],
        [1.1, 0.2, 0.3, 0.4],
    ])
    np.testing.assert_array_equal(prefix[:, -1], [1.0, 0.0])


def test_stitch_preserves_prefix_and_replaces_only_gate2_initial_last_action():
    prefix = np.zeros((2, RECORD_WIDTH), dtype=np.float32)
    prefix[0, -1] = 1.0
    prefix[:, 24] = 1.0
    prefix[-1, OBSERVATIONS:OBSERVATIONS + ACTIONS] = [0.1, 0.2, 0.3, 0.4]
    teacher = np.zeros((3, RECORD_WIDTH), dtype=np.float32)
    teacher[0, -1] = 1.0
    teacher[:, 23] = 1.0 / 6.0
    teacher[:, 25] = 1.0
    teacher[:, OBSERVATIONS:OBSERVATIONS + ACTIONS] = [0.5, 0.6, 0.7, 0.8]

    records, lengths = build_prefixed_records(prefix, [teacher])

    assert lengths == [5]
    np.testing.assert_array_equal(records[:2], prefix)
    np.testing.assert_allclose(records[2, 19:23], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        records[2:, OBSERVATIONS:OBSERVATIONS + ACTIONS],
        [[0.5, 0.6, 0.7, 0.8]] * 3,
    )
    np.testing.assert_array_equal(records[:, -1], [1.0, 0.0, 0.0, 0.0, 0.0])
