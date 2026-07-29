from __future__ import annotations

import numpy as np

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, PRIVILEGED_NAMES
from pufferlib.vq2_n294_visual_composite import (
    LegacyN294ObservationAdapter,
    select_complete_actions,
    visual_suffix_observation,
)


def _environment_observation(
    *, gate=(10.78, 0.084, 0.56), phase=0.0, previous=(0.1, -0.2, 0.3, -0.4)
) -> np.ndarray:
    values = np.zeros((1, ENV_OBS_SIZE), dtype=np.float32)
    privileged = values[:, LEGAL_OBS_SIZE:]
    lookup = {name: index for index, name in enumerate(PRIVILEGED_NAMES)}
    privileged[0, lookup["gate_relative_body_x"] : lookup["gate_relative_body_z"] + 1] = np.tanh(
        np.asarray(gate, dtype=np.float32) / np.float32(10.0)
    )
    privileged[0, lookup["attitude_qw"]] = 1.0
    privileged[0, lookup["ordered_gate_phase"]] = phase
    values[0, 4103:4107] = np.asarray(previous, dtype=np.float32)
    return values


def test_visual_suffix_observation_excludes_privileged_tail():
    values = _environment_observation()
    first = visual_suffix_observation(values, np.asarray([1.0 / 6.0], dtype=np.float32))
    values[:, LEGAL_OBS_SIZE:] = 0.75
    second = visual_suffix_observation(values, np.asarray([1.0 / 6.0], dtype=np.float32))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 4119)
    assert first[0, -1] == np.float32(1.0 / 6.0)


def test_complete_action_selection_never_blends_channels():
    prefix = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    suffix = -prefix
    selected, suffix_rows = select_complete_actions(
        prefix, suffix, np.asarray([0.0, 1.0 / 6.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(selected[0], prefix[0])
    np.testing.assert_array_equal(selected[1], suffix[1])
    np.testing.assert_array_equal(suffix_rows, np.asarray([False, True]))


def test_legacy_adapter_reconstructs_gate_pose_previous_action_and_phase():
    values = _environment_observation()
    adapter = LegacyN294ObservationAdapter(1)
    observation = adapter.observe(
        values,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        step=0,
    )
    assert observation.shape == (1, 32)
    assert observation[0, 10] == 1.0
    np.testing.assert_allclose(
        observation[0, 11:14],
        np.tanh(np.asarray([10.78, 0.084, -0.56]) * np.asarray([0.1, 0.2, 0.2])),
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        observation[0, 19:23], np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    )
    assert observation[0, 23] == 0.0
    np.testing.assert_array_equal(observation[0, 24:30], [1, 0, 0, 0, 0, 0])


def test_legacy_adapter_holds_pose_between_camera_samples_and_blanks_new_gate():
    adapter = LegacyN294ObservationAdapter(1)
    first = _environment_observation(gate=(10.0, 0.0, 0.0))
    initial = adapter.observe(first, np.asarray([0.0]), np.asarray([0.0]), step=0)
    moved = _environment_observation(gate=(9.0, 0.0, 0.0))
    held = adapter.observe(moved, np.asarray([0.0]), np.asarray([0.0]), step=1)
    assert held[0, 11] == initial[0, 11]

    next_gate = _environment_observation(
        gate=(14.0, 8.0, 1.0), phase=1.0 / 6.0
    )
    blank = adapter.observe(
        next_gate,
        np.asarray([1.0 / 6.0]),
        np.asarray([0.0]),
        step=5,
    )
    assert blank[0, 10] == 0.0
    sampled = adapter.observe(
        next_gate,
        np.asarray([1.0 / 6.0]),
        np.asarray([1.0 / 6.0]),
        step=8,
    )
    assert sampled[0, 10] == 1.0
    assert sampled[0, 23] == np.float32(1.0 / 6.0)
    assert sampled[0, 25] == 1.0
