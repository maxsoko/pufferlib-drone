from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
)
from scripts.collect_vq2_oracle_bc_dataset import (
    ACTION_SIZE,
    AGENTS,
    EPISODES,
    LEGAL_TAIL_SIZE,
    MASK_SCALE,
    TimeMajorDatasetWriter,
    dataset_admission_passes,
    executed_action_labels,
    legal_storage_parts,
    verify_frozen_sources,
)


def test_legal_storage_excludes_privilege_and_bounds_mask_quantization() -> None:
    rng = np.random.default_rng(4)
    observation = np.zeros((2, ENV_OBS_SIZE), dtype=np.float32)
    observation[:, :MASK_SIZE] = rng.random((2, MASK_SIZE), dtype=np.float32)
    observation[:, MASK_SIZE:LEGAL_OBS_SIZE] = rng.standard_normal(
        (2, LEGAL_TAIL_SIZE), dtype=np.float32
    )
    observation[:, LEGAL_OBS_SIZE:] = 123456.0

    mask, tail = legal_storage_parts(observation)
    assert mask.shape == (2, MASK_SIZE)
    assert mask.dtype == np.uint8
    assert tail.shape == (2, LEGAL_TAIL_SIZE)
    assert tail.dtype == np.float32
    decoded = mask.astype(np.float32) * MASK_SCALE
    assert np.max(np.abs(decoded - observation[:, :MASK_SIZE])) <= 0.5 / 255 + 1e-7

    changed = observation.copy()
    changed[:, LEGAL_OBS_SIZE:] = -987654.0
    changed_mask, changed_tail = legal_storage_parts(changed)
    assert np.array_equal(mask, changed_mask)
    assert np.array_equal(tail, changed_tail)


def test_sf012_frozen_collector_rejects_variable_gate_sources() -> None:
    # The admitted SF012 artifact remains frozen evidence. VG001 deliberately
    # changes default-off native infrastructure, so the old collector must
    # fail closed rather than silently blessing a new extension under old
    # source hashes. The variable-gate corpus gets a new collector and tag.
    with pytest.raises(RuntimeError, match="source-lock mismatch"):
        verify_frozen_sources()


def test_legal_storage_rejects_wrong_abi_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="native width"):
        legal_storage_parts(np.zeros((1, LEGAL_OBS_SIZE), dtype=np.float32))
    invalid = np.zeros((1, ENV_OBS_SIZE), dtype=np.float32)
    invalid[0, 0] = 1.1
    with pytest.raises(RuntimeError, match="escaped"):
        legal_storage_parts(invalid)


def test_executed_label_comes_from_next_newest_history_and_audits_shift() -> None:
    current = np.zeros((2, LEGAL_OBS_SIZE), dtype=np.float32)
    current[:, ACTION_HISTORY] = np.arange(24, dtype=np.float32).reshape(2, 12)
    following = np.zeros((2, ENV_OBS_SIZE), dtype=np.float32)
    following[:, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE] = [
        [0.1, -0.2, 0.3, -0.4],
        [9.0, 9.0, 9.0, 9.0],
    ]
    following[
        :, ACTION_HISTORY.start + ACTION_SIZE : ACTION_HISTORY.stop
    ] = current[:, ACTION_HISTORY.start : ACTION_HISTORY.stop - ACTION_SIZE]

    labels, shift_error = executed_action_labels(
        current, following, np.asarray([True, False])
    )
    assert np.allclose(labels[0], [0.1, -0.2, 0.3, -0.4])
    assert shift_error == 0.0

    following[0, ACTION_HISTORY.start + ACTION_SIZE] += 0.25
    _, shift_error = executed_action_labels(
        current, following, np.asarray([True, False])
    )
    assert shift_error == pytest.approx(0.25)


def _batch(agents: int, value: int) -> dict[str, np.ndarray]:
    return {
        "mask": np.full((agents, MASK_SIZE), value, dtype=np.uint8),
        "tail": np.full((agents, LEGAL_TAIL_SIZE), value, dtype=np.float32),
        "action": np.full((agents, ACTION_SIZE), value / 10, dtype=np.float32),
        "terminal": np.zeros(agents, dtype=np.uint8),
        "valid": np.ones(agents, dtype=np.uint8),
    }


def test_time_major_writer_preserves_variable_full_histories(tmp_path: Path) -> None:
    writer = TimeMajorDatasetWriter(tmp_path / "staging", max_steps=4, agents=2)
    writer.append(**_batch(2, 1))
    second = _batch(2, 2)
    second["terminal"][0] = 1
    writer.append(**second)
    third = _batch(2, 3)
    third["mask"][0] = 0
    third["tail"][0] = 0
    third["action"][0] = 0
    third["valid"][0] = 0
    third["terminal"][1] = 1
    writer.append(**third)

    counts, last = writer.validate_episode_layout(np.asarray([2, 3]))
    assert counts.tolist() == [1, 1]
    assert last.tolist() == [True, True]
    manifest = writer.finalize(tmp_path / "dataset")

    assert manifest["mask.npy"]["shape"] == [3, 2, MASK_SIZE]
    assert manifest["tail.npy"]["shape"] == [3, 2, LEGAL_TAIL_SIZE]
    assert manifest["action.npy"]["shape"] == [3, 2, ACTION_SIZE]
    mask = np.load(tmp_path / "dataset/mask.npy", mmap_mode="r")
    valid = np.load(tmp_path / "dataset/valid.npy", mmap_mode="r")
    terminal = np.load(tmp_path / "dataset/terminal.npy", mmap_mode="r")
    assert mask.shape == (3, 2, MASK_SIZE)
    assert valid[:, 0].tolist() == [1, 1, 0]
    assert terminal[:, 0].tolist() == [0, 1, 0]
    assert terminal[:, 1].tolist() == [0, 0, 1]


def _passing_metrics() -> dict[str, float]:
    metrics = {
        "env/n": float(EPISODES),
        "env/success_rate": 1.0,
        "env/gates_passed": 6.0,
        "env/crash": 0.0,
        "env/timeout": 0.0,
        "env/missed_gate": 0.0,
        "env/out_of_order": 0.0,
        "env/valid_run_rate": 1.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }
    for gate in range(6):
        metrics[f"env/ordered_gate{gate}_sampled"] = 1.0
        metrics[f"env/ordered_gate{gate}_radial"] = 0.01
    return metrics


def test_dataset_admission_requires_full_oracle_and_exact_episode_layout() -> None:
    lengths = np.full(AGENTS, 10, dtype=np.int32)
    counts = np.ones(AGENTS, dtype=np.int32)
    last = np.ones(AGENTS, dtype=bool)
    assert dataset_admission_passes(
        _passing_metrics(),
        lengths=lengths,
        terminal_count=counts,
        terminal_is_last=last,
        label_count=int(lengths.sum()),
        history_shift_max_error=0.0,
    )

    failed = _passing_metrics()
    failed["env/ordered_gate4_radial"] = 0.10001
    assert not dataset_admission_passes(
        failed,
        lengths=lengths,
        terminal_count=counts,
        terminal_is_last=last,
        label_count=int(lengths.sum()),
        history_shift_max_error=0.0,
    )
    last[3] = False
    assert not dataset_admission_passes(
        _passing_metrics(),
        lengths=lengths,
        terminal_count=counts,
        terminal_is_last=last,
        label_count=int(lengths.sum()),
        history_shift_max_error=0.0,
    )
