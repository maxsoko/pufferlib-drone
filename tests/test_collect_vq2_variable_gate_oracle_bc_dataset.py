from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.collect_vq2_variable_gate_oracle_bc_dataset as collector
from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_public_phase import PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_variable_gate_oracle_bc_dataset import (
    ACTION_SIZE,
    AGENTS,
    EPISODES,
    EXPECTED_PHASE_INCREMENTS,
    LEGAL_TAIL_SIZE,
    MINIMUM_RECORDS,
    PHASE_PRIVILEGED_INDEX,
    PHASE_TAIL_SIZE,
    VariableGateDatasetWriter,
    phase_storage_parts,
    prepare_collection_state,
    update_held_phase,
    variable_collection_config,
    variable_dataset_admission_passes,
)


def test_phase_storage_persists_legal_values_and_one_explicit_public_scalar() -> None:
    rng = np.random.default_rng(429030)
    observation = np.zeros((2, ENV_OBS_SIZE), dtype=np.float32)
    observation[:, :MASK_SIZE] = rng.random((2, MASK_SIZE), dtype=np.float32)
    observation[:, MASK_SIZE:LEGAL_OBS_SIZE] = rng.standard_normal(
        (2, LEGAL_TAIL_SIZE), dtype=np.float32
    )
    observation[:, LEGAL_OBS_SIZE:] = 12345.0
    phase = np.asarray([1.0 / 16.0, 7.0 / 16.0], dtype=np.float32)

    mask, tail = phase_storage_parts(observation, phase)
    assert mask.shape == (2, MASK_SIZE)
    assert tail.shape == (2, PHASE_TAIL_SIZE)
    assert PHASE_LEGAL_OBS_SIZE == 4119
    assert np.array_equal(tail[:, :-1], observation[:, MASK_SIZE:LEGAL_OBS_SIZE])
    assert np.array_equal(tail[:, -1], phase)

    changed = observation.copy()
    changed[:, LEGAL_OBS_SIZE:] = -98765.0
    changed_mask, changed_tail = phase_storage_parts(changed, phase)
    assert np.array_equal(mask, changed_mask)
    assert np.array_equal(tail, changed_tail)


def test_phase_storage_rejects_misaligned_phase() -> None:
    observation = np.zeros((2, ENV_OBS_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="does not align"):
        phase_storage_parts(observation, np.zeros(1, dtype=np.float32))


def test_held_phase_samples_only_at_public_ticks_and_requires_index_over_16() -> None:
    held = np.zeros(2, dtype=np.float32)
    held, sampled, error = update_held_phase(
        np.zeros(2, dtype=np.float32), held, step=0
    )
    assert sampled
    assert error == 0.0

    raw = np.asarray([1.0 / 16.0, 2.0 / 16.0], dtype=np.float32)
    held, sampled, error = update_held_phase(raw, held, step=1)
    assert not sampled
    assert np.array_equal(held, np.zeros(2, dtype=np.float32))
    assert error == 0.0

    held, sampled, error = update_held_phase(
        raw, held, step=PUBLIC_STATUS_INTERVAL_STEPS
    )
    assert sampled
    assert np.array_equal(held, raw)
    assert error == 0.0

    with pytest.raises(RuntimeError, match="divided by 16"):
        update_held_phase(
            np.asarray([0.1, 0.2], dtype=np.float32),
            held,
            step=2 * PUBLIC_STATUS_INTERVAL_STEPS,
        )


def _writer_batch(agents: int, value: int) -> dict[str, np.ndarray]:
    return {
        "mask": np.full((agents, MASK_SIZE), value, dtype=np.uint8),
        "tail": np.full((agents, PHASE_TAIL_SIZE), value, dtype=np.float32),
        "action": np.full((agents, ACTION_SIZE), value / 10, dtype=np.float32),
        "terminal": np.zeros(agents, dtype=np.uint8),
        "valid": np.ones(agents, dtype=np.uint8),
    }


def test_variable_writer_uses_4119_input_contract(tmp_path: Path) -> None:
    writer = VariableGateDatasetWriter(
        tmp_path / "staging", max_steps=2, agents=2
    )
    first = _writer_batch(2, 1)
    writer.append(**first)
    second = _writer_batch(2, 2)
    second["terminal"][:] = 1
    writer.append(**second)
    counts, last = writer.validate_episode_layout(np.asarray([2, 2]))
    assert counts.tolist() == [1, 1]
    assert last.tolist() == [True, True]
    manifest = writer.finalize(tmp_path / "dataset")
    assert manifest["tail.npy"]["shape"] == [2, 2, PHASE_TAIL_SIZE]


def _passing_metrics() -> dict[str, float]:
    metrics = {
        "env/n": float(EPISODES),
        "env/success_rate": 1.0,
        "env/gates_passed": 8.5,
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
    for count in range(1, 17):
        expected = 0.125 if 5 <= count <= 12 else 0.0
        metrics[f"env/gate_count{count}_episode"] = expected
        metrics[f"env/gate_count{count}_success"] = expected
    for gate in range(12):
        metrics[f"env/ordered_gate{gate}_sampled"] = (
            1.0 if gate < 5 else (12 - gate) / 8.0
        )
        metrics[f"env/ordered_gate{gate}_radial"] = 0.05
    return metrics


def _admission_arguments() -> dict[str, object]:
    lengths = np.full(AGENTS, MINIMUM_RECORDS // AGENTS + 1, dtype=np.int32)
    return {
        "lengths": lengths,
        "terminal_count": np.ones(AGENTS, dtype=np.int32),
        "terminal_is_last": np.ones(AGENTS, dtype=bool),
        "label_count": int(lengths.sum()),
        "history_shift_max_error": 0.0,
        "phase_changes_off_tick": 0,
        "phase_decreases": 0,
        "phase_increments": EXPECTED_PHASE_INCREMENTS,
        "raw_phase_encoding_max_error": 0.0,
    }


def test_variable_dataset_admission_requires_every_sf012_invariant() -> None:
    metrics = _passing_metrics()
    arguments = _admission_arguments()
    assert variable_dataset_admission_passes(metrics, **arguments)

    failed = dict(metrics)
    failed["env/crash"] = 1.0 / EPISODES
    assert not variable_dataset_admission_passes(failed, **arguments)

    failed = dict(metrics)
    failed["env/gate_count7_episode"] = 0.0
    assert not variable_dataset_admission_passes(failed, **arguments)

    failed = dict(metrics)
    failed["env/ordered_gate11_radial"] = 0.10001
    assert not variable_dataset_admission_passes(failed, **arguments)

    for key, value in (
        ("label_count", MINIMUM_RECORDS - 1),
        ("history_shift_max_error", 1e-6),
        ("phase_changes_off_tick", 1),
        ("phase_decreases", 1),
        ("phase_increments", EXPECTED_PHASE_INCREMENTS - 1),
        ("raw_phase_encoding_max_error", 1e-5),
    ):
        changed = dict(arguments)
        changed[key] = value
        assert not variable_dataset_admission_passes(metrics, **changed)


def test_variable_collection_config_is_uniform_fixed_instance_and_360_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"max_steps": 23040, "evaluation_episode_limit": 1}

    def fake_load(*args: object, **kwargs: object):
        assert kwargs == {
            "agents": AGENTS,
            "episodes": EPISODES,
            "seed": collector.SEED,
            "num_gates": 12,
        }
        return {"env": dict(environment)}, ["locked"]

    monkeypatch.setattr(collector, "load_variable_config", fake_load)
    config, overrides = variable_collection_config(object())
    env = config["env"]
    assert overrides == ["locked"]
    assert env["num_gates"] == 6
    assert env["num_gates_per_env_randomize"] == 1
    assert env["num_gates_per_env_min"] == 5
    assert env["num_gates_per_env_max"] == 12
    assert env["num_gates_per_env_seed"] == collector.SEED
    assert env["teacher_roll_until_gate_index"] == 16
    assert env["observable_gate_index_denominator"] == 16.0
    assert env["max_steps"] == int(collector.EPISODE_SECONDS * 64)


def _identity() -> dict[str, object]:
    return {
        "schema": "test",
        "tag": collector.TAG,
        "source_sha256": {"source": "a" * 64},
        "runtime": {"python": "test"},
    }


def test_collection_resume_recovers_only_interrupted_exact_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    state_path = tmp_path / "state.json"
    staging = tmp_path / ".staging"
    identity = _identity()
    state, report = prepare_collection_state(
        output=output,
        state_path=state_path,
        staging=staging,
        state_identity=identity,
        source_sha256=identity["source_sha256"],
        resume=False,
    )
    assert report is None
    assert state["status"] == "collecting"

    output.mkdir()
    (output / "partial.npy").write_bytes(b"partial")
    staging.mkdir()
    (staging / "partial.npy").write_bytes(b"partial")
    state, report = prepare_collection_state(
        output=output,
        state_path=state_path,
        staging=staging,
        state_identity=identity,
        source_sha256=identity["source_sha256"],
        resume=True,
    )
    assert report is None
    assert state["resume_count"] == 1
    assert not output.exists()
    assert not staging.exists()


def test_collection_resume_rejects_terminal_or_mismatched_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    state_path = tmp_path / "state.json"
    staging = tmp_path / ".staging"
    identity = _identity()
    terminal = {**identity, "status": "rejected", "resume_count": 0}
    state_path.write_text(json.dumps(terminal))
    with pytest.raises(RuntimeError, match="rejection is terminal"):
        prepare_collection_state(
            output=output,
            state_path=state_path,
            staging=staging,
            state_identity=identity,
            source_sha256=identity["source_sha256"],
            resume=True,
        )

    terminal["status"] = "collecting"
    terminal["runtime"] = {"python": "changed"}
    state_path.write_text(json.dumps(terminal))
    with pytest.raises(RuntimeError, match="resume mismatch for runtime"):
        prepare_collection_state(
            output=output,
            state_path=state_path,
            staging=staging,
            state_identity=identity,
            source_sha256=identity["source_sha256"],
            resume=True,
        )


def test_phase_privileged_index_is_training_mirror_not_legal_abi() -> None:
    assert PHASE_PRIVILEGED_INDEX >= LEGAL_OBS_SIZE
    assert PHASE_PRIVILEGED_INDEX < ENV_OBS_SIZE
