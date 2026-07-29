from __future__ import annotations

import numpy as np
import pytest

import scripts.collect_vq2_variable_gate_dagger as collector
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP
from scripts.collect_vq2_variable_gate_dagger import (
    AGENTS,
    COLLECTION_STEP_LIMIT,
    EPISODES,
    MAXIMUM_CRASH_RATE,
    MINIMUM_GATE1_RATE,
    MINIMUM_GATE2_RATE,
    MINIMUM_RECORDS,
    dagger_collection_passes,
    dagger_config,
    verify_inputs,
)


def _passing_metrics() -> dict[str, float]:
    metrics = {
        "env/n": float(EPISODES),
        "env/crash": MAXIMUM_CRASH_RATE,
        "env/out_of_order": 0.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        "env/ordered_gate0_sampled": MINIMUM_GATE1_RATE,
        "env/ordered_gate1_sampled": MINIMUM_GATE2_RATE,
    }
    for count in range(1, 17):
        metrics[f"env/gate_count{count}_episode"] = (
            0.125 if 5 <= count <= 12 else 0.0
        )
    return metrics


def _passing_arguments() -> dict[str, object]:
    lengths = np.full(AGENTS, MINIMUM_RECORDS // AGENTS + 1, dtype=np.int32)
    phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
    phase_records[0] = 50_000
    phase_records[1] = 50_000
    return {
        "lengths": lengths,
        "terminal_count": np.ones(AGENTS, dtype=np.int32),
        "terminal_is_last": np.ones(AGENTS, dtype=bool),
        "labels": int(lengths.sum()),
        "executed_action_max_error": 0.0,
        "phase_changes_off_tick": 0,
        "phase_decreases": 0,
        "phase_skips": 0,
        "raw_phase_encoding_max_error": 0.0,
        "phase_records": phase_records,
    }


def test_vg007_admission_preserves_failure_states_but_requires_safe_transport() -> None:
    metrics = _passing_metrics()
    arguments = _passing_arguments()
    assert dagger_collection_passes(metrics, **arguments)

    metrics = _passing_metrics()
    metrics["env/crash"] = MAXIMUM_CRASH_RATE + 1.0 / EPISODES
    assert not dagger_collection_passes(metrics, **arguments)

    metrics = _passing_metrics()
    metrics["env/ordered_gate1_sampled"] = MINIMUM_GATE2_RATE - 1e-6
    assert not dagger_collection_passes(metrics, **arguments)

    metrics = _passing_metrics()
    changed = dict(arguments)
    phase_records = np.asarray(changed["phase_records"]).copy()
    phase_records[1] = 0
    changed["phase_records"] = phase_records
    assert not dagger_collection_passes(metrics, **changed)

    changed = dict(arguments)
    changed["executed_action_max_error"] = 1e-6
    assert not dagger_collection_passes(metrics, **changed)


def test_vg007_config_is_exact_uniform_teacher_free_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(*args: object, **kwargs: object):
        assert kwargs == {
            "agents": AGENTS,
            "episodes": EPISODES,
            "seed": collector.SEED,
            "num_gates": 12,
        }
        return {"env": {}}, ["locked"]

    monkeypatch.setattr(collector, "load_variable_config", fake_load)
    config, overrides = dagger_config(object())
    environment = config["env"]
    assert overrides == ["locked"]
    assert environment["num_gates"] == 6
    assert environment["num_gates_per_env_randomize"] == 1
    assert environment["num_gates_per_env_min"] == 5
    assert environment["num_gates_per_env_max"] == 12
    assert environment["max_steps"] == COLLECTION_STEP_LIMIT
    assert environment["time_limit_seconds"] == COLLECTION_STEP_LIMIT / 64.0
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["observable_gate_index_denominator"] == 16.0


def test_vg007_frozen_failure_and_oracle_query_inputs_verify() -> None:
    verify_inputs()
