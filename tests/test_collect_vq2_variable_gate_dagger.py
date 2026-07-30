from __future__ import annotations

import json
import os
from pathlib import Path

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
    crossing_margin_diagnostic,
    dagger_collection_predicates,
    dagger_collection_passes,
    dagger_config,
    persist_rejection_evidence,
    verify_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_vq2_vg009_vast.sh"


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


def test_vg009_admission_preserves_failure_states_but_requires_safe_transport() -> None:
    metrics = _passing_metrics()
    metrics["env/crossing_margin_violation"] = 0.4375
    arguments = _passing_arguments()
    assert dagger_collection_passes(metrics, **arguments)
    predicates = dagger_collection_predicates(metrics, **arguments)
    assert all(type(value) is bool for value in predicates.values())
    assert "crossing_margin_violation" not in predicates
    assert crossing_margin_diagnostic(metrics) == {
        "crossing_margin_violation": 0.4375,
        "admission_predicate": False,
        "diagnostic_radius_m": 0.50,
        "official_aperture_radius_m": 0.75,
    }

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

    metrics = _passing_metrics()
    metrics["env/action_envelope_violation"] = 1.0 / EPISODES
    assert not dagger_collection_passes(metrics, **arguments)


def test_vg009_config_is_exact_uniform_teacher_free_and_bounded(
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


def test_vg009_rejection_predicates_identify_the_record_floor() -> None:
    metrics = _passing_metrics()
    arguments = _passing_arguments()
    arguments["lengths"] = np.full(
        AGENTS, MINIMUM_RECORDS // AGENTS - 1, dtype=np.int32
    )
    arguments["labels"] = int(np.asarray(arguments["lengths"]).sum())
    predicates = dagger_collection_predicates(metrics, **arguments)
    assert predicates["minimum_record_count"] is False
    assert all(
        passed for name, passed in predicates.items() if name != "minimum_record_count"
    )


def test_vg009_rejection_persists_complete_predicate_map_before_raise_path(
    tmp_path,
) -> None:
    output_path = tmp_path / "vq2_vg009_dataset"
    state_path = tmp_path / "vq2_vg009_dataset_state.json"
    state = {"status": "collecting"}
    report = {
        "records": 123_456,
        "vector_steps": 2_048,
        "admission_predicates": {
            "minimum_record_count": True,
            "executed_action_parity": False,
        },
        "failed_admission_predicates": ["executed_action_parity"],
    }
    rejection_path = persist_rejection_evidence(
        output_path=output_path,
        state_path=state_path,
        state=state,
        rejection_report=report,
    )
    persisted_report = json.loads(rejection_path.read_text())
    persisted_state = json.loads(state_path.read_text())
    assert persisted_report["admission_predicates"] == report["admission_predicates"]
    assert persisted_state["status"] == "rejected"
    assert persisted_state["admission_predicates"] == report["admission_predicates"]
    assert persisted_state["failed_admission_predicates"] == [
        "executed_action_parity"
    ]


def test_vg009_rejection_refuses_an_inconsistent_predicate_map(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="predicate map is inconsistent"):
        persist_rejection_evidence(
            output_path=tmp_path / "vq2_vg009_dataset",
            state_path=tmp_path / "vq2_vg009_dataset_state.json",
            state={"status": "collecting"},
            rejection_report={
                "records": 1,
                "vector_steps": 1,
                "admission_predicates": {"safe": False},
                "failed_admission_predicates": [],
            },
        )


def test_vg009_frozen_failure_and_oracle_query_inputs_verify() -> None:
    verify_inputs()


def test_vg009_vast_runner_is_source_locked_resumable_and_flightsim_free() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "vq2_vg009_variable_gate_dagger_round1_corrected_512" in text
    assert "git clang ccache nvcc nvidia-smi python" in text
    assert "clang -fopenmp -x c - -fsyntax-only" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "assert _C.precision_bytes == 4" in text
    assert "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest" in text
    assert "tests/test_collect_vq2_variable_gate_dagger.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index("test_drone_race_native_regressions")
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
