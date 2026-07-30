from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.collect_vq2_vg024_variable_gate_dagger as vg024
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


def _snapshot(module, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(module, name) for name in names}


def test_vg024_frozen_candidate_rejection_and_oracle_verify() -> None:
    vg024.verify_inputs()


def test_vg024_configuration_collects_new_lateral_recovery_states() -> None:
    collector_names = (
        "TAG",
        "SCHEMA",
        "STATE_SCHEMA",
        "REPORT_SCHEMA",
        "REJECTION_SCHEMA",
        "COLLECTION_LABEL",
        "AGENTS",
        "EPISODES",
        "SEED",
        "COLLECTION_STEP_LIMIT",
        "MINIMUM_RECORDS",
        "MINIMUM_GATE1_RATE",
        "MINIMUM_GATE2_RATE",
        "MAXIMUM_CRASH_RATE",
        "CRASH_RATE_IS_ADMISSION",
        "GATE2_REACH_IS_ADMISSION",
        "PREREGISTRATION",
        "RUNNER",
        "EVIDENCE_PATHS",
        "EXTRA_SOURCE_PATHS",
        "EVIDENCE_SHA256",
        "QUERY_ACTION_SOURCE",
        "PLANT_ACTION_SOURCE",
        "verify_inputs",
    )
    evaluator_names = (
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
    )
    old_collector = _snapshot(collector, collector_names)
    old_evaluator = _snapshot(evaluator, evaluator_names)
    try:
        vg024.configure_collector()
        assert collector.TAG == vg024.TAG
        assert collector.SEED == 429109
        assert collector.COLLECTION_STEP_LIMIT == 1024
        assert collector.MINIMUM_RECORDS == 300_000
        assert collector.MINIMUM_GATE1_RATE == 0.95
        assert collector.CRASH_RATE_IS_ADMISSION is False
        assert collector.GATE2_REACH_IS_ADMISSION is False
        assert evaluator.CHECKPOINT_SHA256 == vg024.CHECKPOINT_SHA256
        source_paths = collector.collection_source_paths()
        for required in (
            Path(vg024.__file__).resolve(),
            vg024.PREREGISTRATION.resolve(),
            vg024.RUNNER.resolve(),
            vg024.CHECKPOINT.resolve(),
            vg024.VG023_REPORT.resolve(),
            vg024.VG023_REJECTION.resolve(),
        ):
            assert required in source_paths

        metrics = {
            "env/n": 512.0,
            "env/crash": 1.0,
            "env/missed_gate": 0.0,
            "env/timeout": 0.0,
            "env/out_of_order": 0.0,
            "env/crossing_margin_violation": 1.0,
            "env/action_envelope_violation": 0.0,
            "env/wire_rate_envelope_violation": 0.0,
            "env/thrust_envelope_violation": 0.0,
            "env/ordered_gate0_sampled": 0.95,
            "env/ordered_gate1_sampled": 0.10,
        }
        for count in range(1, 17):
            metrics[f"env/gate_count{count}_episode"] = (
                0.125 if 5 <= count <= 12 else 0.0
            )
        lengths = np.full(512, 700, dtype=np.int32)
        phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
        phase_records[:3] = (120_000, 230_000, int(lengths.sum()) - 350_000)
        arguments = {
            "lengths": lengths,
            "terminal_count": np.ones(512, dtype=np.int32),
            "terminal_is_last": np.ones(512, dtype=bool),
            "labels": int(lengths.sum()),
            "executed_action_max_error": 0.0,
            "phase_changes_off_tick": 0,
            "phase_decreases": 0,
            "phase_skips": 0,
            "raw_phase_encoding_max_error": 0.0,
            "phase_records": phase_records,
        }
        predicates = collector.dagger_collection_predicates(metrics, **arguments)
        assert all(predicates.values())
        assert "crash_rate" not in predicates
        assert "gate_2_reach_rate" not in predicates
        diagnostic = collector.failure_distribution_diagnostic(metrics)
        assert diagnostic["crash_rate"] == 1.0
        assert diagnostic["crash_rate_admission_predicate"] is False
        assert diagnostic["gate_2_reach_admission_predicate"] is False

        changed = dict(arguments)
        changed["labels"] = 299_999
        assert not collector.dagger_collection_passes(metrics, **changed)
        changed = dict(arguments)
        changed["executed_action_max_error"] = 1e-6
        assert not collector.dagger_collection_passes(metrics, **changed)
        metrics["env/out_of_order"] = 1.0 / 512.0
        assert not collector.dagger_collection_passes(metrics, **arguments)
    finally:
        for name, value in old_collector.items():
            setattr(collector, name, value)
        for name, value in old_evaluator.items():
            setattr(evaluator, name, value)


def test_vg024_runner_is_source_locked_resumable_and_flightsim_free() -> None:
    text = vg024.RUNNER.read_text()
    assert os.access(vg024.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg024.TAG in text
    assert "tests/test_collect_vq2_vg024_variable_gate_dagger.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
