from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.collect_vq2_vg016_variable_gate_dagger as vg016
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


def _snapshot(module, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(module, name) for name in names}


def test_vg016_frozen_candidate_rejection_and_oracle_verify() -> None:
    vg016.verify_inputs()


def test_vg016_configuration_focuses_early_failure_states() -> None:
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
        vg016.configure_collector()
        assert collector.TAG == vg016.TAG
        assert collector.SEED == 429079
        assert collector.COLLECTION_STEP_LIMIT == 1024
        assert collector.MINIMUM_RECORDS == 350_000
        assert collector.MINIMUM_GATE1_RATE == 0.90
        assert collector.CRASH_RATE_IS_ADMISSION is False
        assert collector.GATE2_REACH_IS_ADMISSION is False
        assert evaluator.CHECKPOINT_SHA256 == vg016.CHECKPOINT_SHA256
        source_paths = collector.collection_source_paths()
        for required in (
            Path(vg016.__file__).resolve(),
            vg016.PREREGISTRATION.resolve(),
            vg016.RUNNER.resolve(),
            vg016.CHECKPOINT.resolve(),
            vg016.VG015_REPORT.resolve(),
            vg016.VG015_REJECTION.resolve(),
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
            "env/ordered_gate0_sampled": 0.90,
            "env/ordered_gate1_sampled": 0.0,
        }
        for count in range(1, 17):
            metrics[f"env/gate_count{count}_episode"] = (
                0.125 if 5 <= count <= 12 else 0.0
            )
        lengths = np.full(512, 800, dtype=np.int32)
        phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
        phase_records[:2] = (120_000, 289_600)
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
        changed["labels"] = 349_999
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


def test_vg016_runner_is_source_locked_resumable_and_flightsim_free() -> None:
    text = vg016.RUNNER.read_text()
    assert os.access(vg016.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg016.TAG in text
    assert "tests/test_collect_vq2_vg016_variable_gate_dagger.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
