from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.collect_vq2_staged_variable_gate_dagger as staged
import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


@pytest.fixture(autouse=True)
def restore_configured_modules():
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
        "GOAL_PROMPT",
        "GOAL_PROMPT_SHA256",
        "EVIDENCE_PATHS",
        "EXTRA_SOURCE_PATHS",
        "EVIDENCE_SHA256",
        "QUERY_ACTION_SOURCE",
        "PLANT_ACTION_SOURCE",
        "dagger_collection_predicates",
        "verify_inputs",
    )
    evaluator_names = (
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
    )
    old_collector = {name: getattr(collector, name) for name in collector_names}
    old_evaluator = {name: getattr(evaluator, name) for name in evaluator_names}
    yield
    for name, value in old_collector.items():
        setattr(collector, name, value)
    for name, value in old_evaluator.items():
        setattr(evaluator, name, value)


def _manifest() -> dict[str, object]:
    return {
        "schema": staged.MANIFEST_SCHEMA,
        "tag": "vq2_test_higher_phase",
        "collection_label": "TEST",
        "state_schema": "test_state_v1",
        "report_schema": "test_report_v1",
        "rejection_schema": "test_rejection_v1",
        "seed": 429131,
        "agents": 512,
        "episodes": 512,
        "collection_step_limit": 4096,
        "minimum_records": 1_000_000,
        "minimum_gate1_rate": 0.95,
        "minimum_gate2_rate": 0.50,
        "minimum_gate3_rate": 0.01,
        "minimum_phase2_records": 1,
        "minimum_phase3_records": 1,
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "a" * 64,
        "train_report": "train.json",
        "train_report_sha256": "b" * 64,
        "train_report_schema": "train_v1",
        "train_report_tag": "train",
        "candidate_admission": "admission.json",
        "candidate_admission_sha256": "c" * 64,
        "candidate_admission_schema": "admission_v1",
        "screen_evidence": "screen.json",
        "screen_evidence_sha256": "d" * 64,
        "screen_evidence_schema": "screen_v1",
        "screen_tag": "screen",
        "oracle_report": "oracle.json",
        "oracle_report_sha256": "e" * 64,
        "preregistration": "prereg.md",
        "runner": "runner.sh",
        "goal_prompt": "goal.md",
        "goal_prompt_sha256": "f" * 64,
        "query_action_source": "oracle_label_only",
        "plant_action_source": "recurrent_policy_mean",
    }


def test_manifest_requires_fresh_higher_phase_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = staged.load_manifest(path)
    assert loaded["seed"] == 429131

    manifest["minimum_gate3_rate"] = 0.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="rates"):
        staged.load_manifest(path)


def test_bound_inputs_require_admitted_actor_rejected_screen_and_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(staged, "ROOT", tmp_path)
    oracle_source = tmp_path / "pufferlib/vq2_oracle.py"
    oracle_source.parent.mkdir(parents=True)
    oracle_source.write_text("oracle source\n")
    monkeypatch.setattr(
        collector, "ORACLE_QUERY_SHA256", collector.sha256_path(oracle_source)
    )

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = collector.sha256_path(checkpoint)
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps({
            "schema": "train_v1",
            "tag": "train",
            "completed": True,
            "numerically_admitted": True,
            "checkpoint_sha256": checkpoint_sha,
            "minimum_transition_window_exposure": 4.0,
            "equal_source_weight_audit": True,
        })
    )
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps({
            "schema": "admission_v1",
            "completed": True,
            "numerically_admitted": True,
            "artifact_sha256": {
                "checkpoint": checkpoint_sha,
                "report": collector.sha256_path(train),
            },
            "safety": {
                "flight_sim_packets_sent": 0,
                "submission_authorized": False,
            },
        })
    )
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps({
            "schema": "screen_v1",
            "screen_tag": "screen",
            "completed": True,
            "admitted": False,
            "unchanged_retry_forbidden": True,
            "checkpoint_sha256": checkpoint_sha,
            "episodes": 256,
            "hard_transport_pass": True,
            "gate_reach": {"3": 4},
            "safety": {
                "flight_sim_packets_sent": 0,
                "submission_authorized": False,
            },
        })
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"parity_passed": True}))
    goal = tmp_path / "goal.md"
    goal.write_text("goal\n")
    manifest = _manifest()
    manifest.update({
        "checkpoint_sha256": checkpoint_sha,
        "train_report_sha256": collector.sha256_path(train),
        "candidate_admission_sha256": collector.sha256_path(admission),
        "screen_evidence_sha256": collector.sha256_path(screen),
        "oracle_report_sha256": collector.sha256_path(oracle),
        "goal_prompt_sha256": collector.sha256_path(goal),
    })
    staged.verify_bound_inputs(manifest)

    rejected = json.loads(screen.read_text())
    rejected["hard_transport_pass"] = False
    screen.write_text(json.dumps(rejected))
    manifest["screen_evidence_sha256"] = collector.sha256_path(screen)
    with pytest.raises(RuntimeError, match="eligible frontier"):
        staged.verify_bound_inputs(manifest)


def test_higher_phase_predicates_require_gate3_and_phase3_records(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    original = collector.dagger_collection_predicates
    try:
        staged.configure_collector(path, manifest)
        phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
        phase_records[:4] = (200_000, 300_000, 300_000, 200_000)
        metrics = {
            "env/n": 512.0,
            "env/crash": 0.25,
            "env/missed_gate": 0.25,
            "env/timeout": 0.50,
            "env/out_of_order": 0.0,
            "env/crossing_margin_violation": 0.0,
            "env/action_envelope_violation": 0.0,
            "env/wire_rate_envelope_violation": 0.0,
            "env/thrust_envelope_violation": 0.0,
            "env/ordered_gate0_sampled": 1.0,
            "env/ordered_gate1_sampled": 0.75,
            "env/ordered_gate2_sampled": 0.25,
        }
        for count in range(1, 17):
            metrics[f"env/gate_count{count}_episode"] = (
                0.125 if 5 <= count <= 12 else 0.0
            )
        lengths = np.full(512, 2048, dtype=np.int32)
        kwargs = {
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
        predicates = collector.dagger_collection_predicates(metrics, **kwargs)
        assert all(predicates.values())
        assert predicates["gate_3_reach_rate"]
        assert predicates["minimum_phase_3_records"]

        metrics["env/ordered_gate2_sampled"] = 0.0
        assert not collector.dagger_collection_predicates(
            metrics, **kwargs
        )["gate_3_reach_rate"]
        metrics["env/ordered_gate2_sampled"] = 0.25
        no_phase3 = phase_records.copy()
        no_phase3[2] += no_phase3[3]
        no_phase3[3] = 0
        assert not collector.dagger_collection_predicates(
            metrics, **{**kwargs, "phase_records": no_phase3}
        )["minimum_phase_3_records"]
    finally:
        collector.dagger_collection_predicates = original


def test_source_surface_binds_manifest_and_runtime_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    original_extra = collector.EXTRA_SOURCE_PATHS
    original_evidence = collector.EVIDENCE_PATHS
    try:
        staged.configure_collector(path, manifest)
        sources = set(collector.collection_source_paths())
        assert Path(staged.__file__).resolve() in sources
        assert path.resolve() in sources
        for name in (
            "checkpoint",
            "train_report",
            "candidate_admission",
            "screen_evidence",
            "oracle_report",
        ):
            assert staged._root_path(str(manifest[name])) in sources
        assert collector.CRASH_RATE_IS_ADMISSION is False
        assert collector.GATE2_REACH_IS_ADMISSION is True
    finally:
        collector.EXTRA_SOURCE_PATHS = original_extra
        collector.EVIDENCE_PATHS = original_evidence
