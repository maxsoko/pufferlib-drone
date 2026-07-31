from __future__ import annotations

import copy

import scripts.compare_vq2_staged_count5_diagnostic as comparator
import scripts.eval_vq2_staged_count5_component as component


def _count_report(
    *,
    phase_distribution: dict[int, int],
    crashes: int,
    successes: int = 0,
    episodes: int = 16,
) -> dict[str, object]:
    misses = episodes - crashes - successes
    return {
        "schema": "vq2_staged_count5_component_v1",
        "completed": True,
        "num_gates": 5,
        "agents": episodes,
        "episodes": episodes,
        "seed": 429121,
        "checkpoint_sha256": "a" * 64,
        "teacher_action_blend": 0.0,
        "nonfinite_action": False,
        "action_envelope_violations": 0,
        "executed_action_max_error": 0.0,
        "phase_changes_off_tick": 0,
        "phase_decreases": 0,
        "phase_skips": 0,
        "raw_phase_encoding_max_error": 0.0,
        "maximum_held_public_index_distribution": {
            str(index): phase_distribution.get(index, 0) for index in range(17)
        },
        "vector_steps": 2048,
        "wall_time_seconds": 10.0,
        "metrics": {
            "env/success_rate": successes / episodes,
            "env/crash": crashes / episodes,
            "env/missed_gate": misses / episodes,
            "env/timeout": 0.0,
            "env/gates_passed": 1.0,
            "env/out_of_order": 0.0,
            "env/action_envelope_violation": 0.0,
            "env/wire_rate_envelope_violation": 0.0,
            "env/thrust_envelope_violation": 0.0,
        },
        "safety": {
            "teacher_labels_written": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }


def test_count5_manifest_rejects_stale_or_nonpaired_contract() -> None:
    manifest = {
        "schema": component.MANIFEST_SCHEMA,
        "role": "candidate",
        "tag": "vq2_test",
        "num_gates": 5,
        "agents": 16,
        "episodes": 16,
        "num_threads": 4,
        "max_steps": 2560,
        "seed": 429120,
        "checkpoint": "checkpoint.pt",
        "train_report": "report.json",
        "admission": "admission.json",
        "preregistration": "prereg.md",
        "runner": "runner.sh",
        "goal_prompt": "goal.md",
        "checkpoint_sha256": "a" * 64,
        "train_report_sha256": "b" * 64,
        "admission_sha256": "c" * 64,
        "goal_prompt_sha256": "d" * 64,
    }
    import json
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory(dir=component.ROOT) as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        try:
            component.load_manifest(path)
        except RuntimeError as exc:
            assert "seed is not fresh" in str(exc)
        else:
            raise AssertionError("stale count-5 seed was accepted")


def test_paired_diagnostic_requires_downstream_improvement_without_regression() -> None:
    parent = comparator.summarize_count(
        _count_report(phase_distribution={1: 12, 2: 4}, crashes=10)
    )
    candidate = comparator.summarize_count(
        _count_report(phase_distribution={1: 10, 2: 5, 3: 1}, crashes=9)
    )
    assert comparator.diagnostic_passes(parent, candidate)

    gate1_regression = copy.deepcopy(candidate)
    gate1_regression["gate_reach"]["1"] = parent["gate_reach"]["1"] - 1
    assert not comparator.diagnostic_passes(parent, gate1_regression)

    crash_regression = copy.deepcopy(candidate)
    crash_regression["crashes"] = parent["crashes"] + 1
    assert not comparator.diagnostic_passes(parent, crash_regression)

    no_downstream_gain = copy.deepcopy(candidate)
    no_downstream_gain["gate_reach"]["3"] = parent["gate_reach"]["3"]
    no_downstream_gain["successes"] = parent["successes"]
    assert not comparator.diagnostic_passes(parent, no_downstream_gain)


def test_paired_diagnostic_keeps_transport_faults_hard() -> None:
    parent_report = _count_report(
        phase_distribution={1: 12, 2: 4}, crashes=10
    )
    candidate_report = _count_report(
        phase_distribution={1: 10, 2: 5, 3: 1}, crashes=9
    )
    candidate_report["phase_skips"] = 1
    parent = comparator.summarize_count(parent_report)
    candidate = comparator.summarize_count(candidate_report)
    assert parent["hard_transport_pass"]
    assert not candidate["hard_transport_pass"]
    assert not comparator.diagnostic_passes(parent, candidate)
