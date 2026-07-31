from __future__ import annotations

import copy
import json

import pytest

import scripts.summarize_vq2_variable_gate_screen as summary


VG026_SCREEN = (
    summary.ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg026_variable_gate_recurrent_teacher_free_256"
)
VG025_ADMISSION = (
    summary.ROOT / "docs/vq2_vg025_six_source_refit_admission_2026-07-30.json"
)


def test_historical_vg026_summary_matches_frozen_rejection() -> None:
    evidence = summary.build_evidence(
        screen=VG026_SCREEN,
        candidate_admission=VG025_ADMISSION,
    )
    assert evidence["completed"]
    assert not evidence["admitted"]
    assert evidence["unchanged_retry_forbidden"]
    assert evidence["episodes"] == 256
    assert evidence["successes"] == 0
    assert evidence["crashes"] == 168
    assert evidence["crashes_low"] == 72
    assert evidence["crashes_xy"] == 96
    assert evidence["crashes_high"] == 0
    assert evidence["misses"] == 88
    assert evidence["timeouts"] == 0
    assert evidence["gate_reach"]["1"] == 256
    assert evidence["gate_reach"]["2"] == 97
    assert evidence["gate_reach"]["3"] == 2
    assert evidence["gate_reach"]["4"] == 0
    assert evidence["mean_gates_passed"] == pytest.approx(1.38671875)
    assert evidence["hard_transport_pass"]
    assert evidence["transport"]["executed_action_max_error"] == 0.0


def test_count_summary_rejects_invalid_phase_mass() -> None:
    report = json.loads((VG026_SCREEN / "count_5.json").read_text())
    broken = copy.deepcopy(report)
    broken["maximum_held_public_index_distribution"]["1"] -= 1
    with pytest.raises(RuntimeError, match="distribution"):
        summary.summarize_count(broken)


def test_count_summary_rejects_noninteger_terminal_rate() -> None:
    report = json.loads((VG026_SCREEN / "count_5.json").read_text())
    report["metrics"]["env/crash"] += 0.001
    with pytest.raises(RuntimeError, match="integer rate"):
        summary.summarize_count(report)


def test_screen_summary_rejects_unbound_candidate_admission(tmp_path) -> None:
    admission = json.loads(VG025_ADMISSION.read_text())
    admission["artifact_sha256"]["checkpoint"] = "0" * 64
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission))
    with pytest.raises(RuntimeError, match="does not bind"):
        summary.build_evidence(screen=VG026_SCREEN, candidate_admission=path)


def test_terminal_prefix_freezes_only_mathematically_impossible_screen(
    tmp_path,
) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    count = json.loads((VG026_SCREEN / "count_5.json").read_text())
    (screen / "count_5.json").write_text(json.dumps(count))
    state = json.loads((VG026_SCREEN / "state.json").read_text())
    state.update({"status": "screening", "completed_counts": [5]})
    state.pop("report_sha256", None)
    state.pop("success_rate", None)
    (screen / "state.json").write_text(json.dumps(state))
    evidence = summary.build_terminal_prefix_evidence(
        screen=screen,
        candidate_admission=VG025_ADMISSION,
    )
    assert evidence["completed"]
    assert not evidence["screen_completed"]
    assert not evidence["admitted"]
    assert evidence["terminal_admission_impossible"]
    assert evidence["episodes"] == 64
    assert evidence["planned_episodes"] == 256
    assert evidence["minimum_successes"] == 231
    assert evidence["maximum_possible_successes"] == 192
    assert set(evidence["terminal_reasons"]) == {
        "zero_crash_admission_is_impossible",
        "minimum_finish_count_is_impossible",
    }
    assert evidence["gate_reach"]["3"] == 2
    assert evidence["hard_transport_pass"]

    passing = copy.deepcopy(count)
    for name in ("env/crash", "env/crash_low", "env/crash_xy", "env/crash_high"):
        passing["metrics"][name] = 0.0
    passing["metrics"]["env/success_rate"] = 1.0
    passing["metrics"]["env/missed_gate"] = 0.0
    (screen / "count_5.json").write_text(json.dumps(passing))
    with pytest.raises(RuntimeError, match="not a terminal"):
        summary.build_terminal_prefix_evidence(
            screen=screen,
            candidate_admission=VG025_ADMISSION,
        )
