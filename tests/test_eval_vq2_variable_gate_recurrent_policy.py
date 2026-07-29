from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from scripts.eval_vq2_variable_gate_recurrent_policy import (
    AGENTS,
    COUNTS,
    EPISODES_PER_COUNT,
    MINIMUM_SUCCESS_RATE,
    SEEDS,
    TOTAL_EPISODES,
    aggregate_admission_passes,
    completed_count_screen,
    count_safety_passes,
)


def _count_report(count: int, *, successes: int = 64) -> dict[str, object]:
    success_rate = successes / EPISODES_PER_COUNT
    metrics = {
        "env/n": float(EPISODES_PER_COUNT),
        "env/success_rate": success_rate,
        "env/gates_passed": count * success_rate,
        "env/crash": 0.0,
        "env/timeout": 1.0 - success_rate,
        "env/missed_gate": 0.0,
        "env/out_of_order": 0.0,
        "env/valid_run_rate": success_rate,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        f"env/gate_count{count}_episode": 1.0,
    }
    return {
        "schema": evaluator.SCHEMA,
        "completed": True,
        "num_gates": count,
        "teacher_action_blend": 0.0,
        "nonfinite_action": False,
        "action_envelope_violations": 0,
        "executed_action_max_error": 0.0,
        "phase_changes_off_tick": 0,
        "phase_decreases": 0,
        "phase_skips": 0,
        "raw_phase_encoding_max_error": 0.0,
        "metrics": metrics,
    }


def test_vg006_screen_contract_is_256_new_courses_and_90_percent() -> None:
    assert COUNTS == (5, 8, 11, 12)
    assert AGENTS == EPISODES_PER_COUNT == 64
    assert TOTAL_EPISODES == 256
    assert MINIMUM_SUCCESS_RATE == 0.90
    assert len(set(SEEDS.values())) == len(COUNTS)
    assert all(seed != 429030 for seed in SEEDS.values())


def test_completed_count_screen_requires_all_terminal_metrics() -> None:
    report = _count_report(8)
    assert completed_count_screen(report["metrics"], num_gates=8)
    report["metrics"]["env/n"] = 63.0
    assert not completed_count_screen(report["metrics"], num_gates=8)


def test_aggregate_admits_90_percent_only_with_zero_crash_and_transport() -> None:
    reports = [
        _count_report(5, successes=58),
        _count_report(8, successes=58),
        _count_report(11, successes=58),
        _count_report(12, successes=58),
    ]
    assert sum(r["metrics"]["env/success_rate"] * 64 for r in reports) == 232
    assert all(count_safety_passes(report) for report in reports)
    assert aggregate_admission_passes(reports)

    reports[2]["metrics"]["env/crash"] = 1.0 / 64.0
    assert not count_safety_passes(reports[2])
    assert not aggregate_admission_passes(reports)


def test_aggregate_rejects_below_90_percent() -> None:
    reports = [_count_report(count, successes=57) for count in COUNTS]
    assert not aggregate_admission_passes(reports)


def test_admission_resumes_completed_count_without_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "screen"
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("locked\n")
    identity = {
        "source_commit": "a" * 40,
        "source_sha256": {"source": "b" * 64},
        "runtime": {"python": "test"},
        "compiled_extension_path": "/tmp/_C.so",
    }
    calls: list[int] = []

    def fake_run_count(*, num_gates: int, **_: object) -> dict[str, object]:
        calls.append(num_gates)
        report = _count_report(num_gates)
        report["source_identity"] = identity
        return report

    monkeypatch.setattr(evaluator, "PREREGISTRATION", preregistration)
    monkeypatch.setattr(evaluator, "CHECKPOINT_SHA256", "c" * 64)
    monkeypatch.setattr(evaluator, "source_identity", lambda: identity)
    monkeypatch.setattr(evaluator, "run_count", fake_run_count)
    original_write = evaluator.write_json_once
    writes = 0

    def interrupt_after_first(path: Path, payload: dict[str, object]) -> None:
        nonlocal writes
        original_write(path, payload)
        if path.name.startswith("count_"):
            writes += 1
            if writes == 1:
                raise KeyboardInterrupt

    monkeypatch.setattr(evaluator, "write_json_once", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        evaluator.run_admission(output=output, device_name="cpu")
    first_bytes = (output / "count_5.json").read_bytes()

    monkeypatch.setattr(evaluator, "write_json_once", original_write)
    report = evaluator.run_admission(output=output, device_name="cpu", resume=True)
    assert report["admitted"]
    assert calls == [5, 8, 11, 12]
    assert (output / "count_5.json").read_bytes() == first_bytes
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "admitted"
