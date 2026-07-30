from __future__ import annotations

import os

import pytest

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg011_variable_gate_recurrent_policy as vg011
import scripts.eval_vq2_vg015_variable_gate_recurrent_policy as vg015
import scripts.eval_vq2_vg018_variable_gate_recurrent_policy as vg018


def _safe_count_report(*, crossing_margin: float = 0.0) -> dict[str, object]:
    return {
        "completed": True,
        "teacher_action_blend": 0.0,
        "nonfinite_action": False,
        "action_envelope_violations": 0,
        "executed_action_max_error": 0.0,
        "phase_changes_off_tick": 0,
        "phase_decreases": 0,
        "phase_skips": 0,
        "raw_phase_encoding_max_error": 0.0,
        "metrics": {
            "env/crash": 0.0,
            "env/out_of_order": 0.0,
            "env/crossing_margin_violation": crossing_margin,
            "env/action_envelope_violation": 0.0,
            "env/wire_rate_envelope_violation": 0.0,
            "env/thrust_envelope_violation": 0.0,
        },
    }


def test_vg018_candidate_and_admission_evidence_are_exact() -> None:
    vg018.verify_candidate()


def test_vg018_uses_fresh_fixed_course_seeds() -> None:
    assert tuple(vg018.SEEDS) == evaluator.COUNTS
    assert len(set(vg018.SEEDS.values())) == len(evaluator.COUNTS)
    assert set(vg018.SEEDS.values()).isdisjoint(evaluator.SEEDS.values())
    assert set(vg018.SEEDS.values()).isdisjoint(vg011.SEEDS.values())
    assert set(vg018.SEEDS.values()).isdisjoint(vg015.SEEDS.values())
    assert all(seed > 429080 for seed in vg018.SEEDS.values())


def test_vg018_keeps_crossing_margin_diagnostic_but_crash_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "REQUIRE_ZERO_CROSSING_MARGIN", False)
    report = _safe_count_report(crossing_margin=1.0)
    assert evaluator.count_safety_passes(report)
    report["metrics"]["env/crash"] = 1.0 / evaluator.EPISODES_PER_COUNT
    assert not evaluator.count_safety_passes(report)


def test_base_contract_still_requires_zero_crossing_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "REQUIRE_ZERO_CROSSING_MARGIN", True)
    assert not evaluator.count_safety_passes(
        _safe_count_report(crossing_margin=1.0)
    )


def test_vg018_vast_runner_is_source_locked_resumable_and_offline() -> None:
    text = vg018.RUNNER.read_text()
    assert os.access(vg018.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg018.TAG in text
    assert "git clang ccache nvcc nvidia-smi python" in text
    assert "clang -fopenmp -x c - -fsyntax-only" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "assert _C.precision_bytes == 4" in text
    assert "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest" in text
    assert "tests/test_eval_vq2_vg018_variable_gate_recurrent_policy.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index("test_drone_race_native_regressions")
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
