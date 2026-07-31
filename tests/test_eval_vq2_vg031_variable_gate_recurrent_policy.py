from __future__ import annotations

import os

import pytest

from pufferlib import pufferl
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg023_variable_gate_recurrent_policy as vg023
import scripts.eval_vq2_vg026_variable_gate_recurrent_policy as vg026
import scripts.eval_vq2_vg031_variable_gate_recurrent_policy as vg031


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


def test_vg031_candidate_and_staged_admission_evidence_are_exact() -> None:
    vg031.verify_candidate()


def test_vg031_uses_fresh_fixed_course_seeds() -> None:
    assert tuple(vg031.SEEDS) == evaluator.COUNTS
    assert len(set(vg031.SEEDS.values())) == len(evaluator.COUNTS)
    for old in (evaluator.SEEDS, vg023.SEEDS, vg026.SEEDS):
        assert set(vg031.SEEDS.values()).isdisjoint(old.values())
    assert all(seed > 429122 for seed in vg031.SEEDS.values())


def test_vg031_retains_four_thread_terminal_episode_contract() -> None:
    old_seeds = evaluator.SEEDS
    try:
        evaluator.SEEDS = dict(vg031.SEEDS)
        config, overrides = vg031.fixed_thread_config(pufferl, num_gates=5)
    finally:
        evaluator.SEEDS = old_seeds
    assert config["vec"]["total_agents"] == evaluator.AGENTS
    assert config["vec"]["num_threads"] == 4
    assert config["vec"]["num_buffers"] == 2
    assert config["env"]["evaluation_episode_limit"] == 1
    assert overrides[-2:] == ["--vec.num-threads", "4"]


def test_vg031_keeps_crossing_margin_diagnostic_but_crash_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "REQUIRE_ZERO_CROSSING_MARGIN", False)
    report = _safe_count_report(crossing_margin=1.0)
    assert evaluator.count_safety_passes(report)
    report["metrics"]["env/crash"] = 1.0 / evaluator.EPISODES_PER_COUNT
    assert not evaluator.count_safety_passes(report)


def test_vg031_vast_runner_is_source_locked_resumable_and_offline() -> None:
    text = vg031.RUNNER.read_text()
    assert os.access(vg031.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg031.TAG in text
    assert "git clang ccache nvcc nvidia-smi python" in text
    assert "clang -fopenmp -x c - -fsyntax-only" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "assert _C.precision_bytes == 4" in text
    assert "tests/test_eval_vq2_vg031_variable_gate_recurrent_policy.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
