from __future__ import annotations

import os

from pufferlib import pufferl
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg023_variable_gate_recurrent_policy as vg023
import scripts.eval_vq2_vg026_variable_gate_recurrent_policy as vg026
import scripts.eval_vq2_vg031_variable_gate_recurrent_policy as vg031
import scripts.eval_vq2_vg036_variable_gate_recurrent_policy as vg036
import scripts.eval_vq2_vg037_variable_gate_recurrent_policy as vg037


def test_vg037_candidate_and_vg036_abort_evidence_are_exact() -> None:
    vg037.verify_candidate()


def test_vg037_uses_fresh_fixed_course_seeds() -> None:
    assert tuple(vg037.SEEDS) == evaluator.COUNTS
    assert len(set(vg037.SEEDS.values())) == len(evaluator.COUNTS)
    for old in (
        evaluator.SEEDS,
        vg023.SEEDS,
        vg026.SEEDS,
        vg031.SEEDS,
        vg036.SEEDS,
    ):
        assert set(vg037.SEEDS.values()).isdisjoint(old.values())
    assert all(seed > 429142 for seed in vg037.SEEDS.values())


def test_vg037_retains_four_thread_terminal_episode_contract() -> None:
    old_seeds = evaluator.SEEDS
    try:
        evaluator.SEEDS = dict(vg037.SEEDS)
        config, overrides = vg037.fixed_thread_config(pufferl, num_gates=5)
    finally:
        evaluator.SEEDS = old_seeds
    assert config["vec"]["total_agents"] == evaluator.AGENTS
    assert config["vec"]["num_threads"] == 4
    assert config["vec"]["num_buffers"] == 2
    assert config["env"]["evaluation_episode_limit"] == 1
    assert overrides[-2:] == ["--vec.num-threads", "4"]


def test_vg037_vast_runner_bounds_global_threads_and_stays_offline() -> None:
    text = vg037.RUNNER.read_text()
    assert os.access(vg037.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg037.TAG in text
    assert "git clang ccache nvcc nvidia-smi python" in text
    assert "clang -fopenmp -x c - -fsyntax-only" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "assert _C.precision_bytes == 4" in text
    assert "tests/test_eval_vq2_vg037_variable_gate_recurrent_policy.py" in text
    assert text.count("OMP_NUM_THREADS=4 MKL_NUM_THREADS=1") >= 3
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
