from __future__ import annotations

import math

import pytest

import scripts.check_vq2_vg026_screen_thread_parity as parity


def test_vg026_thread_parity_contract_is_bounded_and_state_free() -> None:
    assert parity.THREADS == (4, 32)
    assert parity.COUNT == 5
    assert parity.STEPS == 128
    assert parity.MAXIMUM_NUMERIC_ERROR == 1e-7
    source = parity.Path(parity.__file__).read_text()
    assert '"optimizer_steps": 0' in source
    assert '"state_writes": 0' in source
    assert '"flight_sim_packets_sent": 0' in source


def test_vg026_recursive_numeric_error_is_exact_and_fail_closed() -> None:
    left = {"a": [1.0, 2.0], "b": {"c": 3}}
    assert parity._numeric_max_error(left, left) == 0.0
    assert parity._numeric_max_error(
        left, {"a": [1.0, 2.1], "b": {"c": 3}}
    ) == pytest.approx(0.1)
    assert math.isinf(parity._numeric_max_error(left, {"a": [1.0]}))


def test_vg026_thread_configs_change_only_thread_and_bounded_horizon() -> None:
    import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
    import scripts.eval_vq2_vg026_variable_gate_recurrent_policy as vg026
    from pufferlib import pufferl

    old_seeds = evaluator.SEEDS
    try:
        evaluator.SEEDS = dict(vg026.SEEDS)
        four, _ = parity._thread_config(4)(pufferl, num_gates=5)
        thirty_two, _ = parity._thread_config(32)(pufferl, num_gates=5)
    finally:
        evaluator.SEEDS = old_seeds
    assert four["vec"]["num_threads"] == 4
    assert thirty_two["vec"]["num_threads"] == 32
    assert four["env"]["max_steps"] == thirty_two["env"]["max_steps"] == 128
    assert four["env"]["time_limit_seconds"] == 2.0


def test_vg026_parity_admits_only_equal_thread_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: parity.Path
) -> None:
    import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
    from pufferlib import pufferl

    monkeypatch.setattr(parity.vg026, "verify_candidate", lambda: None)
    monkeypatch.setattr(parity, "source_identity", lambda: {"source": "fixed"})

    def fake_run_count(**_kwargs):
        config, _ = evaluator.teacher_free_config(pufferl, num_gates=5)
        assert config["vec"]["num_threads"] in parity.THREADS
        return {
            "completed": True,
            "num_gates": 5,
            "agents": 64,
            "episodes": 64,
            "seed": parity.vg026.SEEDS[5],
            "checkpoint_sha256": parity.vg026.CHECKPOINT_SHA256,
            "checkpoint_best_epoch": 2,
            "crossing_margin_admission_predicate": False,
            "vector_steps": 128,
            "teacher_action_blend": 0.0,
            "actor_input_width": 4119,
            "actor_input_privileged_values": 0,
            "status_hold_steps": 16,
            "phase_samples": 512,
            "phase_changes_off_tick": 0,
            "phase_decreases": 0,
            "phase_skips": 0,
            "raw_phase_encoding_max_error": 0.0,
            "nonfinite_action": False,
            "action_samples": 8192,
            "action_envelope_violations": 0,
            "executed_action_max_error": 0.0,
            "maximum_raw_public_index_distribution": {"0": 64},
            "maximum_held_public_index_distribution": {"0": 64},
            "action_min": [-0.1, -0.2, 0.3, 0.0],
            "action_max": [0.1, 0.2, 0.4, 0.0],
            "action_mean": [0.0, 0.0, 0.35, 0.0],
            "action_std": [0.01, 0.02, 0.03, 0.0],
            "metrics": {"env/n": 64.0, "env/crash": 0.0},
            "safety": {"flight_sim_packets_sent": 0},
        }

    monkeypatch.setattr(evaluator, "run_count", fake_run_count)
    output = tmp_path / "parity" / "report.json"
    report = parity.check(output=output, device_name="cpu")
    assert report["admitted"] is True
    assert output.is_file()
    assert all(value == 0.0 for value in report["numeric_max_abs_error"].values())
