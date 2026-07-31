from __future__ import annotations

import os

import scripts.eval_vq2_vg047_offset8_fraction_bracket as vg047


def test_vg047_fixed_offset8_fraction_contract() -> None:
    assert vg047.SEED == 429161
    assert vg047.EPISODE_OFFSET == 8
    assert vg047.AGENTS == vg047.EPISODES == 64
    assert vg047.NUM_THREADS == 4
    assert vg047.MAX_STEPS == 2560
    assert vg047.ALPHAS == (
        0.0,
        0.025,
        0.04,
        0.05,
        0.06,
        0.075,
        0.085,
        0.10,
    )


def test_vg047_inputs_bind_qualified_smaller_fraction_and_rejection() -> None:
    vg047.verify_inputs()


def test_vg047_config_applies_offset8() -> None:
    evaluator = vg047.base.evaluator
    changed_names = (
        "SCHEMA",
        "AGENTS",
        "EPISODES_PER_COUNT",
        "TOTAL_EPISODES",
        "SEEDS",
        "REQUIRE_ZERO_CROSSING_MARGIN",
        "teacher_free_config",
    )
    original = {name: getattr(evaluator, name) for name in changed_names}

    def base_config(_module, *, num_gates):
        assert num_gates == 5
        return {"vec": {}, "env": {}}, []

    evaluator.teacher_free_config = base_config
    try:
        vg047.configure_evaluator()
        config, overrides = evaluator.teacher_free_config(
            None, num_gates=5
        )
        assert config["env"]["evaluation_episode_offset"] == 8
        index = overrides.index("--env.evaluation-episode-offset")
        assert overrides[index + 1] == "8"
    finally:
        for name, value in original.items():
            setattr(evaluator, name, value)


def test_vg047_runner_is_resumable_source_locked_and_offline() -> None:
    text = vg047.RUNNER.read_text()
    assert os.access(vg047.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "eval_vq2_vg047_offset8_fraction_bracket.py" in text
    assert "tests/test_eval_vq2_vg047_offset8_fraction_bracket.py" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
