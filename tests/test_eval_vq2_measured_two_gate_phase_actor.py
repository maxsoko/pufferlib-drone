from __future__ import annotations

import numpy as np

from scripts.eval_vq2_measured_two_gate_phase_actor import (
    AGENTS,
    EPISODES,
    MAX_EXECUTED_ACTION_ERROR,
    MEASURED_GATES,
    SEED,
    STEP_LIMIT,
    milestone_passes,
    teacher_free_measured_config,
)


class FakePufferl:
    @staticmethod
    def load_config(_name: str) -> dict:
        import sys

        pairs = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
        return {
            "backend_env_name": "drone_race_vision",
            "vec": {"total_agents": int(pairs["--vec.total-agents"])},
            "env": {"start_gate_index": 0},
        }


def test_sf051_is_exact_measured_and_teacher_free() -> None:
    config, overrides = teacher_free_measured_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert config["seed"] == SEED
    assert AGENTS == EPISODES == 512
    assert STEP_LIMIT == 2048
    assert environment["evaluation_episode_limit"] == 1
    assert environment["num_gates"] == 6
    assert environment["gate_radius"] == environment["gate0_radius"] == 0.75
    assert environment["reset_position_noise_xy"] == 0.0
    assert environment["reset_position_noise_z"] == 0.0
    assert environment["gate_position_domain_randomize"] == 0
    assert environment["course_geometry_scale_randomize"] == 0
    assert environment["sitl_plant_domain_randomize"] == 0
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0
    for index, gate in enumerate(MEASURED_GATES):
        assert tuple(environment[f"gate{index}_{axis}"] for axis in "xyz") == gate


def test_milestone_predicate_requires_ordered_complete_clean_batch() -> None:
    reached = np.ones(EPISODES, dtype=bool)
    first = np.full(EPISODES, 496, dtype=np.int32)
    second = np.full(EPISODES, 1472, dtype=np.int32)
    terminal = np.zeros(EPISODES, dtype=bool)
    good = dict(
        reached=reached,
        first_gate_step=first,
        second_gate_step=second,
        premature_terminal=terminal,
        phase_decreases=0,
        phase_skips=0,
        phase_changes_off_tick=0,
        nonfinite_action=False,
        action_envelope_violations=0,
        executed_action_max_error=MAX_EXECUTED_ACTION_ERROR,
    )
    assert milestone_passes(**good)
    bad = dict(good)
    bad["second_gate_step"] = second.copy()
    bad["second_gate_step"][0] = 495
    assert not milestone_passes(**bad)
    bad = dict(good)
    bad["premature_terminal"] = terminal.copy()
    bad["premature_terminal"][0] = True
    assert not milestone_passes(**bad)
