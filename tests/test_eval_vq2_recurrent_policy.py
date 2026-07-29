from __future__ import annotations

import pytest
import torch

from scripts.eval_vq2_recurrent_policy import (
    AGENTS,
    EPISODES,
    SEED,
    policy_screen_passes,
    preserve_frozen_state,
    teacher_free_config,
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


def test_teacher_free_screen_is_full_course_randomized_and_zero_blend() -> None:
    config, overrides = teacher_free_config(FakePufferl)
    environment = config["env"]
    assert ["--seed", str(SEED)] == overrides[:2]
    assert environment["evaluation_episode_limit"] == 1
    assert environment["num_gates"] == 6
    assert environment["gate_radius"] == 0.75
    assert environment["gate_position_domain_randomize"] == 1
    assert environment["course_geometry_scale_randomize"] == 1
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_course_spline"] == 0
    assert environment["teacher_segment_minimum_jerk"] == 0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0
    assert AGENTS == EPISODES == 64


def test_frozen_agents_do_not_advance_recurrent_state() -> None:
    previous = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    candidate = previous + 100.0
    active = torch.tensor([True, False, True])
    result = preserve_frozen_state(previous, candidate, active)
    assert torch.equal(result[:, 0], candidate[:, 0])
    assert torch.equal(result[:, 1], previous[:, 1])
    assert torch.equal(result[:, 2], candidate[:, 2])
    with pytest.raises(ValueError, match="align"):
        preserve_frozen_state(previous, candidate, torch.ones(2, dtype=torch.bool))


def _passing_metrics() -> dict[str, float]:
    return {
        "env/n": float(EPISODES),
        "env/success_rate": 1.0,
        "env/gates_passed": 6.0,
        "env/crash": 0.0,
        "env/timeout": 0.0,
        "env/missed_gate": 0.0,
        "env/out_of_order": 0.0,
        "env/valid_run_rate": 1.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }


def test_policy_admission_requires_every_teacher_free_episode() -> None:
    passing = _passing_metrics()
    assert policy_screen_passes(passing, episodes=EPISODES)
    for key in passing:
        failed = dict(passing)
        failed[key] = 0.0 if passing[key] else 1.0
        assert not policy_screen_passes(failed, episodes=EPISODES), key
