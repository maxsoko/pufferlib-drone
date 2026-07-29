from __future__ import annotations

from scripts.eval_vq2_recurrent_dagger_broad import (
    AGENTS,
    EPISODES,
    SEED,
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


def test_sf023_is_fresh_full_course_and_strictly_teacher_free() -> None:
    config, overrides = teacher_free_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert AGENTS == EPISODES == 512
    assert environment["evaluation_episode_limit"] == 1
    assert environment["num_gates"] == 6
    assert environment["gate_radius"] == 0.75
    assert environment["gate_position_domain_randomize"] == 1
    assert environment["course_geometry_scale_randomize"] == 1
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0
