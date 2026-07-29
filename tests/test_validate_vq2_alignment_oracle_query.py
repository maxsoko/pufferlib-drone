from __future__ import annotations

import numpy as np

from scripts.validate_vq2_alignment_oracle_query import (
    AGENTS,
    EPISODES,
    MAX_PARITY_ERROR,
    SEED,
    parity_config,
    parity_passes,
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


def test_parity_screen_uses_admitted_randomized_oracle() -> None:
    config, overrides = parity_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert environment["evaluation_episode_limit"] == 1
    assert environment["teacher_action_blend"] == 1.0
    assert environment["teacher_alignment_governor"] == 1
    assert environment["teacher_pitch_speed_target_m_s"] == 2.0
    assert environment["gate_position_domain_randomize"] == 1
    assert environment["course_geometry_scale_randomize"] == 1
    assert AGENTS == EPISODES == 64


def _metrics() -> dict[str, float]:
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


def test_parity_admission_requires_oracle_success_and_tight_error() -> None:
    assert parity_passes(
        _metrics(),
        samples=100,
        maximum_error=np.full(4, MAX_PARITY_ERROR),
    )
    assert not parity_passes(
        _metrics(),
        samples=100,
        maximum_error=np.full(4, MAX_PARITY_ERROR * 1.01),
    )
    failed = _metrics()
    failed["env/success_rate"] = 0.99
    assert not parity_passes(
        failed, samples=100, maximum_error=np.zeros(4)
    )
