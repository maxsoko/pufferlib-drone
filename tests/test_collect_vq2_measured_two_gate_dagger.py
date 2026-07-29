from __future__ import annotations

import numpy as np

from scripts.collect_vq2_measured_two_gate_dagger import (
    AGENTS,
    COLLECTION_STEP_LIMIT,
    EPISODES,
    SEED,
    exact_collection_passes,
    exact_dagger_config,
)


class FakePufferl:
    @staticmethod
    def load_config(_name: str) -> dict:
        import sys

        pairs = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
        return {
            "backend_env_name": "drone_race_vision",
            "seed": int(pairs["--seed"]),
            "vec": {"total_agents": int(pairs["--vec.total-agents"])},
            "env": {"start_gate_index": 0},
        }


def test_sf052_exact_config_is_zero_teacher() -> None:
    config, overrides = exact_dagger_config(FakePufferl)
    environment = config["env"]
    assert config["seed"] == SEED
    assert overrides[:2] == ["--seed", str(SEED)]
    assert config["vec"]["total_agents"] == AGENTS == EPISODES == 64
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["reset_position_noise_xy"] == 0.0
    assert environment["reset_position_noise_z"] == 0.0
    assert environment["gate_position_domain_randomize"] == 0


def test_sf052_collection_predicate_requires_real_final_terminals() -> None:
    lengths = np.full(AGENTS, 486, dtype=np.int32)
    terminals = np.ones(AGENTS, dtype=np.int32)
    terminal_is_last = np.ones(AGENTS, dtype=bool)
    metrics = {
        "env/n": float(EPISODES),
        "env/timeout": 0.0,
        "env/out_of_order": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }
    assert COLLECTION_STEP_LIMIT == 768
    assert exact_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_is_last,
        labels=int(lengths.sum()),
        executed_action_max_error=1e-7,
    )
    terminals[0] = 0
    assert not exact_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_is_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
