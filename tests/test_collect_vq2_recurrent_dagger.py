from __future__ import annotations

import numpy as np

from scripts.collect_vq2_recurrent_dagger import (
    AGENTS,
    EPISODES,
    SEED,
    dagger_collection_passes,
    dagger_config,
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


def test_dagger_config_executes_student_with_zero_teacher_blend() -> None:
    config, overrides = dagger_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert environment["evaluation_episode_limit"] == 1
    assert environment["num_gates"] == 6
    assert environment["gate_radius"] == 0.75
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0
    assert AGENTS == EPISODES == 64


def _metrics() -> dict[str, float]:
    return {
        "env/n": float(EPISODES),
        "env/crash": 0.0,
        "env/timeout": 0.0,
        "env/out_of_order": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }


def test_dagger_admission_allows_misses_but_requires_exact_safe_layout() -> None:
    metrics = _metrics()
    metrics["env/missed_gate"] = 1.0
    lengths = np.full(AGENTS, 10, dtype=np.int32)
    terminals = np.ones(AGENTS, dtype=np.int32)
    last = np.ones(AGENTS, dtype=bool)
    assert dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
    metrics["env/crash"] = 0.01
    assert not dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
