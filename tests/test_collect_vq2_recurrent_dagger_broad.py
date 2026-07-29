from __future__ import annotations

import numpy as np

from scripts.collect_vq2_recurrent_dagger_broad import (
    AGENTS,
    COLLECTION_STEP_LIMIT,
    EPISODES,
    SEED,
    broad_dagger_collection_passes,
    broad_dagger_config,
)


class FakePufferl:
    @staticmethod
    def load_config(_name: str) -> dict:
        import sys

        pairs = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
        return {
            "backend_env_name": "drone_race_vision",
            "vec": {"total_agents": int(pairs["--vec.total-agents"])},
            "env": {"start_gate_index": 0, "max_steps": 11520},
        }


def test_broad_config_is_sf014_only_and_expands_course_coverage() -> None:
    config, overrides = broad_dagger_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert environment["evaluation_episode_limit"] == 1
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0
    assert AGENTS == EPISODES == 512
    assert COLLECTION_STEP_LIMIT == 512


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


def test_broad_admission_requires_every_clean_terminal_inside_cap() -> None:
    metrics = _metrics()
    lengths = np.full(AGENTS, 250, dtype=np.int32)
    terminals = np.ones(AGENTS, dtype=np.int32)
    last = np.ones(AGENTS, dtype=bool)
    assert broad_dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
    terminals[-1] = 0
    assert not broad_dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
