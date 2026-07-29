from __future__ import annotations

import numpy as np

from scripts.collect_vq2_recurrent_dagger_next import (
    AGENTS,
    COLLECTION_STEP_LIMIT,
    EPISODES,
    SEED,
    next_dagger_collection_passes,
    next_dagger_config,
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


def test_next_dagger_is_new_sf022_distribution_with_zero_blend() -> None:
    config, overrides = next_dagger_config(FakePufferl)
    environment = config["env"]
    assert overrides[:2] == ["--seed", str(SEED)]
    assert AGENTS == EPISODES == 512
    assert COLLECTION_STEP_LIMIT == 2048
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["w_action_teacher"] == 0.0


def test_next_admission_retains_crashes_but_rejects_transport_faults() -> None:
    metrics = {
        "env/n": float(EPISODES),
        "env/crash": 0.9,
        "env/timeout": 0.0,
        "env/out_of_order": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }
    lengths = np.full(AGENTS, 400, dtype=np.int32)
    terminals = np.ones(AGENTS, dtype=np.int32)
    last = np.ones(AGENTS, dtype=bool)
    assert next_dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
    metrics["env/timeout"] = 0.1
    assert not next_dagger_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
