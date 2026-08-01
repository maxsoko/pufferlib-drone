from __future__ import annotations

import scripts.collect_vq2_lc079_phase6_uncensored_outcomes as lc079


def test_lc079_uses_complete_fresh_trajectory_contract() -> None:
    assert lc079.AGENTS == lc079.EPISODES == 256
    assert lc079.THREADS == 32
    assert lc079.TARGET_PHASE == 6
    assert lc079.STEP_LIMIT == 12_000
    assert lc079.MINIMUM_RECORDS == 20_000
    assert lc079.MINIMUM_OUTCOME_CLASS_AGENTS == 2


def test_lc079_rejection_chain_is_source_locked() -> None:
    lc079.verify_inputs()


def test_lc079_outcome_predicate_rejects_censoring() -> None:
    report = {
        "num_gates": 24,
        "agents": 256,
        "episodes": 256,
        "completed_agents": 255,
        "stopped_on_record_target": False,
        "vector_steps": 12_000,
        "feature_records": 20_000,
        "query_agents": 10,
        "query_outcome_complete_agents": 9,
        "query_outcome_success_agents": 2,
        "query_outcome_failure_agents": 7,
        "query_outcome_censored_agents": 1,
        "metrics": {},
    }
    predicates = lc079.predicates(report)
    assert not predicates["all_trajectories_completed"]
    assert not predicates["uncensored_target_outcomes"]
