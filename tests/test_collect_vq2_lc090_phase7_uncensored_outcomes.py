from __future__ import annotations

import scripts.collect_vq2_lc090_phase7_uncensored_outcomes as lc090


def test_lc090_complete_phase7_contract() -> None:
    assert lc090.AGENTS == lc090.EPISODES == 512
    assert lc090.THREADS == 32
    assert lc090.TARGET_PHASE == 7
    assert lc090.STEP_LIMIT == 12_000
    assert lc090.MINIMUM_RECORDS == 10_000
    assert lc090.MINIMUM_QUERY_AGENTS == 8
    assert lc090.MINIMUM_OUTCOME_CLASS_AGENTS == 2


def test_lc090_rejection_chain_is_source_locked() -> None:
    lc090.verify_inputs()


def test_lc090_outcome_predicate_rejects_censoring() -> None:
    report = {
        "num_gates": 24,
        "agents": 512,
        "episodes": 512,
        "completed_agents": 511,
        "stopped_on_record_target": False,
        "vector_steps": 12_000,
        "feature_records": 10_000,
        "query_agents": 10,
        "query_outcome_complete_agents": 9,
        "query_outcome_success_agents": 2,
        "query_outcome_failure_agents": 7,
        "query_outcome_censored_agents": 1,
        "metrics": {},
    }
    predicates = lc090.predicates(report)
    assert not predicates["all_trajectories_completed"]
    assert not predicates["uncensored_target_outcomes"]
