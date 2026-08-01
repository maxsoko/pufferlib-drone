from __future__ import annotations

import scripts.collect_vq2_lc091_phase7_success_recapture as lc091


def test_lc091_source_locked_recapture_contract() -> None:
    assert lc091.AGENTS == lc091.EPISODES == 128
    assert lc091.SEED == 431_870
    assert lc091.TARGET_PHASE == 7
    assert lc091.EXPECTED_QUERY_AGENTS == 3
    assert lc091.EXPECTED_SUCCESS_AGENTS == 2
    assert lc091.EXPECTED_FAILURE_AGENTS == 1


def test_lc091_inputs_are_source_locked() -> None:
    lc091.verify_inputs()


def test_lc091_exact_outcome_predicate() -> None:
    report = {
        "num_gates": 24,
        "agents": 128,
        "episodes": 128,
        "completed_agents": 128,
        "stopped_on_record_target": False,
        "vector_steps": 12_000,
        "feature_records": 1_000,
        "query_agents": 3,
        "query_outcome_complete_agents": 3,
        "query_outcome_success_agents": 2,
        "query_outcome_failure_agents": 1,
        "query_outcome_censored_agents": 0,
        "metrics": {},
    }
    assert lc091.predicates(report)["source_locked_outcomes_exact"]
    report["query_outcome_success_agents"] = 1
    assert not lc091.predicates(report)["source_locked_outcomes_exact"]
