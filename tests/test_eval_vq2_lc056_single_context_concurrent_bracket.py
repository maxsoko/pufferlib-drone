from __future__ import annotations

import scripts.eval_vq2_lc056_single_context_concurrent_bracket as lc056


def test_candidate_parity_requires_all_decision_fields() -> None:
    expected = {
        "alpha": 0.01,
        "candidate_state_sha256": "a" * 64,
        "mean_gates_passed": 3.5,
        "maximum_raw_index": 7,
        "maximum_raw_index_distribution": {"7": 1},
        "crash_rate": 0.0,
        "miss_rate": 0.25,
        "timeout_rate": 0.75,
        "diagnostic_valid": True,
    }
    assert lc056.candidate_result_matches(dict(expected), expected)
    changed = dict(expected)
    changed["mean_gates_passed"] = 3.51
    assert not lc056.candidate_result_matches(changed, expected)


def test_concurrent_layout_is_four_paired_32_agent_vectors() -> None:
    assert len(lc056.ALPHAS) == 4
    assert lc056.AGENTS_PER_CANDIDATE == 32
    assert lc056.EPISODES_PER_CANDIDATE == 32
    assert lc056.THREADS_PER_CANDIDATE == 32
    assert lc056.MINIMUM_SPEEDUP == 2.0
