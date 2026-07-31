from __future__ import annotations

import scripts.eval_vq2_lc061_phase2_bias_full_course as lc061
import scripts.eval_vq2_lc062_phase2_bias_full_course as lc062
from scripts.eval_vq2_variable_gate_oracle import sha256_path


def test_lc061_pair_is_128_full_course_episodes() -> None:
    assert lc061.GROUPS == 2
    assert lc061.GROUP_SIZE == 64
    assert lc061.TOTAL_AGENTS == 128
    assert lc061.NUM_GATES == 24
    assert lc061.MAX_STEPS == 12_000
    assert sha256_path(lc061.LC060_REPORT) == lc061.LC060_REPORT_SHA256


def test_lc061_selection_requires_progress_and_safety() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.5,
        "gate3_passes": 20, "crash_rate": 0.05, "maximum_raw_index": 8,
    }
    candidate = {
        "transport_pass": True, "mean_gates_passed": 3.56,
        "gate3_passes": 21, "crash_rate": 0.05, "maximum_raw_index": 8,
    }
    assert lc061.choose_candidate(parent, candidate)
    candidate["crash_rate"] = 0.06
    assert not lc061.choose_candidate(parent, candidate)


def test_lc062_uses_a_distinct_report_and_checkpoint_schema() -> None:
    assert lc062.TAG != lc061.TAG
    assert lc062.SCHEMA != lc061.SCHEMA
    assert lc062.CHECKPOINT_SCHEMA != lc061.CHECKPOINT_SCHEMA
    assert lc062._BASE_VERIFY_INPUTS is not lc062.verify_inputs
