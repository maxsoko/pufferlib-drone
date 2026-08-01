from __future__ import annotations

import scripts.eval_vq2_lc195_phase16_17_teacher_free_milestone as target


def test_source_lock_constants() -> None:
    assert target.TARGET_RAW_INDEX == 18
    assert target.GROUP_SIZE == 128
    assert target.CANDIDATE_CHECKPOINT_SHA256 == "f175dfc7a568cd8dd2ed95e9a9baf558e5351dd6f3df1f065a1ce1ceece7158b"
    assert target.CANDIDATE_REPORT_SHA256 == "5fe3258953eaf75d15059800939771e88ccf97599d1229032b6346d2d75be796"


def test_selection_requires_raw17_retention_and_raw18_completion() -> None:
    baseline = {
        "target_passes": 0, "pre_target_terminals": 128,
        "maximum_raw_index_distribution": {"17": 128},
    }
    candidate = {
        "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert target.choose_candidate([baseline, candidate]) is candidate
    baseline["maximum_raw_index_distribution"]["17"] = 127
    assert target.choose_candidate([baseline, candidate]) is None
