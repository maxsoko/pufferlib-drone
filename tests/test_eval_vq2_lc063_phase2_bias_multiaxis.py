from __future__ import annotations

import scripts.eval_vq2_lc063_phase2_bias_multiaxis as lc063
from scripts.eval_vq2_variable_gate_oracle import sha256_path


def test_lc063_keeps_one_fast_256_episode_vector() -> None:
    assert lc063.GROUP_SIZE * len(lc063.BIAS_CANDIDATES) == 256
    assert lc063.BIAS_CANDIDATES[0][1] == (0.0, 0.0, 0.0, 0.0)
    assert sha256_path(lc063.PARENT_CHECKPOINT) == lc063.PARENT_CHECKPOINT_SHA256
    assert sha256_path(lc063.PARENT_REPORT) == lc063.PARENT_REPORT_SHA256


def test_lc063_parent_is_promoted_lc062() -> None:
    payload = lc063.verify_inputs()
    assert payload["schema"] == "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
    assert payload["numerically_admitted"]
