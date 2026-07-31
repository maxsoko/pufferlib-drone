from __future__ import annotations

import scripts.eval_vq2_lc067_phase2_failure_confirmation as lc067
from scripts.eval_vq2_variable_gate_oracle import sha256_path


def test_lc067_keeps_total_work_at_256_episodes() -> None:
    assert lc067.GROUP_SIZE * len(lc067.COEFFICIENTS) == 256
    assert sha256_path(lc067.LC066_REPORT) == lc067.LC066_REPORT_SHA256


def test_lc067_binds_the_selected_threshold_candidate() -> None:
    payload = lc067.verify_inputs()
    assert payload["schema"] == "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
