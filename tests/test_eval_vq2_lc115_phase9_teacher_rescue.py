from __future__ import annotations

import numpy as np

import scripts.eval_vq2_lc115_phase9_teacher_rescue as lc115


def test_lc115_source_lock_and_exact_pair_contract() -> None:
    payload = lc115.verify_inputs()
    assert payload["schema"] == "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
    assert lc115.GROUP_SIZE == 128
    assert lc115.TOTAL_AGENTS == 256
    assert lc115.SEED == 432050
    assert lc115.TARGET_PHASE == 9
    assert lc115.TARGET_RAW_INDEX == 10


def test_lc115_teacher_is_candidate_phase9_only() -> None:
    student = np.zeros((lc115.TOTAL_AGENTS, 4), dtype=np.float32)
    teacher = np.full_like(student, 0.5)
    active = np.ones(lc115.TOTAL_AGENTS, dtype=bool)
    phase = np.full(lc115.TOTAL_AGENTS, lc115.TARGET_PHASE, dtype=np.int32)
    phase[lc115.group_slice(1).start] = lc115.TARGET_PHASE - 1

    plant, teacher_mask = lc115.select_plant_actions(student, teacher, active, phase)

    assert not teacher_mask[lc115.group_slice(0)].any()
    assert teacher_mask[lc115.group_slice(1)].sum() == lc115.GROUP_SIZE - 1
    assert np.array_equal(plant[lc115.group_slice(0)], student[lc115.group_slice(0)])
    assert np.all(plant[teacher_mask] == 0.5)
