from __future__ import annotations

import numpy as np

import scripts.eval_vq2_lc116_phase8_9_teacher_rescue as lc116


def test_lc116_source_lock_and_phase_window() -> None:
    lc116.verify_inputs()
    assert lc116.TEACHER_PHASE_MIN == 8
    assert lc116.TEACHER_PHASE_MAX_EXCLUSIVE == 10
    assert lc116.base.TARGET_RAW_INDEX == 10


def test_lc116_teacher_is_candidate_phase8_9_only() -> None:
    student = np.zeros((lc116.base.TOTAL_AGENTS, 4), dtype=np.float32)
    teacher = np.ones_like(student)
    active = np.ones(lc116.base.TOTAL_AGENTS, dtype=bool)
    phase = np.full(lc116.base.TOTAL_AGENTS, 8, dtype=np.int32)
    phase[lc116.base.group_slice(1).start] = 7
    phase[lc116.base.group_slice(1).start + 1] = 10

    plant, mask = lc116.select_plant_actions(student, teacher, active, phase)

    assert not mask[lc116.base.group_slice(0)].any()
    assert mask[lc116.base.group_slice(1)].sum() == lc116.base.GROUP_SIZE - 2
    assert np.all(plant[mask] == 1.0)
