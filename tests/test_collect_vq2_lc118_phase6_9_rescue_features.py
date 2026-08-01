from __future__ import annotations

import numpy as np

import scripts.collect_vq2_lc118_phase6_9_rescue_features as lc118


def test_lc118_source_lock_and_exact_counts() -> None:
    lc118.verify_inputs()
    assert lc118.PHASE_MIN == 6
    assert lc118.PHASE_MAX_EXCLUSIVE == 10
    assert lc118.EXPECTED_RECORDS == 120_164
    assert lc118.EXPECTED_QUERY_AGENTS == 38
    assert lc118.EXPECTED_SUCCESS_AGENTS == 8


def test_lc118_teacher_window_selector() -> None:
    student = np.zeros((4, 4), dtype=np.float32)
    teacher = np.ones_like(student)
    active = np.ones(4, dtype=bool)
    phase = np.array([5, 6, 9, 10], dtype=np.int32)
    plant, mask = lc118.select_window_plant_actions(student, teacher, active, phase)
    assert mask.tolist() == [False, True, True, False]
    assert np.array_equal(plant[mask], teacher[mask])
