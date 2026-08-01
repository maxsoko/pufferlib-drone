from __future__ import annotations

import scripts.collect_vq2_lc152_phase11_failure_state_dagger as lc152


def test_lc152_source_lock() -> None:
    parent = lc152.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc152.TARGET_RAW_INDEX == 12


def test_lc152_enables_failure_state_teacher_labels() -> None:
    originals = lc152.configure()
    try:
        assert lc152.base.CAPTURE_CONTROL_FEATURES
        assert lc152.base.CAPTURE_CONTROL_TEACHER_TARGETS
        assert lc152.base.ENV_SEED_GROUP_SIZE == 1
        assert lc152.base.ENV_SEED_INDEX_OFFSET == 15
    finally:
        lc152.restore(originals)
