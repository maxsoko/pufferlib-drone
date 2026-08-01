from __future__ import annotations

import scripts.train_vq2_lc199_phase16_success_anchor_fit as target


def test_fit_contract() -> None:
    assert target.TARGET_PHASE == 16
    assert len(target.SUCCESS_AGENTS) == len(target.CONTROL_AGENTS) == 256
    assert target.TRAINING_SUCCESS_COUNT == 192
    assert target.SCALES == (0.003, 0.01, 0.03, 0.1, 0.3)
    assert target.MAXIMUM_CONTROL_PARENT_DRIFT_MSE == 2e-4
    assert target.LC198_REPORT_SHA256 == "050fbc1b097ac03e5397adab9c52f7e8e11bc7d3e1dc39f7d6506f8954050c16"


def test_wrapper_restores_base_globals() -> None:
    names = ("TAG", "TARGET_PHASE", "SCALES", "verify_inputs")
    before = tuple(getattr(target.base, name) for name in names)
    snapshot = target.configure()
    try:
        assert target.base.TARGET_PHASE == 16
        assert target.base.SCALES == target.SCALES
        assert target.base.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.base, name) for name in names) == before
