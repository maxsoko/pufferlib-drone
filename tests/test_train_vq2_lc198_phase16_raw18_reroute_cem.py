from __future__ import annotations

import scripts.train_vq2_lc198_phase16_raw18_reroute_cem as target


def test_search_contract() -> None:
    assert target.TARGET_PHASE == 16
    assert target.TARGET_RAW_INDEX == 18
    assert target.GENERATIONS == 3
    assert target.PARENT_CHECKPOINT_SHA256 == "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
    assert target.LC197_REPORT_SHA256 == "6be49534abe4f571a43f5cc3290c17f9b3b84b7c62f31867d89b6a163f7f9cad"


def test_wrapper_restores_prior_globals() -> None:
    names = ("TAG", "TARGET_PHASE", "TARGET_RAW_INDEX", "GENERATIONS", "verify_inputs")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure()
    try:
        assert target.prior.TARGET_PHASE == 16
        assert target.prior.TARGET_RAW_INDEX == 18
        assert target.prior.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
