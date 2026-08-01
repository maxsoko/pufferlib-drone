from __future__ import annotations

import scripts.train_vq2_lc194_phase16_17_full_residual_fit as target


def test_source_lock_constants() -> None:
    assert target.PHASES == (16, 17)
    assert target.OPTIMIZER_STEPS == 512
    assert target.PARENT_CHECKPOINT_SHA256 == "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
    assert target.FEATURES_SHA256 == "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
    assert target.DATASET_REPORT_SHA256 == "50b1a311a153dc0277fa2ba85d97e043cd31b022aa01f5b519acd0df83c10b64"


def test_wrapper_restores_prior_globals() -> None:
    names = ("TAG", "SCHEMA", "PHASES", "LEARNING_RATE", "verify_inputs")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure()
    try:
        assert target.prior.PHASES == (16, 17)
        assert target.prior.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
