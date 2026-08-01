from __future__ import annotations

import scripts.collect_vq2_lc193_phase16_17_rescue_dagger as target


def test_source_lock_constants() -> None:
    assert target.PHASE_MIN == 16
    assert target.PHASE_MAX_EXCLUSIVE == 18
    assert target.TARGET_RAW_INDEX == 18
    assert target.PARENT_CHECKPOINT_SHA256 == "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
    assert target.LC192_REPORT_SHA256 == "b9ffe592be2f00e20b67d1df8ff012498751ed9ce4a989ed1bda07e8b774f144"


def test_wrapper_restores_prior_globals() -> None:
    names = (
        "TAG", "SCHEMA", "FEATURE_SCHEMA", "MAX_STEPS", "TARGET_RAW_INDEX",
        "PHASE_MIN", "PHASE_MAX_EXCLUSIVE", "verify_inputs", "corrected_writer",
    )
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure()
    try:
        assert target.prior.PHASE_MIN == 16
        assert target.prior.PHASE_MAX_EXCLUSIVE == 18
        assert target.prior.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
