from __future__ import annotations

import scripts.eval_vq2_lc200_phase16_success_anchor_milestone as target


def test_source_lock_constants() -> None:
    assert target.CANDIDATE_CHECKPOINT_SHA256 == "ffceb4452072c87095728e3ffc4c6d68238383517a09b18dbbda9951cc25656c"
    assert target.CANDIDATE_REPORT_SHA256 == "f96e002f72e8a14fdea68c2becc55094ff3426bdd145571f5aca8b4ba54d5b0a"
    assert len(target.BIASES) == 2


def test_wrapper_restores_prior_globals() -> None:
    names = ("TAG", "BIASES", "CANDIDATE_CHECKPOINT", "verify_inputs", "configure")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure_wrapper()
    try:
        assert target.prior.TAG == target.TAG
        assert target.prior.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
