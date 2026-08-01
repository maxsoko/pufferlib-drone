from __future__ import annotations

import scripts.eval_vq2_lc201_phase16_success_anchor_scale_bracket as target


def test_scale_and_context_contract() -> None:
    assert target.GROUP_SIZE == 128
    assert target.PAIR_SIZE == 256
    assert target.SCALE_PAIRS == ((0.0, 0.0), (0.01, 0.0), (0.03, 0.0), (0.1, 0.0), (0.3, 0.0))
    assert target.LC200_REPORT_SHA256 == "ebcb8caed767b22709b45b225f7675570b55254ee4610e5c180050761bd42af1"


def test_wrapper_restores_prior_globals() -> None:
    names = ("TAG", "GROUP_SIZE", "PAIR_SIZE", "SCALE_PAIRS", "FIT_CHECKPOINT", "verify_inputs")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure_wrapper()
    try:
        assert target.prior.SCALE_PAIRS == target.SCALE_PAIRS
        assert target.prior.FIT_CHECKPOINT == target.FIT_CHECKPOINT
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
