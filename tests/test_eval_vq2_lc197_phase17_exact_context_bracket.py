from __future__ import annotations

import scripts.eval_vq2_lc197_phase17_exact_context_bracket as target


def test_exact_context_contract() -> None:
    assert target.GROUP_SIZE == 128
    assert target.PAIR_SIZE == 256
    assert target.SCALE_PAIRS == ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0))
    assert target.LC196_REPORT_SHA256 == "32de9ea322b25d8e126be6bafce108c322d9bc3260e1b22775f0a787ca83c8cb"


def test_wrapper_restores_prior_globals() -> None:
    names = ("TAG", "GROUP_SIZE", "PAIR_SIZE", "SCALE_PAIRS", "verify_inputs", "configure")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure_wrapper()
    try:
        assert target.prior.GROUP_SIZE == 128
        assert target.prior.PAIR_SIZE == 256
        assert target.prior.verify_inputs is target.verify_inputs
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
