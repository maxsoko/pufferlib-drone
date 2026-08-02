from __future__ import annotations

import scripts.eval_vq2_lc218_all24_action_sequence_confirmation as target


def test_lc218_source_lock() -> None:
    parent = target.verify_inputs()
    assert parent["model"]["sequence_length"] == 2_329


def test_lc218_confirmation_contract() -> None:
    assert target.SEED != target.base.SEED
    assert target.base.TARGET_RAW_INDEX == 24
    assert target.base.GROUP_SIZE == 128
    assert target.base.PAIR_SIZE == 256
