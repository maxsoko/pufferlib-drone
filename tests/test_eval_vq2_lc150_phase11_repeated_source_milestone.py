from __future__ import annotations

import scripts.eval_vq2_lc150_phase11_repeated_source_milestone as lc150


def test_lc150_source_lock() -> None:
    parent = lc150.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc150.TARGET_RAW_INDEX == 12


def test_lc150_repeats_exact_source_seed() -> None:
    assert 2 * lc150.GROUP_SIZE == 256
    assert lc150.ENV_SEED_GROUP_SIZE == 1
    assert lc150.ENV_SEED_INDEX_OFFSET == 15
    assert lc150.PAIR_SIZE == 256


def test_lc150_requires_complete_repeated_source_gain() -> None:
    baseline = {
        "target_passes": 0, "paired_target_gains_vs_baseline": 0,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 128,
    }
    partial = {
        "target_passes": 127, "paired_target_gains_vs_baseline": 127,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 1,
    }
    complete = {
        **partial, "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "pre_target_terminals": 0,
    }
    assert lc150.choose_candidate([baseline, partial]) is None
    assert lc150.choose_candidate([baseline, complete]) is complete
