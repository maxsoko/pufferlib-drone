from __future__ import annotations

import torch

import scripts.eval_vq2_lc217_all24_action_sequence_milestone as target


def test_lc217_source_lock_and_actor_states() -> None:
    target.verify_inputs()
    parent = target.load_actor(target.parent_payload(), torch.device("cpu"))
    candidate = target.load_actor(target.candidate_payload(), torch.device("cpu"))
    assert parent.initial_state(256, device="cpu").shape == (1, 256, 321)
    assert candidate.initial_state(256, device="cpu").shape == (1, 256, 321)
    assert parent.sequence_length == 2_329
    assert candidate.sequence_length == 9_592


def test_lc217_selection_contract() -> None:
    baseline = {
        "target_passes": 0,
        "maximum_raw_index_distribution": {"18": 128},
        "pre_target_terminals": 128,
    }
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert target.choose_candidate([baseline, candidate]) is candidate
    candidate["paired_target_losses_vs_baseline"] = 1
    assert target.choose_candidate([baseline, candidate]) is None


def test_lc217_course_contract() -> None:
    assert target.TARGET_RAW_INDEX == 24
    assert target.GROUP_SIZE == 128
    assert target.PAIR_SIZE == 256
    assert target.ENV_SEED_GROUP_SIZE == 1
    assert target.ENV_SEED_INDEX_OFFSET == 15
