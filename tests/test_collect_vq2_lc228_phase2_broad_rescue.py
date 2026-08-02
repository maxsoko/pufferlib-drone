from __future__ import annotations

import torch

import scripts.collect_vq2_lc228_phase2_broad_rescue as target


def test_lc228_source_lock_and_action_sequence_loader() -> None:
    payload = target.verify_inputs()
    assert payload["model"]["class"] == "VQ2PhaseActionSequenceActor"
    actor = target.load_actor(payload, torch.device("cpu"))
    assert actor.phase_action_sequence.shape[0] == 9_592


def test_lc228_contract() -> None:
    assert target.GROUP_SIZE == 256
    assert target.TOTAL_AGENTS == 512
    assert target.PHASE_MIN == 2
    assert target.PHASE_MAX_EXCLUSIVE == 3
    assert target.TARGET_RAW_INDEX == 3
    assert target.ENV_SEED_INDEX_OFFSET == 287
    assert target.MAX_STEPS == 12_000


def test_lc228_sequence_feature_contract() -> None:
    class Result:
        pre_tanh_mean = torch.zeros(target.TOTAL_AGENTS, 4)

    recurrent = torch.zeros(1, target.TOTAL_AGENTS, 321)
    index = torch.tensor([0, 255, 256, 511])
    hidden, pre_tanh = target.sequence_feature_components(
        object(), Result(), recurrent, index
    )
    assert hidden.shape == (4, 256)
    assert pre_tanh.shape == (4, 4)
