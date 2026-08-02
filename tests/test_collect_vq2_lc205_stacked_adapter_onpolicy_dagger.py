from __future__ import annotations

import torch

import scripts.collect_vq2_lc205_stacked_adapter_onpolicy_dagger as target


def test_lc205_source_lock_and_actor_contract() -> None:
    parent = target.verify_inputs()
    actor = target.split_actor_loader(parent, torch.device("cpu"))
    assert len(actor.actors) == 2
    assert all(child.initial_state(256, device="cpu").shape == (1, 256, 384) for child in actor.actors)


def test_lc205_configuration_roundtrip() -> None:
    names = (
        "TAG", "GROUP_SIZE", "TOTAL_AGENTS", "PHASE_MIN",
        "PHASE_MAX_EXCLUSIVE", "feature_components",
    )
    before = tuple(getattr(target.base, name) for name in names)
    snapshot = target.configure()
    try:
        assert target.base.TAG == target.TAG
        assert target.base.feature_components is target.adapter_feature_components
        assert target.base.CAPTURE_CONTROL_TEACHER_TARGETS
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.base, name) for name in names) == before
