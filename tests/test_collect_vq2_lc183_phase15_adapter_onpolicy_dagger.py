from __future__ import annotations

import torch

from pufferlib.vq2_recurrent import RecurrentActorOutput
import scripts.collect_vq2_lc183_phase15_adapter_onpolicy_dagger as lc183


def test_lc183_source_lock() -> None:
    parent = lc183.verify_inputs()
    assert parent["numerically_admitted"]
    assert parent["model"]["adapter_target_phase"] == 15


def test_lc183_adapter_feature_extraction_excludes_adapter() -> None:
    parent = lc183.verify_inputs()
    actor = lc183.split_actor_loader(parent, torch.device("cpu"))
    recurrent = actor.initial_state(lc183.TOTAL_AGENTS, device="cpu")
    recurrent[0, :, :256] = 2.0
    recurrent[0, :, 256:] = 0.5
    residual = torch.cat([
        child.phase_adapter_output(recurrent[0, group * 256 : (group + 1) * 256, 256:])
        for group, child in enumerate(actor.actors)
    ])
    base = torch.full((lc183.TOTAL_AGENTS, 4), 0.25)
    result = RecurrentActorOutput(
        mean=torch.tanh(base + residual),
        pre_tanh_mean=base + residual,
        log_std=torch.zeros_like(base),
    )
    index = torch.tensor([0, 255, 256, 511])
    hidden, recovered = lc183.adapter_feature_components(actor, result, recurrent, index)
    assert hidden.shape == (4, 256)
    assert torch.equal(hidden, torch.full_like(hidden, 2.0))
    assert torch.allclose(recovered, torch.full_like(recovered, 0.25))


def test_lc183_configuration_roundtrip() -> None:
    original_hook = lc183.base.feature_components
    originals = lc183.configure()
    try:
        assert lc183.base.feature_components is lc183.adapter_feature_components
        assert lc183.base.CAPTURE_CONTROL_TEACHER_TARGETS
    finally:
        lc183.restore(originals)
    assert lc183.base.feature_components is original_hook
