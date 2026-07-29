from __future__ import annotations

import numpy as np
import torch

from scripts import collect_vq2_c009_measured_visual_suffix_dagger as collect


def test_candidate_prefix_uses_c009_self_replay():
    prefix = collect.candidate_warm_prefix(torch.device("cpu"))
    assert prefix["visual_observation"].shape == (208, 4119)
    assert prefix["suffix_action"].shape == (208, 4)
    actor, _, _ = collect.handoff.load_candidate_suffix(torch.device("cpu"), "c009")
    replayed, state = collect.handoff.warm_suffix_stepwise(
        actor, prefix["visual_observation"], device=torch.device("cpu")
    )
    assert np.array_equal(replayed, prefix["suffix_action"])
    assert state.shape[1] == 1


def test_configuration_selects_c009_without_changing_collection_boundary():
    previous = {
        "tag": collect.base.TAG,
        "output": collect.base.DEFAULT_OUTPUT,
        "seed": collect.base.SEED,
        "load_suffix": collect.base.load_suffix,
        "load_warm_prefix": collect.base.load_warm_prefix,
    }
    try:
        collect.configure_base(torch.device("cpu"))
        assert collect.base.TAG == collect.TAG
        assert collect.base.SEED == 43011
        actor, payload = collect.base.load_suffix(torch.device("cpu"))
        assert actor.training is False
        assert payload["combined_numerical_admission"] is True
        assert collect.base.AGENTS == 128
        assert collect.base.ALIAS_START_AGENT == 64
    finally:
        collect.base.TAG = previous["tag"]
        collect.base.DEFAULT_OUTPUT = previous["output"]
        collect.base.SEED = previous["seed"]
        collect.base.load_suffix = previous["load_suffix"]
        collect.base.load_warm_prefix = previous["load_warm_prefix"]
