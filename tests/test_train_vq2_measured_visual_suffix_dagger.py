from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import train_vq2_measured_visual_suffix_dagger as fit


def test_c006_loader_keeps_128_intact_histories():
    dataset = fit.C006PublicPhaseDataset(verify_hashes=False)
    assert dataset.agents == 128
    assert dataset.time_steps == 687
    assert dataset.lengths.min() == 526
    assert dataset.lengths.max() == 687


def test_balanced_mapping_has_eight_validation_agents_per_group():
    dataset = fit.BalancedSuffixDataset(verify_c006_hashes=False)
    assert dataset.agents == 320
    assert np.bincount(dataset.source_for_agent).tolist() == [64] * 5
    validation = np.arange(dataset.agents - fit.CONFIG.validation_agents, dataset.agents)
    assert np.bincount(dataset.source_for_agent[validation]).tolist() == [8] * 5
    true = dataset.local_agent[dataset.source_for_agent == 3]
    alias = dataset.local_agent[dataset.source_for_agent == 4]
    assert true.tolist() == list(range(64))
    assert alias.tolist() == list(range(64, 128))


def test_balanced_chunk_routes_true_and_alias_c006_agents():
    dataset = fit.BalancedSuffixDataset(verify_c006_hashes=False)
    # Logical agents 3 and 4 are local agent 0 in C006 true/alias groups.
    observation, action, valid, gate2 = dataset.chunk(
        np.asarray([3, 4]), 0, 1, device=torch.device("cpu")
    )
    assert observation.shape == (2, 1, 4119)
    assert action.shape == (2, 1, 4)
    assert valid.all()
    assert gate2.all()
    # The aliased first mask is materially larger than the true-range mask.
    assert observation[1, 0, :4096].sum() > observation[0, 0, :4096].sum()


def test_source_audit_gate_rejects_bad_new_distribution():
    good = {
        name: {
            "phase_zero": {"weighted_mse": 0.01, "mse": [0.01] * 4},
            "gate2": {"weighted_mse": 0.01, "mse": [0.01] * 4},
        }
        for name in fit.GROUP_NAMES[:3]
    }
    good["c006_true"] = {"weighted_mse": 0.01, "mse": [0.01] * 4}
    good["c006_alias"] = {"weighted_mse": 0.01, "mse": [0.01] * 4}
    assert fit.source_audits_pass(good)
    bad = dict(good)
    bad["c006_alias"] = {"weighted_mse": 0.03, "mse": [0.01] * 4}
    assert not fit.source_audits_pass(bad)
