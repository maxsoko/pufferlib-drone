import numpy as np
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from scripts.continue_vq2_plane_representation_offline import (
    _changed_model_keys,
    _initialize_probe_ridge,
    _progress_metrics,
    _rolling_samples,
)


def test_rolling_samples_preserve_event_major_target_order():
    observation = torch.zeros(3, 7, ENV_OBS_SIZE)
    action = torch.zeros(3, 7, 4)
    for event in range(3):
        for step in range(7):
            observation[event, step, LEGAL_OBS_SIZE + 3] = event * 10 + step
    obs, act, target = _rolling_samples(
        observation,
        action,
        np.asarray([0, 2]),
        targets=(4, 6),
        context=3,
    )
    assert obs.shape == (4, 3, ENV_OBS_SIZE)
    assert act.shape == (4, 3, 4)
    assert target.tolist() == [3.0, 5.0, 23.0, 25.0]


def test_progress_metrics_reject_constant_by_ordering():
    target = np.asarray([[0.4, 0.3, 0.2], [0.5, 0.4, 0.3]])
    prediction = np.zeros_like(target)
    result = _progress_metrics(target, prediction)
    assert result["correlation"] == 0.0
    assert result["strictly_decreasing_window_fraction"] == 0.0
    assert result["final_is_minimum_window_fraction"] == 0.0


def test_change_scope_rejects_actor_and_allows_posterior():
    before = {
        "rssm.posterior.0.weight": torch.zeros(1),
        "actor.0.weight": torch.zeros(1),
    }
    after = {
        "rssm.posterior.0.weight": torch.ones(1),
        "actor.0.weight": torch.zeros(1),
    }
    changed, forbidden = _changed_model_keys(before, after)
    assert changed == ["rssm.posterior.0.weight"]
    assert forbidden == []
    after["actor.0.weight"] = torch.ones(1)
    _changed, forbidden = _changed_model_keys(before, after)
    assert forbidden == ["actor.0.weight"]


def test_ridge_initialization_recovers_normalized_linear_target():
    feature = torch.tensor(
        [[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=torch.float32,
    )
    target = feature[:, 0] * 0.5
    probe = torch.nn.Linear(2, 1)
    summary = _initialize_probe_ridge(probe, feature, target, ridge=1e-6)
    prediction = probe(feature).squeeze(-1)
    assert summary["active_features"] == 1
    assert torch.allclose(prediction, target, atol=1e-4)
