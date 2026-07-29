import torch

from scripts.continue_vq2_plane_sequence_offline import (
    _allowed_changes,
    _apply_standardized_probe,
    _fit_standardized_probe,
)


def test_standardized_probe_fits_linear_deterministic_feature():
    feature = torch.tensor(
        [[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=torch.float32,
    )
    target = feature[:, 0] * 0.5
    probe = _fit_standardized_probe(feature, target, ridge=1e-6)
    prediction = _apply_standardized_probe(feature, probe)
    assert probe["active_features"] == 1
    assert torch.allclose(prediction, target, atol=1e-4)


def test_sequence_only_change_scope():
    before = {
        "rssm.sequence.weight_ih": torch.zeros(1),
        "rssm.posterior.0.weight": torch.zeros(1),
    }
    after = {
        "rssm.sequence.weight_ih": torch.ones(1),
        "rssm.posterior.0.weight": torch.zeros(1),
    }
    changed, forbidden = _allowed_changes(before, after)
    assert changed == ["rssm.sequence.weight_ih"]
    assert forbidden == []
    after["rssm.posterior.0.weight"] = torch.ones(1)
    _changed, forbidden = _allowed_changes(before, after)
    assert forbidden == ["rssm.posterior.0.weight"]
