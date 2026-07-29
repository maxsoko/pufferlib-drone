import json

import numpy as np
import torch

from pufferlib.vq2_dreamer import RSSMState
from scripts.continue_vq2_success_prefix_representation_offline import (
    _changed_model_keys,
    _crop_initial_state,
    _replay_geometry,
    _soft_posterior_features,
)


def test_soft_posterior_features_carry_logit_gradient() -> None:
    deterministic = torch.zeros(2, 3, requires_grad=True)
    logits = torch.randn(2, 2, 3, requires_grad=True)
    hard = torch.nn.functional.one_hot(logits.argmax(-1), 3).float()
    state = RSSMState(deterministic, hard, logits)
    weight = torch.arange(9, dtype=torch.float32)
    loss = (_soft_posterior_features(state) * weight).sum()
    loss.backward()
    assert deterministic.grad is not None
    assert logits.grad is not None
    assert float(logits.grad.abs().max()) > 0.0


def test_crop_initial_state_uses_stored_then_replayed_boundary() -> None:
    arrays = {
        "initial_deterministic": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "initial_logits": np.asarray([[[0.0, 1.0]]], dtype=np.float32),
        "initial_stochastic_index": np.asarray([[1]], dtype=np.uint8),
    }
    reference = {
        "deterministic": np.asarray([[[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        "logits": np.asarray([[[[2.0, 0.0]], [[0.0, 3.0]]]], dtype=np.float32),
    }
    state = _crop_initial_state(
        arrays,
        reference,
        np.asarray([0, 0]),
        np.asarray([0, 0]),
        np.asarray([0, 2]),
        device=torch.device("cpu"),
    )
    assert state.deterministic.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert state.stochastic.argmax(-1).tolist() == [[1], [1]]


def test_changed_scope_accepts_representation_and_rejects_actor() -> None:
    before = {
        "rssm.encoder.0.weight": torch.zeros(1),
        "rssm.sequence.weight_ih": torch.zeros(1),
        "rssm.posterior.0.weight": torch.zeros(1),
        "actor.0.weight": torch.zeros(1),
    }
    after = {name: value.clone() for name, value in before.items()}
    after["rssm.encoder.0.weight"].fill_(1)
    after["rssm.sequence.weight_ih"].fill_(1)
    after["rssm.posterior.0.weight"].fill_(1)
    changed, forbidden = _changed_model_keys(before, after)
    assert len(changed) == 3
    assert forbidden == []
    after["actor.0.weight"].fill_(1)
    _changed, forbidden = _changed_model_keys(before, after)
    assert forbidden == ["actor.0.weight"]


def test_replay_geometry_comes_from_validated_metadata(tmp_path) -> None:
    metadata = {
        "schema": "vq2_quantized_sequence_replay_v1",
        "capacity": 17,
        "agents": 3,
        "size": 11,
        "position": 5,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert _replay_geometry(tmp_path) == (17, 3)
    metadata["size"] = 18
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    try:
        _replay_geometry(tmp_path)
    except RuntimeError as error:
        assert "bounds" in str(error)
    else:
        raise AssertionError("invalid replay metadata was accepted")
