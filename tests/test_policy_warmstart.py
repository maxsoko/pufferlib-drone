import torch

from pufferlib import pufferl


def test_warmstart_prefix_filter_loads_only_selected_keys():
    target_state = {
        "encoder.0.weight": torch.zeros(4, 8),
        "encoder.0.bias": torch.zeros(4),
        "decoder_mean.weight": torch.zeros(4, 4),
    }
    source_state = {
        "encoder.0.weight": torch.ones(4, 8),
        "encoder.0.bias": torch.ones(4),
        "decoder_mean.weight": torch.full((4, 4), 3.0),
    }

    merged, stats = pufferl._warmstart_state_dict(
        target_state=target_state,
        source_state=source_state,
        prefixes="encoder",
        encoder_overlap=False,
    )

    assert torch.allclose(merged["encoder.0.weight"], torch.ones(4, 8))
    assert torch.allclose(merged["encoder.0.bias"], torch.ones(4))
    assert torch.allclose(merged["decoder_mean.weight"], torch.zeros(4, 4))
    assert stats["loaded_exact"] == ["encoder.0.weight", "encoder.0.bias"]


def test_warmstart_encoder_overlap_copies_shared_slice_only():
    target_state = {
        "encoder.0.weight": torch.zeros(4, 10),
        "encoder.0.bias": torch.zeros(4),
    }
    source_state = {
        "encoder.0.weight": torch.ones(4, 6),
        "encoder.0.bias": torch.full((4,), 2.0),
    }

    merged, stats = pufferl._warmstart_state_dict(
        target_state=target_state,
        source_state=source_state,
        prefixes="encoder",
        encoder_overlap=True,
    )

    assert torch.allclose(merged["encoder.0.weight"][:, :6], torch.ones(4, 6))
    assert torch.allclose(merged["encoder.0.weight"][:, 6:], torch.zeros(4, 4))
    assert torch.allclose(merged["encoder.0.bias"], torch.full((4,), 2.0))
    assert ("encoder.0.weight", (4, 6)) in stats["loaded_overlap"]
