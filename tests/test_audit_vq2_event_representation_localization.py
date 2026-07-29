from scripts.audit_vq2_event_representation_localization import (
    _classify_bottleneck,
    _passes_event_gate,
)


def _result(auc=0.95, ap=0.8, rank=0.8, phase=0.8):
    return {
        "test": {"roc_auc": auc, "average_precision": ap},
        "test_within_window_all_negative_rank_fraction": rank,
        "test_minimum_phase_roc_auc": phase,
    }


def test_event_gate_requires_every_metric_and_action_margin():
    assert _passes_event_gate(_result(), action_auc=0.5)
    assert not _passes_event_gate(_result(auc=0.89), action_auc=0.5)
    assert not _passes_event_gate(_result(), action_auc=0.90)


def test_bottleneck_classification_prior_and_posterior_cases():
    base = {
        "image_encoder_action": False,
        "sensor_encoder_action": False,
        "fused_encoder_action": False,
        "current_deterministic_action": False,
        "current_stochastic_action": False,
        "current_posterior_action": False,
        "next_deterministic_action": False,
        "next_stochastic_action": False,
        "next_prior_action": False,
    }
    assert _classify_bottleneck(base) == "no_frozen_linear_representation_separates_events"
    posterior = dict(base, current_posterior_action=True)
    assert _classify_bottleneck(posterior) == "learned_prior_transition"
    encoder = dict(base, fused_encoder_action=True)
    assert _classify_bottleneck(encoder) == "posterior_recurrent_compression"
