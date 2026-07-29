from __future__ import annotations

import scripts.train_vq2_recurrent_gate2_causal_fit as fit
from scripts.continue_vq2_recurrent_gate2_causal_fit import (
    CONFIG,
    PARENT_CHECKPOINT_SHA256,
    configure_fit_module,
)


def test_sf038_changes_only_seed_epochs_learning_rate_and_parent() -> None:
    assert CONFIG.seed == 42038
    assert CONFIG.epochs == 8
    assert CONFIG.learning_rate == 2e-5
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert CONFIG.loss_action_weights == (1.0, 4.0, 4.0, 1.0)
    assert CONFIG.prefer_admitted is True
    assert PARENT_CHECKPOINT_SHA256.startswith("0526cb27")


def test_sf038_reuses_sf037_causal_data_contract() -> None:
    names = (
        "TAG",
        "SCHEMA",
        "DEFAULT_OUTPUT",
        "PREREGISTRATION",
        "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256",
        "PARENT_REPORT",
        "PARENT_REPORT_SHA256",
        "CONFIG",
    )
    previous = {name: getattr(fit, name) for name in names}
    try:
        configure_fit_module()
        assert fit.CAUSAL_HORIZON_STEPS == 768
        assert fit.DAGGER_SOURCE_PATTERN == ("broad", "next", "next")
        assert fit.PARENT_CHECKPOINT_SHA256 == PARENT_CHECKPOINT_SHA256
    finally:
        for name, value in previous.items():
            setattr(fit, name, value)

