from __future__ import annotations

import scripts.analyze_vq2_dagger_causal_horizon as analysis
from scripts.analyze_vq2_gate2_causal_horizon import (
    BINS,
    CAPS,
    CHECKPOINT_SHA256,
    DATASET_REPORT_SHA256,
    configure_analysis_module,
)


def test_sf036_uses_complete_nonoverlapping_2048_step_bins() -> None:
    assert BINS[0] == (0, 256)
    assert BINS[-1] == (1792, 2048)
    assert all(left[1] == right[0] for left, right in zip(BINS, BINS[1:]))
    assert CAPS == tuple(range(256, 2049, 256))


def test_sf036_binds_sf033_and_sf035() -> None:
    names = (
        "TAG",
        "DEFAULT_OUTPUT",
        "PREREGISTRATION",
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
        "DATASET",
        "DATASET_REPORT_SHA256",
        "DATASET_METADATA_SHA256",
        "BINS",
        "CAPS",
    )
    previous = {name: getattr(analysis, name) for name in names}
    try:
        configure_analysis_module()
        assert analysis.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
        assert analysis.DATASET_REPORT_SHA256 == DATASET_REPORT_SHA256
        assert analysis.BINS == BINS
    finally:
        for name, value in previous.items():
            setattr(analysis, name, value)

