from __future__ import annotations

import scripts.eval_vq2_recurrent_dagger_broad as screen
from scripts.eval_vq2_recurrent_dagger_causal_prefix import (
    AGENTS,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    EPISODES,
    SEED,
    TAG,
    configure_screen_module,
)


def test_sf029_binds_fresh_full_course_screen() -> None:
    names = (
        "TAG",
        "DEFAULT_OUTPUT",
        "PREREGISTRATION",
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
        "AGENTS",
        "EPISODES",
        "SEED",
        "load_sf022",
    )
    previous = {name: getattr(screen, name) for name in names}
    try:
        configure_screen_module()
        assert screen.TAG == TAG
        assert screen.CHECKPOINT == CHECKPOINT
        assert screen.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
        assert screen.AGENTS == screen.EPISODES == AGENTS == EPISODES == 512
        assert screen.SEED == SEED == 42029
    finally:
        for name, value in previous.items():
            setattr(screen, name, value)
