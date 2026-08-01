from __future__ import annotations

import scripts.eval_vq2_lc117_teacher_rescue_horizon_ladder as lc117


def test_lc117_source_lock_and_descending_horizon() -> None:
    lc117.verify_inputs()
    assert lc117.HORIZONS == (7, 6, 5)
    assert lc117.base.TEACHER_PHASE_MAX_EXCLUSIVE == 10
    assert lc117.base.base.TARGET_RAW_INDEX == 10


def test_lc117_rescue_predicate() -> None:
    report = {
        "diagnostic_valid": True,
        "training_oracle_rescued_phase8_9": True,
        "items": [
            {"target_passes": 0},
            {
                "target_passes": 1,
                "paired_target_gains_vs_control": 1,
                "paired_target_losses_vs_control": 0,
            },
        ],
    }
    assert lc117.rung_rescued(report)
