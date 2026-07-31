from __future__ import annotations

import scripts.train_vq2_lc013_index1_student_dagger_head as lc013


def test_fit_is_small_phase_one_only_and_preserves_other_rows() -> None:
    assert lc013.TARGET_PHASES == (1,)
    assert lc013.CONFIG.epochs == 10
    assert lc013.CONFIG.learning_rate == 2e-4
    assert lc013.CONFIG.weight_decay == 0.0


def test_numerical_admission_requires_improvement_and_exact_freeze() -> None:
    metrics = {
        "phases": {
            "1": {
                "baseline_action_mse": 0.1,
                "selected_action_mse": 0.09,
                "improvement_factor": 0.1 / 0.09,
            }
        }
    }
    assert lc013.numerically_admitted(
        metrics, base_exact=True, non_target_zero=True, trainable_l2=31.0
    )
    assert not lc013.numerically_admitted(
        metrics, base_exact=True, non_target_zero=False, trainable_l2=31.0
    )
