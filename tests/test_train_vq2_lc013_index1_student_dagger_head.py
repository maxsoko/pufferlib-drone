from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc013_index1_student_dagger_head as lc013
import scripts.train_vq2_vg068_indexed_mlp_intervention_features as trainer


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


def test_equal_agent_loss_prevents_long_trajectory_domination() -> None:
    prediction = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    teacher = torch.tensor([[1.0], [1.0], [1.0], [3.0]])
    agents = np.array([0, 0, 0, 1], dtype=np.int64)
    counts = np.array([3, 1], dtype=np.int64)
    ordinary = trainer.phase_training_loss(
        prediction, teacher, agents, counts, equal_agent_weighting=False
    )
    balanced = trainer.phase_training_loss(
        prediction, teacher, agents, counts, equal_agent_weighting=True
    )
    assert ordinary.item() == 3.0
    assert abs(balanced.item() - 5.0) < 1e-6
