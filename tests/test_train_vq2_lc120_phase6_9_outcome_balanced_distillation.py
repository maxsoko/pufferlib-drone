from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc120_phase6_9_outcome_balanced_distillation as lc120


def test_lc120_source_lock_and_training_contract() -> None:
    payload = lc120.verify_inputs()
    assert payload["schema"] == "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
    assert lc120.PHASES == (6, 7, 8, 9)
    assert lc120.OPTIMIZER_STEPS == 512
    assert lc120.BATCH_SIZE == 4096


def test_lc120_stratified_split_keeps_both_classes() -> None:
    present = np.array([128, 129, 130, 131, 132, 133], dtype=np.int64)
    outcome = np.zeros(256, dtype=np.uint8)
    outcome[[128, 129, 130]] = 1
    train, validation = lc120.stratified_agent_split(present, outcome, phase=6)
    assert set(outcome[train]) == {0, 1}
    assert set(outcome[validation]) == {0, 1}
