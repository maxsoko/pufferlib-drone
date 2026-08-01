from __future__ import annotations

import torch

import scripts.train_vq2_lc111_phase8_9_full_residual_endpoint as lc111


def test_lc111_dense_full_head_contract() -> None:
    assert lc111.PHASES == (8, 9)
    assert lc111.OPTIMIZER_STEPS == 512
    assert lc111.BATCH_SIZE == 4_096
    assert lc111.MINIMUM_PHASE_IMPROVEMENT == 1.50
    assert lc111.MAXIMUM_PHASE_DELTA_L2 == 64.0


def test_lc111_residual_action_shapes_and_delta() -> None:
    hidden = torch.zeros((3, 5))
    base = torch.zeros((3, 4))
    parameters = (
        torch.zeros((2, 5)), torch.zeros(2),
        torch.zeros((4, 2)), torch.zeros(4),
    )
    action = lc111.residual_action(hidden, base, parameters)
    assert action.shape == (3, 4)
    assert torch.equal(action, torch.zeros_like(action))
    shifted = tuple(value + 1.0 for value in parameters)
    assert float(lc111.parameter_delta_l2(shifted, parameters)) > 0.0


def test_lc111_dataset_report_is_source_locked_without_large_feature_read() -> None:
    import json
    from scripts.eval_vq2_variable_gate_oracle import sha256_path

    assert sha256_path(lc111.DATASET_REPORT) == lc111.DATASET_REPORT_SHA256
    report = json.loads(lc111.DATASET_REPORT.read_text())
    assert report["training_dataset_admitted"]
    assert report["feature_phase_records"][8] == 62_896
    assert report["feature_phase_records"][9] == 58_288
