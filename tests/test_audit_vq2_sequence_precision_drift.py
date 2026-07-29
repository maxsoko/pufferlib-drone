import torch
import pytest

from scripts.audit_vq2_sequence_precision_drift import _drift


def test_drift_reports_rmse_and_max():
    parent = {"x": torch.zeros(4)}
    child = {"x": torch.tensor([0.0, 0.0, 1.0, -1.0])}
    report = _drift(parent, child)["x"]
    assert report["rmse"] == pytest.approx(2**-0.5)
    assert report["max_abs"] == 1.0
