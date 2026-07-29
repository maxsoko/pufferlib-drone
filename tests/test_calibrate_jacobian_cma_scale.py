import json

import numpy as np
import pytest

from scripts.calibrate_jacobian_cma_scale import calibrate_scale


def test_calibration_uses_strictest_pre_outcome_constraint(tmp_path, monkeypatch):
    basis_path = tmp_path / "basis.npz"
    np.savez_compressed(
        basis_path,
        basis=np.eye(2, dtype=np.float32),
        metadata=np.asarray(json.dumps({"coefficient_scale": 2.0})),
    )

    monkeypatch.setattr(
        "scripts.calibrate_jacobian_cma_scale.materialize_candidate",
        lambda parent, basis, coefficient, output: {
            "path": str(output), "coefficients": coefficient.tolist()
        },
    )
    calls = {"value": 0}

    def fake_drift(parent, candidate, trace, agents):
        calls["value"] += 1
        if "failure" in str(trace):
            return {"focus_rms": 0.002, "full_rms": 0.002, "focus_max": 0.004, "full_max": 0.004}
        return {"focus_rms": 0.004, "full_rms": 0.003, "focus_max": 0.02, "full_max": 0.01}

    monkeypatch.setattr(
        "scripts.calibrate_jacobian_cma_scale.trace_action_drift", fake_drift
    )
    report = calibrate_scale(
        tmp_path / "parent.bin",
        basis_path,
        tmp_path / "failure.npz",
        tmp_path / "anchor.npz",
        tmp_path / "out",
        dimension=2,
        population_size=4,
        parent_count=2,
        maximum_anchor_rms=0.001,
        maximum_anchor_max=0.01,
    )
    calibration = report["calibration"]
    assert calibration["limiting_constraint"] == "anchor_rms_limit"
    assert calibration["scale_factor"] == pytest.approx(0.25)
    updated = np.load(basis_path, allow_pickle=False)
    assert json.loads(str(updated["metadata"]))["coefficient_scale"] == pytest.approx(0.5)
    assert calls["value"] == 8
