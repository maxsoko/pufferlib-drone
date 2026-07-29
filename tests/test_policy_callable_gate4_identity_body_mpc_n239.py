from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_gate4_identity_body_mpc as evaluate_n239
import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_identity_body_mpc_n239 as n239


MODEL = ROOT / "logs/sitl/n239_gate4_identity_body_dynamics_fit_20260719.json"
N238_TRACES = sorted(
    (ROOT / "logs/sitl").glob(
        "competition_smoke_policy_competitive_n238_identity_lock_batch_007_"
        "il_run_*_attempt_001.json"
    )
)


def _observation(gate: int, pose=None, *, yaw_error_rad: float | None = None):
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    if pose is not None:
        values[10] = 1.0
        values[11] = math.tanh(pose[0] / 10.0)
        values[12] = math.tanh(pose[1] / 5.0)
        values[13] = math.tanh(pose[2] / 5.0)
        error = (
            math.atan2(pose[1], pose[0])
            if yaw_error_rad is None
            else yaw_error_rad
        )
        values[14] = error / (math.pi / 4.0)
    return values


def _controller() -> n239.Gate4IdentityBodyMPCController:
    return n239.Gate4IdentityBodyMPCController(
        body_mpc.LateralDynamicsEnsemble.load(MODEL)
    )


def _feed(controller, poses, *, start_s: float):
    output = None
    for index, pose in enumerate(poses):
        output = controller.apply(
            _observation(3, pose), [0.0] * 4, now_s=start_s + 0.1 * index
        )
    return output


def test_model_loads_only_the_fold_consistent_lateral_axis() -> None:
    ensemble = body_mpc.LateralDynamicsEnsemble.load(MODEL)
    assert ensemble.members == 6
    assert np.all(ensemble.roll_gain_m_s2 > 3.0)
    assert np.all(ensemble.roll_gain_m_s2 < 4.0)
    assert ensemble.source_sha256 == n239.EXPECTED_MODEL_SHA256


def test_model_loader_rejects_artifact_without_deployable_lateral_axis(
    tmp_path: Path,
) -> None:
    payload = json.loads(MODEL.read_text(encoding="utf-8"))
    payload["deployable_action_axes"] = []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not deployable"):
        body_mpc.LateralDynamicsEnsemble.load(bad)


def test_policy_model_hash_is_checked_once_per_controller_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(n239.MODEL_ENVIRONMENT_VARIABLE, str(MODEL))
    n239._CONTROLLER = None
    n239._CONTROLLER_MODEL_PATH = None
    calls = 0
    original = n239._sha256

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(n239, "_sha256", counted)
    try:
        controller = n239._controller()
        assert all(n239._controller() is controller for _ in range(100))
        assert calls == 1
    finally:
        n239._CONTROLLER = None
        n239._CONTROLLER_MODEL_PATH = None


def test_optimizer_is_deterministic_bounded_and_finite() -> None:
    optimizer = body_mpc.Gate4IdentityLateralMPC(
        body_mpc.LateralDynamicsEnsemble.load(MODEL)
    )
    kwargs = {
        "forward_m": 14.0,
        "forward_rate_m_s": -7.0,
        "right_m": -5.4,
        "right_rate_m_s": 0.0,
    }
    first = optimizer.plan(**kwargs)
    optimizer.reset()
    second = optimizer.plan(**kwargs)
    assert first == second
    assert -1.0 <= first.roll_norm <= 0.85
    assert math.isfinite(first.objective)
    assert math.isfinite(first.worst_abs_terminal_right_m)
    assert len(first.terminal_right_m) == 6


def test_far_associated_family_is_quarantined_without_action_authority() -> None:
    controller = _controller()
    output = _feed(
        controller,
        ((35.0, 9.0, 4.0), (34.0, 8.8, 3.8), (33.0, 8.5, 3.6)),
        start_s=1.0,
    )
    assert output[:3] == [n239.n238.BRAKE_PITCH_NORM, 0.0, 0.0]
    assert not controller.trusted_near_family
    assert controller.optimized_lateral_plans == 0
    assert controller.quarantined_associated_calls > 0


def test_near_closer_reacquisition_unlocks_mpc_and_rejected_alias_is_inert() -> None:
    controller = _controller()
    _feed(
        controller,
        ((35.0, 9.0, 4.0), (34.0, 8.8, 3.8), (33.0, 8.5, 3.6)),
        start_s=1.0,
    )
    trusted = _feed(
        controller,
        ((15.5, -5.8, -1.5), (14.8, -5.5, -1.3), (14.0, -5.2, -1.1)),
        start_s=1.4,
    )
    assert controller.trusted_near_family
    assert controller.trusted_family_triggers == 1
    assert controller.optimized_lateral_plans == 1
    assert trusted[1] != 0.0
    alias = controller.apply(
        _observation(3, (34.0, 12.0, 9.0), yaw_error_rad=0.7),
        [1.0, 1.0, 1.0, 1.0],
        now_s=1.65,
    )
    assert alias == trusted
    assert controller.optimized_lateral_plans == 1


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = _controller()
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent


def test_counterfactual_selects_only_near_reacquired_live_entries() -> None:
    assert len(N238_TRACES) == 3
    entries = [evaluate_n239._entry_cases(path) for path in N238_TRACES]
    assert [len(rows) for rows in entries] == [1, 1, 1]
    assert all(
        row[0]["forward_m"]
        <= evaluate_n239.TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M
        for row in entries
    )
