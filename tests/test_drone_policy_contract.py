from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "drone_policy_contract.py"
spec = importlib.util.spec_from_file_location("drone_policy_contract", MODULE_PATH)
contract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)


def test_contract_sizes_and_field_names():
    assert contract.OBSERVATION_SIZE == 23
    assert contract.ACTION_SIZE == 4
    assert contract.OBSERVATION_FIELDS[10:18] == (
        "gate_visible",
        "gate_forward_norm",
        "gate_right_norm",
        "gate_down_norm",
        "gate_yaw_error_norm",
        "gate_pitch_error_norm",
        "gate_apparent_size_norm",
        "gate_pose_confidence",
    )
    assert contract.ACTION_FIELDS == (
        "cmd_forward_norm",
        "cmd_right_norm",
        "cmd_down_norm",
        "cmd_yaw_rate_norm",
    )


def test_decode_policy_action_scales_and_rotates_body_to_local_ned():
    action, setpoint = contract.decode_policy_action(
        [2.0, -0.5, 0.25, -2.0],
        yaw_rad=math.pi / 2.0,
    )

    assert action.normalized == (1.0, -0.5, 0.25, -1.0)
    assert action.forward_m_s == pytest.approx(2.0)
    assert action.right_m_s == pytest.approx(-0.5)
    assert action.down_m_s == pytest.approx(0.2)
    assert action.yaw_rate_rad_s == pytest.approx(-1.0)
    assert setpoint.vx_m_s == pytest.approx(0.5)
    assert setpoint.vy_m_s == pytest.approx(2.0)
    assert setpoint.vz_m_s == pytest.approx(0.2)
    assert setpoint.yaw_rate_rad_s == pytest.approx(-1.0)


def test_validate_observation_rejects_wrong_shape_and_bounds():
    obs = [0.0] * contract.OBSERVATION_SIZE
    assert contract.validate_observation(obs) == tuple(obs)

    with pytest.raises(ValueError, match="expected 23"):
        contract.validate_observation(obs[:-1])

    obs[3] = 1.1
    with pytest.raises(ValueError, match="gyro_roll_rate_norm"):
        contract.validate_observation(obs)
