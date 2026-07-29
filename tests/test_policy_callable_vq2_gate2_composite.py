import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_vq2_gate2_prefixed_checkpoint import (
    advance_gate_kinematic_state,
    apply_gate_kinematic_state,
    initialize_gate_kinematic_state,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "policy_callable_vq2_gate2_composite",
    SCRIPTS / "policy_callable_vq2_gate2_composite.py",
)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


class FakePolicy:
    def __init__(self, action):
        self.action = list(action)
        self.observations = []
        self.resets = 0
        self.state = np.zeros((1, 2), dtype=np.float32)

    def infer(self, observation):
        self.observations.append(list(observation))
        return list(self.action)

    def reset_state(self):
        self.resets += 1


@pytest.fixture
def fake_state(monkeypatch):
    state = policy._CompositeState(
        prefix=FakePolicy([0.1, 0.2, 0.3, 0.4]),
        gate2=FakePolicy([-0.1, -0.2, -0.3, -0.4]),
    )
    monkeypatch.setattr(policy, "_resolve", lambda: state)
    return state


def observation(gate):
    values = np.zeros(32, dtype=np.float32)
    values[23] = gate / 6.0
    if gate < 6:
        values[24 + gate] = 1.0
    return values


def test_both_puffer_policies_advance_before_whole_vector_selection(fake_state):
    gate1 = policy.infer(observation(0))
    gate2 = policy.infer(observation(1))

    assert gate1 == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert gate2 == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert len(fake_state.prefix.observations) == 2
    assert len(fake_state.gate2.observations) == 2


def test_reset_resets_both_recurrent_puffer_states(fake_state):
    policy.reset()

    assert fake_state.prefix.resets == 1
    assert fake_state.gate2.resets == 1


def test_fixed_public_prefix_holds_gate2_state_and_normalizes_its_clock(monkeypatch):
    state = policy._CompositeState(
        prefix=FakePolicy([0.1, 0.2, 0.3, 0.4]),
        gate2=FakePolicy([-0.1, -0.2, -0.3, -0.4]),
        gate2_warm_state=np.ones((1, 2), dtype=np.float32),
        runtime_duration_s=14.0,
        gate2_time_limit_s=20.0,
    )
    monkeypatch.setattr(policy, "_resolve", lambda: state)

    first = observation(0)
    first[18] = 0.2
    second = observation(1)
    second[18] = 0.25
    assert policy.infer(first) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert policy.infer(second) == pytest.approx([-0.1, -0.2, -0.3, -0.4])

    assert len(state.prefix.observations) == 2
    assert len(state.gate2.observations) == 1
    assert state.gate2.observations[0][18] == pytest.approx(0.175)


def test_reset_restores_fixed_gate2_warm_state(monkeypatch):
    gate2 = FakePolicy([-0.1, -0.2, -0.3, -0.4])
    warm = np.full((1, 2), 3.0, dtype=np.float32)
    state = policy._CompositeState(
        prefix=FakePolicy([0.1, 0.2, 0.3, 0.4]),
        gate2=gate2,
        gate2_warm_state=warm,
    )
    monkeypatch.setattr(policy, "_resolve", lambda: state)
    gate2.state.fill(9.0)

    policy.reset()

    np.testing.assert_array_equal(gate2.state, warm)
    assert gate2.resets == 0


def test_fixed_prefix_loader_accepts_only_gate1_public_observations(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(
        '{"policy_trace":{"samples":['
        '{"official_active_gate_index":0,"observation":['
        + ",".join("0" for _ in range(32))
        + ']},{"official_active_gate_index":1,"observation":['
        + ",".join("1" for _ in range(32))
        + "]}]}}",
        encoding="utf-8",
    )

    observations = policy._gate0_prefix_observations(str(report))

    assert len(observations) == 1
    assert observations[0] == [0.0] * 32


def test_gate2_observation_can_expose_world_vertical_features():
    values = observation(1)
    values[6] = 1.0
    values[10] = 1.0
    values[0:3] = np.tanh([2.0 / 5.0, 1.0 / 3.0, -0.6 / 3.0])
    values[11:14] = np.tanh([10.0 / 10.0, 2.0 / 5.0, -3.0 / 5.0])

    transformed = policy._gate2_observation(
        values,
        runtime_duration_s=0.0,
        gate2_time_limit_s=0.0,
        world_frame_gate_features=True,
    )

    assert np.arctanh(transformed[0]) * 5.0 == pytest.approx(2.0)
    assert np.arctanh(transformed[1]) * 3.0 == pytest.approx(1.0)
    assert np.arctanh(transformed[2]) * 3.0 == pytest.approx(0.6)
    assert np.arctanh(transformed[12]) * 5.0 == pytest.approx(2.0)
    assert np.arctanh(transformed[13]) * 5.0 == pytest.approx(3.0)


def test_gate2_observation_can_linearize_public_gate_features():
    values = observation(1)
    values[10] = 1.0
    values[0:3] = np.tanh([-0.8, 0.25, 1.5])
    values[11:14] = np.tanh([0.4, -1.2, 0.6])

    transformed = policy._gate2_observation(
        values,
        runtime_duration_s=0.0,
        gate2_time_limit_s=0.0,
        linear_gate_features=True,
    )
    np.testing.assert_allclose(
        np.asarray(transformed)[[0, 1, 2, 11, 12, 13]],
        [-0.8, 0.25, 1.0, 0.4, -1.0, 0.6],
        atol=2e-6,
    )


def test_scalar_forward_predictor_matches_calibrated_public_contract():
    values = observation(1)
    values[6] = np.cos(0.05)
    values[8] = np.sin(0.05)

    predicted = policy._advance_forward_gate_rate_world_x(
        -4.0, values, 1.0 / 60.0)

    assert -4.0 < predicted < -3.9


def test_runtime_gate_kinematic_predictor_matches_offline_vector_path():
    values = observation(1)
    values[6] = np.cos(0.07)
    values[8] = np.sin(0.07)
    values[10] = 1.0
    values[11:14] = np.tanh([0.8, -0.3, 0.2])
    initial_rate = (-4.2, -0.03, -0.35)
    initial_position_scale = (1.2, 0.8, 1.5)
    state = policy._CompositeState(
        prefix=FakePolicy([0.0] * 4),
        gate2=FakePolicy([0.0] * 4),
        predict_gate_kinematics=True,
        initial_gate_rate_world=initial_rate,
        initial_gate_position_world_scale=initial_position_scale,
        gate_rate_world=initial_rate,
    )

    runtime_first = np.asarray(
        policy._apply_gate_kinematic_predictor(state, values), dtype=np.float32)
    offline_position, offline_rate = initialize_gate_kinematic_state(
        values[None, :],
        np.asarray([initial_rate], dtype=np.float32),
        np.asarray([initial_position_scale], dtype=np.float32),
    )
    offline_first = apply_gate_kinematic_state(
        values[None, :], offline_position, offline_rate)[0]
    np.testing.assert_allclose(runtime_first, offline_first, atol=3e-7)

    next_values = values.copy()
    next_values[6] = np.cos(0.06)
    next_values[8] = np.sin(0.06)
    next_values[21] = 0.25
    runtime_next = np.asarray(
        policy._apply_gate_kinematic_predictor(state, next_values), dtype=np.float32)
    offline_position, offline_rate = advance_gate_kinematic_state(
        offline_position, offline_rate, next_values[None, :])
    offline_next = apply_gate_kinematic_state(
        next_values[None, :], offline_position, offline_rate)[0]
    np.testing.assert_allclose(runtime_next, offline_next, atol=5e-7)


def test_gate2_selection_waits_for_new_public_gate_association(monkeypatch):
    gate2 = FakePolicy([-0.1, -0.2, -0.3, -0.4])
    state = policy._CompositeState(
        prefix=FakePolicy([0.1, 0.2, 0.3, 0.4]),
        gate2=gate2,
        gate2_warm_state=np.ones((1, 2), dtype=np.float32),
        association_jump_threshold_m=3.0,
        reseed_bearing_on_association=True,
        association_ready=False,
    )
    monkeypatch.setattr(policy, "_resolve", lambda: state)
    gate1 = observation(0)
    gate1[10] = 1.0
    gate1[11:14] = np.tanh([1.0, 1.0, -0.2])
    stale_gate1 = gate1.copy()
    stale_gate1[23] = 1.0 / 6.0
    stale_gate1[24:30] = 0.0
    stale_gate1[25] = 1.0
    new_gate2 = stale_gate1.copy()
    new_gate2[11:14] = np.tanh([0.5, 0.2, -0.05])

    assert policy.infer(gate1) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert policy.infer(stale_gate1) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert policy.infer(new_gate2) == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert state.association_ready
    assert state.association_reseed_pending
    assert len(gate2.observations) == 2


def test_runtime_predictor_reseeds_new_association_bearing_only():
    values = observation(1)
    values[6] = 1.0
    values[10] = 1.0
    values[11:14] = np.tanh([20.0 / 10.0, 2.0 / 5.0, 4.0 / 5.0])
    state = policy._CompositeState(
        prefix=FakePolicy([0.0] * 4),
        gate2=FakePolicy([0.0] * 4),
        initial_gate_position_world=(10.0, 5.0, 2.0),
        association_reseed_pending=True,
    )

    policy._apply_gate_kinematic_predictor(state, values)

    np.testing.assert_allclose(
        state.gate_position_world, [10.0, 1.0, -2.0], atol=2e-5)
    assert not state.association_reseed_pending


def test_runtime_predictor_prefers_fixed_legal_initial_world_position():
    values = observation(1)
    values[6] = 1.0
    values[10] = 1.0
    values[11:14] = np.tanh([0.1, 0.1, 0.1])
    fixed = (14.67, 8.47, 2.40)
    state = policy._CompositeState(
        prefix=FakePolicy([0.0] * 4),
        gate2=FakePolicy([0.0] * 4),
        initial_gate_position_world=fixed,
    )

    policy._apply_gate_kinematic_predictor(state, values)

    assert state.gate_position_world == fixed


def test_progress_must_be_quantized(fake_state):
    values = observation(1)
    values[23] = 0.2
    with pytest.raises(ValueError, match="invalid quantized race progress"):
        policy.infer(values)
