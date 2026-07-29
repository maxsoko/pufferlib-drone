import json

import numpy as np
import pytest
import torch

from scripts.eval_vq2_gate2_prefixed_checkpoint import (
    _overrides,
    _flat_vec_log,
    advance_gate_kinematic_state,
    advance_forward_gate_rate_world_x,
    apply_gate_kinematic_state,
    assisted_dataset_target,
    gate2_first_observation,
    initialize_gate_kinematic_state,
    linear_gate_features,
    load_policy_prefix,
    reseed_gate_position_world_from_body_bearing,
    recover_teacher_action,
    scale_gate_observations,
    scheduled_policy_actions,
    world_frame_gate_features,
)


def test_load_policy_prefix_rejects_non_gate1_samples(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"policy_trace": {"samples": [{
        "observation": [0.0] * 32,
        "normalized_action": [0.0] * 4,
        "official_active_gate_index": 1,
    }]}}))
    try:
        load_policy_prefix(report)
    except ValueError as exc:
        assert "extends beyond Gate 1" in str(exc)
    else:
        raise AssertionError("non-Gate-1 prefix was accepted")


def test_load_policy_prefix_can_truncate_continued_live_attempt(tmp_path):
    report = tmp_path / "report.json"
    samples = []
    for gate in (0, 0, 1):
        samples.append({
            "observation": [float(gate)] * 32,
            "normalized_action": [float(gate)] * 4,
            "official_active_gate_index": gate,
        })
    report.write_text(json.dumps({"policy_trace": {"samples": samples}}))

    observations, actions = load_policy_prefix(
        report, stop_at_gate_advance=True)

    assert observations.shape == (2, 32)
    assert actions.shape == (2, 4)
    np.testing.assert_array_equal(observations, np.zeros((2, 32)))


def test_gate2_first_observation_restores_last_official_action():
    observations = np.zeros((3, 32), dtype=np.float32)
    action = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    patched = gate2_first_observation(observations, action)

    np.testing.assert_allclose(patched[:, 19:23], np.tile(action, (3, 1)))
    np.testing.assert_array_equal(observations, np.zeros((3, 32), dtype=np.float32))


def test_gate2_first_observation_can_carry_legal_prefix_gate_rates():
    observations = np.zeros((2, 32), dtype=np.float32)
    rates = np.asarray([-0.6, 0.1, -0.2], dtype=np.float32)

    patched = gate2_first_observation(
        observations, np.zeros(4, dtype=np.float32), rates)

    np.testing.assert_array_equal(patched[:, 0:3], np.tile(rates, (2, 1)))


def test_gate_observation_scale_preserves_bearing_and_updates_apparent_size():
    observations = np.zeros((2, 32), dtype=np.float32)
    observations[0, 10] = 1.0
    observations[0, 0:3] = np.tanh([1.0 / 5.0, 2.0 / 3.0, -1.0 / 3.0])
    observations[0, 11:14] = np.tanh([10.0 / 10.0, 5.0 / 5.0, -2.0 / 5.0])
    observations[0, 16] = 0.1

    scaled = scale_gate_observations(observations, 0.5)

    np.testing.assert_allclose(
        np.arctanh(scaled[0, 11:14]) * [10.0, 5.0, 5.0],
        [5.0, 2.5, -1.0],
        atol=2e-6,
    )
    assert scaled[0, 16] == np.float32(0.2)
    np.testing.assert_array_equal(scaled[1], observations[1])


def test_gate_observation_scale_supports_per_agent_curriculum():
    observations = np.zeros((2, 32), dtype=np.float32)
    observations[:, 10] = 1.0
    observations[:, 11] = np.tanh(1.0)

    scaled = scale_gate_observations(
        observations, np.asarray([0.5, 1.5], dtype=np.float32))

    forward = np.arctanh(scaled[:, 11]) * 10.0
    np.testing.assert_allclose(forward, [5.0, 15.0], atol=2e-6)


def test_world_frame_gate_features_expose_world_up_with_identity_attitude():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = 1.0
    observations[0, 10] = 1.0
    observations[0, 0:3] = np.tanh([2.0 / 5.0, 1.0 / 3.0, -0.6 / 3.0])
    observations[0, 11:14] = np.tanh([10.0 / 10.0, 2.0 / 5.0, -3.0 / 5.0])

    transformed = world_frame_gate_features(observations)

    assert np.arctanh(transformed[0, 0]) * 5.0 == pytest.approx(2.0)
    assert np.arctanh(transformed[0, 1]) * 3.0 == pytest.approx(1.0)
    assert np.arctanh(transformed[0, 2]) * 3.0 == pytest.approx(0.6)
    assert np.arctanh(transformed[0, 12]) * 5.0 == pytest.approx(2.0)
    assert np.arctanh(transformed[0, 13]) * 5.0 == pytest.approx(3.0)


def test_linear_gate_features_decode_public_tanh_units_and_clip():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 10] = 1.0
    observations[0, [0, 1, 2, 11, 12, 13]] = np.tanh(
        [-0.8, 0.25, 1.5, 0.4, -1.2, 0.6])

    transformed = linear_gate_features(observations)

    np.testing.assert_allclose(
        transformed[0, [0, 1, 2, 11, 12, 13]],
        [-0.8, 0.25, 1.0, 0.4, -1.0, 0.6],
        atol=2e-6,
    )


def test_forward_gate_rate_predictor_uses_public_attitude_and_prior_thrust():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = np.cos(0.05)
    observations[0, 8] = np.sin(0.05)

    predicted = advance_forward_gate_rate_world_x(
        np.asarray([-4.0], dtype=np.float32), observations)

    # Positive pitch points thrust against world-X velocity, while drag also
    # reduces speed; the stationary gate therefore approaches less quickly.
    assert -4.0 < predicted[0] < -3.9


def test_gate_kinematic_predictor_round_trips_public_pose_and_rate():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = np.cos(0.07)
    observations[0, 8] = np.sin(0.07)
    observations[0, 10] = 1.0
    observations[0, 11:14] = np.tanh([0.8, -0.3, 0.2])
    initial_rate = np.asarray([[-4.2, -0.03, -0.35]], dtype=np.float32)

    position, rate = initialize_gate_kinematic_state(
        observations, initial_rate)
    encoded = apply_gate_kinematic_state(observations, position, rate)

    np.testing.assert_allclose(encoded[:, 11:14], observations[:, 11:14], atol=2e-7)
    assert np.isfinite(encoded[:, 0:3]).all()


def test_gate_kinematic_predictor_calibrates_only_initial_world_position():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = 1.0
    observations[0, 11:14] = np.tanh([0.5, 0.4, -0.2])
    rate = np.asarray([[-4.0, 0.0, 0.0]], dtype=np.float32)

    position, returned_rate = initialize_gate_kinematic_state(
        observations,
        rate,
        np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(position, [[10.0, 6.0, 4.0]], atol=2e-6)
    np.testing.assert_array_equal(returned_rate, rate)


def test_gate_kinematic_predictor_can_use_legal_fixed_initial_world_position():
    observations = np.zeros((2, 32), dtype=np.float32)
    observations[:, 6] = 1.0
    rate = np.asarray([[-4.0, 0.0, 0.0], [-4.1, 0.1, -0.2]], dtype=np.float32)
    fixed = np.asarray([[14.0, 8.0, 2.0], [14.1, 8.1, 2.1]], dtype=np.float32)

    position, returned_rate = initialize_gate_kinematic_state(
        observations,
        rate,
        np.full((2, 3), 99.0, dtype=np.float32),
        fixed,
    )

    np.testing.assert_array_equal(position, fixed)
    np.testing.assert_array_equal(returned_rate, rate)


def test_gate_kinematic_predictor_advances_stationary_gate_from_public_controls():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = 1.0
    observations[0, 21] = 0.0
    position = np.asarray([[10.0, 2.0, 1.0]], dtype=np.float32)
    rate = np.asarray([[-4.0, -0.1, -0.2]], dtype=np.float32)

    next_position, next_rate = advance_gate_kinematic_state(
        position, rate, observations)

    assert next_position[0, 0] < position[0, 0]
    assert next_rate[0, 0] > rate[0, 0]
    assert next_rate[0, 2] < rate[0, 2]
    assert np.isfinite(next_position).all()
    assert np.isfinite(next_rate).all()


def test_gate_kinematic_bearing_reseed_preserves_predicted_forward_range():
    observations = np.zeros((1, 32), dtype=np.float32)
    observations[0, 6] = 1.0
    observations[0, 10] = 1.0
    observations[0, 11:14] = np.tanh([20.0 / 10.0, 2.0 / 5.0, 4.0 / 5.0])
    predicted = np.asarray([[10.0, 5.0, 2.0]], dtype=np.float32)

    corrected = reseed_gate_position_world_from_body_bearing(
        predicted, observations)

    np.testing.assert_allclose(corrected, [[10.0, 1.0, -2.0]], atol=2e-5)


def test_flat_vec_log_adds_environment_namespace():
    class FakePufferl:
        @staticmethod
        def unroll_nested_dict(log):
            return [("n", 3.0), ("already/namespaced", 4.0)]

    assert _flat_vec_log(FakePufferl, {}) == {
        "env/n": 3.0,
        "already/namespaced": 4.0,
    }


def test_recover_teacher_action_inverts_training_blend():
    policy = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
    teacher = np.asarray([-0.6, 0.7, -0.8, 0.9], dtype=np.float32)
    blend = 0.05
    executed = (1.0 - blend) * policy + blend * teacher

    np.testing.assert_allclose(
        recover_teacher_action(policy, executed, blend), teacher, atol=2e-6)


def test_assisted_dataset_target_can_retain_public_executed_action():
    policy = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
    executed = np.asarray([-0.1, 0.2, -0.3, 0.4], dtype=np.float32)

    np.testing.assert_array_equal(
        assisted_dataset_target(
            policy, executed, 0.8, target="executed"),
        executed,
    )


def test_scheduled_policy_actions_selects_whole_puffer_vector():
    primary = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    late = torch.tensor([[-0.4, -0.3, -0.2, -0.1]])

    selected, using_late = scheduled_policy_actions(
        primary, late, gate2_elapsed_seconds=1.99, switch_after_seconds=2.0)
    assert selected is primary
    assert not using_late

    selected, using_late = scheduled_policy_actions(
        primary, late, gate2_elapsed_seconds=2.0, switch_after_seconds=2.0)
    assert selected is late
    assert using_late

    selected, using_late = scheduled_policy_actions(
        primary,
        late,
        gate2_elapsed_seconds=3.0,
        switch_after_seconds=2.0,
        switch_until_seconds=3.0,
    )
    assert selected is primary
    assert not using_late


def test_scheduled_policy_actions_does_not_require_a_late_policy():
    primary = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    selected, using_late = scheduled_policy_actions(
        primary, None, gate2_elapsed_seconds=100.0, switch_after_seconds=None)
    assert selected is primary
    assert not using_late


def test_perturbation_scale_applies_to_start_gate_and_plant_jitter():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=True,
        perturbation_scale=0.25,
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.start-vx-jitter"]) == 0.05
    assert float(options["--env.gate-position-jitter-y"]) == 0.10
    assert float(options["--env.sitl-rate-gain-jitter-frac"]) == 0.0125
    assert float(options["--env.reset-position-noise-xy"]) == 0.0


def test_perturbation_components_can_isolate_gate_jitter():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=True,
        perturbation_scale=0.25,
        perturbation_components=("gate",),
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.start-vx-jitter"]) == 0.0
    assert float(options["--env.gate-position-jitter-y"]) == 0.10
    assert int(options["--env.gate-position-domain-randomize"]) == 1
    assert int(options["--env.sitl-plant-domain-randomize"]) == 0


def test_start_perturbation_dimensions_can_isolate_velocity_jitter():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=True,
        perturbation_scale=0.25,
        perturbation_components=("start",),
        start_perturbation_dimensions=("velocity",),
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.start-vx-jitter"]) == 0.05
    assert float(options["--env.start-vy-jitter"]) == 0.025
    assert float(options["--env.start-elapsed-time-jitter"]) == 0.0
    assert float(options["--env.start-x-jitter"]) == 0.0
    assert float(options["--env.start-roll-jitter-rad"]) == 0.0
    assert float(options["--env.start-wx-jitter"]) == 0.0


def test_gate_motion_predictor_overrides_are_explicit_and_default_off():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=False,
        perturbation_scale=1.0,
        predict_gate_motion_dropout=True,
        gate_motion_control_accel_gain=0.75,
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert int(options["--env.sitl-gate-motion-predict-dropout"]) == 1
    assert float(options["--env.sitl-gate-motion-control-accel-gain"]) == 0.75


def test_live_matched_elapsed_clock_overrides_are_explicit():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=False,
        perturbation_scale=1.0,
        start_elapsed_time=3.281,
        time_limit_seconds=14.0,
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.start-elapsed-time"]) == 3.281
    assert float(options["--env.time-limit-seconds"]) == 14.0


def test_official_altitude_envelope_overrides_are_explicit():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=0.0,
        perturbed=False,
        perturbation_scale=1.0,
        crash_height=-4.5,
        safety_altitude=-3.5,
        plant_lateral_accel_scale=0.75,
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.crash-height"]) == -4.5
    assert float(options["--env.safety-altitude"]) == -3.5
    assert float(options["--env.sitl-lateral-accel-scale"]) == 0.75


def test_training_teacher_thrust_overrides_are_explicit():
    from argparse import Namespace

    values = _overrides(Namespace(
        seed=1,
        episodes=8,
        gate_radius=0.75,
        training_teacher_blend=1.0,
        teacher_thrust_bias=0.04,
        teacher_thrust_per_m=0.25,
        teacher_thrust_rate_per_m_s=0.08,
        perturbed=False,
        perturbation_scale=1.0,
    ))
    options = dict(zip(values[::2], values[1::2]))

    assert float(options["--env.teacher-thrust-bias"]) == 0.04
    assert float(options["--env.teacher-thrust-per-m"]) == 0.25
    assert float(options["--env.teacher-thrust-rate-per-m-s"]) == 0.08
