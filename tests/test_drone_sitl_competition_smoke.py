import importlib.util
import hashlib
import math
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "drone_sitl_competition_smoke.py"
SPEC = importlib.util.spec_from_file_location("drone_sitl_competition_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)

ADAPTER_SPEC = importlib.util.spec_from_file_location("drone_sitl_adapter", SCRIPTS / "drone_sitl_adapter.py")
sitl_adapter = importlib.util.module_from_spec(ADAPTER_SPEC)
sys.modules[ADAPTER_SPEC.name] = sitl_adapter
ADAPTER_SPEC.loader.exec_module(sitl_adapter)


def test_deployment_manifest_hashes_all_composite_checkpoints(monkeypatch, tmp_path):
    checkpoints = {}
    for gate in (4, 5, 6):
        path = tmp_path / f"gate{gate}.bin"
        path.write_bytes(f"gate-{gate}".encode())
        checkpoints[gate] = path
        monkeypatch.setenv(
            f"PUFFER_POLICY_GATE{gate}_CHECKPOINT_PATH", str(path)
        )
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(checkpoints[4]))

    manifest = smoke.build_deployment_manifest(None)

    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoints[4].read_bytes()
    ).hexdigest()
    for gate in (4, 5, 6):
        assert manifest[f"gate{gate}_checkpoint"]["sha256"] == hashlib.sha256(
            checkpoints[gate].read_bytes()
        ).hexdigest()


def test_deployment_manifest_hashes_vq2_gate2_checkpoint(monkeypatch, tmp_path):
    prefix = tmp_path / "prefix.bin"
    gate2 = tmp_path / "gate2.bin"
    prefix.write_bytes(b"prefix")
    gate2.write_bytes(b"gate2")
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(prefix))
    monkeypatch.setenv("PUFFER_POLICY_GATE2_CHECKPOINT_PATH", str(gate2))

    manifest = smoke.build_deployment_manifest(None)

    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(
        prefix.read_bytes()).hexdigest()
    assert manifest["gate2_checkpoint"]["sha256"] == hashlib.sha256(
        gate2.read_bytes()).hexdigest()


def test_debug_frames_are_deferred_until_after_flight(monkeypatch, tmp_path):
    writes = []

    def fake_save(output_dir, frame, detection, label):
        writes.append((output_dir, frame.frame_id, detection, label))
        return {"label": label, "frame_id": frame.frame_id}

    monkeypatch.setattr(smoke, "save_debug_frame", fake_save)
    candidates = {}
    first = SimpleNamespace(frame_id=1)
    closest = SimpleNamespace(frame_id=2)
    latest = SimpleNamespace(frame_id=3)
    smoke.defer_debug_frame(candidates, frame=first, detection="first", range_m=12.0)
    smoke.defer_debug_frame(candidates, frame=closest, detection="closest", range_m=4.0)
    smoke.defer_debug_frame(candidates, frame=latest, detection="latest", range_m=7.0)

    assert writes == []
    saved = smoke.flush_debug_frames(tmp_path, candidates)

    assert [entry[3] for entry in writes] == [
        "first_detection",
        "closest_detection",
        "latest_detection",
    ]
    assert [saved[label]["frame_id"] for label in saved] == [1, 2, 3]


def test_build_policy_observation_matches_contract_size_and_bounds():
    telemetry = smoke.TelemetryState(
        roll=0.1,
        pitch=-0.2,
        yaw=0.3,
        rollspeed=0.5,
        pitchspeed=-0.4,
        yawspeed=0.25,
    )
    detection = smoke.GateDetection(
        corners=((250.0, 110.0), (390.0, 110.0), (390.0, 250.0), (250.0, 250.0)),
        area_px=19600.0,
        bounding_width_px=140.0,
        bounding_height_px=140.0,
        fill_ratio=0.9,
        confidence=0.8,
    )
    pose = smoke.estimate_gate_pose_from_corners(detection.corners)
    guidance = SimpleNamespace(
        center_x_norm=-0.25,
        heading_error_rad=0.1,
        confidence=0.8,
        supporting_rows=12,
        pixel_count=200,
    )
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=detection,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        guidance=guidance,
    )

    assert len(obs) == smoke.OBSERVATION_SIZE
    assert max(obs) <= 1.0
    assert min(obs) >= -1.0
    assert obs[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert obs[10] == pytest.approx(1.0)
    assert obs[12] == pytest.approx(0.0)
    assert obs[17] == pytest.approx(1.0)
    assert obs[17] != pytest.approx(pose.confidence)


def test_policy_observation_hides_far_unassociated_first_detection():
    telemetry = smoke.TelemetryState()
    pose = SimpleNamespace(
        body_vector_ned_m=(48.0, 2.0, 6.0),
        yaw_error_rad=0.0,
        range_camera_m=48.4,
        confidence=0.5,
    )
    motion = smoke.ObservableGateMotionFilter()
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.0,
        last_cmd_norm=(0.0, 0.0, 0.0, 0.0),
        observation_time_s=0.0,
        gate_motion_state=motion,
        gate_identity=0,
    )
    assert obs[10:18] == pytest.approx((0.0,) * 8)


def test_control_aware_predictor_can_start_after_proven_prefix():
    assert smoke.control_aware_gate_predictor_gain(
        enabled=True, official_gate_index=2, start_index=3
    ) == pytest.approx(0.0)
    assert smoke.control_aware_gate_predictor_gain(
        enabled=True, official_gate_index=3, start_index=3
    ) == pytest.approx(4.295)
    assert smoke.control_aware_gate_predictor_gain(
        enabled=False, official_gate_index=5, start_index=3
    ) == pytest.approx(0.0)


def test_gate_dropout_prediction_can_start_after_proven_prefix():
    assert not smoke.gate_dropout_prediction_enabled(
        enabled=True, official_gate_index=None, start_index=3
    )
    assert not smoke.gate_dropout_prediction_enabled(
        enabled=True, official_gate_index=2, start_index=3
    )
    assert smoke.gate_dropout_prediction_enabled(
        enabled=True, official_gate_index=3, start_index=3
    )
    assert smoke.gate_dropout_prediction_enabled(
        enabled=True, official_gate_index=None, start_index=0
    )
    assert not smoke.gate_dropout_prediction_enabled(
        enabled=False, official_gate_index=5, start_index=3
    )


def test_fixed_policy_state_clock_is_elapsed_time_deterministic():
    assert smoke.fixed_policy_steps_due(
        elapsed_s=0.0, state_hz=60.0, completed_steps=0
    ) == 1
    assert smoke.fixed_policy_steps_due(
        elapsed_s=(1.0 / 60.0) - 1e-6,
        state_hz=60.0,
        completed_steps=1,
    ) == 0
    assert smoke.fixed_policy_steps_due(
        elapsed_s=1.0 / 60.0,
        state_hz=60.0,
        completed_steps=1,
    ) == 1
    # A slow outer loop catches up every missing recurrent tick at once.
    assert smoke.fixed_policy_steps_due(
        elapsed_s=0.1, state_hz=60.0, completed_steps=2
    ) == 5
    # Repeating the same elapsed time after catch-up advances nothing.
    assert smoke.fixed_policy_steps_due(
        elapsed_s=0.1, state_hz=60.0, completed_steps=7
    ) == 0


def test_zero_policy_state_hz_preserves_legacy_one_step_per_update():
    assert smoke.fixed_policy_steps_due(
        elapsed_s=100.0, state_hz=0.0, completed_steps=9000
    ) == 1


def test_fixed_policy_state_clock_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="state_hz"):
        smoke.fixed_policy_steps_due(
            elapsed_s=0.0, state_hz=-1.0, completed_steps=0
        )
    with pytest.raises(ValueError, match="completed"):
        smoke.fixed_policy_steps_due(
            elapsed_s=0.0, state_hz=60.0, completed_steps=-1
        )


def test_catchup_observation_updates_only_previous_action_fields():
    observation = tuple(float(index) / 100.0 for index in range(32))
    patched = smoke.policy_observation_with_last_command(
        observation,
        (2.0, -2.0, 0.25, -0.5),
    )

    assert patched[:19] == observation[:19]
    assert patched[19:23] == pytest.approx((1.0, -1.0, 0.25, -0.5))
    assert patched[23:] == observation[23:]


def test_policy_stop_guard_requires_final_phase_and_forward_boundary():
    pose = SimpleNamespace(body_vector_ned_m=(17.9, 0.5, -0.25))

    assert smoke.should_stop_before_policy_gate(
        control_mode="policy-attitude",
        official_gate_index=3,
        pose=pose,
        stop_gate_index=3,
        stop_forward_m=18.0,
    )
    assert not smoke.should_stop_before_policy_gate(
        control_mode="policy-attitude",
        official_gate_index=2,
        pose=pose,
        stop_gate_index=3,
        stop_forward_m=18.0,
    )
    assert not smoke.should_stop_before_policy_gate(
        control_mode="course-fsm",
        official_gate_index=3,
        pose=pose,
        stop_gate_index=3,
        stop_forward_m=18.0,
    )
    assert not smoke.should_stop_before_policy_gate(
        control_mode="policy-attitude",
        official_gate_index=3,
        pose=SimpleNamespace(body_vector_ned_m=(18.1, 0.0, 0.0)),
        stop_gate_index=3,
        stop_forward_m=18.0,
    )


def test_official_finish_stop_requires_nonnegative_finish_time():
    assert not smoke.should_stop_after_official_finish(None)
    assert not smoke.should_stop_after_official_finish(
        SimpleNamespace(race_finish_time_ns=-1)
    )
    assert smoke.should_stop_after_official_finish(
        SimpleNamespace(race_finish_time_ns=12_345_678_901)
    )


def test_official_gate_index_stop_is_disabled_or_authoritative():
    assert not smoke.should_stop_after_official_gate_index(None, 4)
    assert not smoke.should_stop_after_official_gate_index(3, 4)
    assert not smoke.should_stop_after_official_gate_index(6, -1)
    assert smoke.should_stop_after_official_gate_index(4, 4)
    assert smoke.should_stop_after_official_gate_index(5, 4)


def test_official_gate_observation_epoch_invalidates_only_on_identity_change():
    epoch = smoke.OfficialGateObservationEpoch()

    assert not epoch.observe(None)
    assert not epoch.observe(0)
    assert not epoch.observe(0)
    assert epoch.observe(1)
    assert not epoch.observe(1)
    assert epoch.observe(2)


def test_gate3plus_official_transition_clears_cached_pose_and_motion_filter():
    source = MODULE_PATH.read_text(encoding="utf-8")
    transition_block = source.split(
        "official_gate_observation_changed = (", 1
    )[1].split("if (", 1)[0]

    scoped_clear_block = source.split(
        "if (\n                official_gate_observation_changed", 1
    )[1].split("if (", 1)[0]

    assert "official_gate_observation_epoch.observe(official_gate_index)" in transition_block
    assert "official_gate_index >= 2" in scoped_clear_block
    assert "gate_detection = None" in scoped_clear_block
    assert "gate_pose = None" in scoped_clear_block
    assert "control_gate_detection = None" in scoped_clear_block
    assert "control_gate_pose = None" in scoped_clear_block
    assert "last_control_detection_s = None" in scoped_clear_block
    assert "policy_gate_motion_state.reset()" in scoped_clear_block


def test_final_any_edge_priority_waits_for_configured_handoff_delay():
    kwargs = {
        "final_phase_edge_preference": True,
        "configured": True,
        "delay_s": 3.0,
    }
    assert not smoke.final_phase_any_edge_priority_enabled(
        **kwargs, phase_elapsed_s=2.999
    )
    assert smoke.final_phase_any_edge_priority_enabled(
        **kwargs, phase_elapsed_s=3.0
    )
    assert not smoke.final_phase_any_edge_priority_enabled(
        final_phase_edge_preference=False,
        configured=True,
        phase_elapsed_s=10.0,
        delay_s=3.0,
    )


def test_raw_gate_observation_interval_can_be_bounded_to_gate4():
    assert not smoke.raw_gate_observation_enabled(
        official_gate_index=2,
        start_index=3,
        end_index=3,
    )
    assert smoke.raw_gate_observation_enabled(
        official_gate_index=3,
        start_index=3,
        end_index=3,
    )
    assert not smoke.raw_gate_observation_enabled(
        official_gate_index=4,
        start_index=3,
        end_index=3,
    )
    assert smoke.raw_gate_observation_enabled(
        official_gate_index=5,
        start_index=3,
        end_index=-1,
    )


def test_post_prefix_edge_delay_restarts_for_each_official_gate():
    gate_index, started_s = smoke.update_post_prefix_edge_phase(
        enabled=True,
        official_gate_index=3,
        now_s=10.0,
        phase_gate_index=None,
        phase_started_s=None,
    )
    assert gate_index == 3
    assert started_s == pytest.approx(10.0)

    unchanged = smoke.update_post_prefix_edge_phase(
        enabled=True,
        official_gate_index=3,
        now_s=12.0,
        phase_gate_index=gate_index,
        phase_started_s=started_s,
    )
    assert unchanged == pytest.approx((3, 10.0))

    gate_index, started_s = smoke.update_post_prefix_edge_phase(
        enabled=True,
        official_gate_index=4,
        now_s=13.0,
        phase_gate_index=gate_index,
        phase_started_s=started_s,
    )
    assert gate_index == 4
    assert started_s == pytest.approx(13.0)

    assert smoke.update_post_prefix_edge_phase(
        enabled=False,
        official_gate_index=4,
        now_s=14.0,
        phase_gate_index=gate_index,
        phase_started_s=started_s,
    ) == (None, None)


def test_policy_observation_can_expose_official_race_phase_in_last_slot():
    telemetry = smoke.TelemetryState()
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=None,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        race_phase_gate_index=2,
        race_phase_denominator=3,
    )

    assert len(obs) == smoke.OBSERVATION_SIZE
    assert obs[-2] == pytest.approx(0.3)
    assert obs[-1] == pytest.approx(2.0 / 3.0)

    six_gate_phases = []
    for gate_index in range(7):
        phase_obs = smoke.build_policy_observation(
            telemetry,
            gate_detection=None,
            gate_pose=None,
            elapsed_fraction=0.25,
            last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
            race_phase_gate_index=gate_index,
            race_phase_denominator=6,
        )
        six_gate_phases.append(phase_obs[-1])
    assert six_gate_phases == pytest.approx([index / 6 for index in range(7)])


def test_policy_observation_appends_six_gate_progress_without_replacing_yaw():
    telemetry = smoke.TelemetryState()
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=None,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        gate_progress_adapter_gate_index=5,
        gate_progress_adapter_denominator=6,
    )

    assert len(obs) == 32
    assert obs[22] == pytest.approx(-0.4)
    assert obs[23] == pytest.approx(5.0 / 6.0)
    assert obs[24:] == pytest.approx((0.0,) * 8)
    assert smoke.GATE_PROGRESS_ADAPTER_OBSERVATION_FIELDS[0] == (
        "official_gate_progress_norm"
    )


def test_vq2_visual_policy_observation_matches_legal_runtime_layout():
    telemetry = smoke.TelemetryState(
        xgyro=2.0,
        ygyro=-4.0,
        zgyro=40.0,
        actuator_outputs=(0.2, 0.4, 0.6, 1.2),
    )
    history = (
        (0.1, 0.2, 0.3, 0.4),
        (-0.1, -0.2, -0.3, -0.4),
        (0.5, 0.6, 0.7, 0.8),
    )
    observation = smoke.build_vq2_visual_policy_observation(
        telemetry,
        visual_mask=[0.25] * smoke.VQ2_VISUAL_MASK_SIZE,
        action_history=history,
        new_frame=True,
        frame_age_s=0.125,
        official_gate_index=20,
    )

    assert len(observation) == 4119
    assert observation[:4096] == pytest.approx([0.25] * 4096)
    assert observation[4096:4099] == pytest.approx((0.1, -0.2, 1.0))
    assert observation[4099:4103] == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert observation[4103:4115] == pytest.approx(sum(history, ()))
    assert observation[4115:4117] == pytest.approx((1.0, 0.5))
    assert observation[4117:] == pytest.approx((20.0 / 6.0, 20.0 / 6.0))


def test_vq2_visual_catchup_advances_action_history_and_frame_age():
    observation = smoke.build_vq2_visual_policy_observation(
        smoke.TelemetryState(),
        visual_mask=[0.0] * smoke.VQ2_VISUAL_MASK_SIZE,
        action_history=((0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8), (0.0,) * 4),
        new_frame=True,
        frame_age_s=0.0,
        official_gate_index=3,
    )
    advanced = smoke.vq2_visual_observation_after_action(
        observation, (-1.0, -0.5, 0.5, 1.0), state_hz=64.0
    )

    assert advanced[4103:4115] == pytest.approx(
        (-1.0, -0.5, 0.5, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    )
    assert advanced[4115] == 0.0
    assert advanced[4116] == pytest.approx(1.0 / 16.0)
    assert advanced[4117:] == observation[4117:]


def test_policy_observation_appends_native_parity_six_gate_onehot():
    telemetry = smoke.TelemetryState()
    for gate_index in range(6):
        obs = smoke.build_policy_observation(
            telemetry,
            gate_detection=None,
            gate_pose=None,
            elapsed_fraction=0.25,
            last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
            gate_phase_onehot_adapter_gate_index=gate_index,
            gate_phase_onehot_adapter_denominator=6,
        )

        assert len(obs) == 32
        assert obs[22] == pytest.approx(-0.4)
        assert obs[23] == pytest.approx(gate_index / 6.0)
        assert obs[24:30] == pytest.approx(
            tuple(1.0 if index == gate_index else 0.0 for index in range(6))
        )
        assert obs[30:32] == pytest.approx((0.0, 0.0))
    assert smoke.GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS == (
        "official_gate_progress_norm",
        "official_gate_one_active",
        "official_gate_two_active",
        "official_gate_three_active",
        "official_gate_four_active",
        "official_gate_five_active",
        "official_gate_six_active",
        "reserved_gate_phase_1",
        "reserved_gate_phase_2",
    )


def test_policy_observation_carries_exact_hybrid_prefix_confidence():
    telemetry = smoke.TelemetryState()
    pose = SimpleNamespace(
        body_vector_ned_m=(12.0, -0.5, 0.25),
        yaw_error_rad=-0.04,
        range_camera_m=12.02,
        confidence=0.03125,
    )
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        gate_phase_onehot_adapter_gate_index=1,
        gate_phase_onehot_adapter_denominator=6,
        hybrid_prefix_confidence_observation=True,
        hybrid_prefix_elapsed_fraction=0.625,
    )

    assert obs[30] == pytest.approx(pose.confidence)
    assert obs[31] == pytest.approx(0.625)
    assert smoke.HYBRID_GATE_PHASE_ONEHOT_ADAPTER_OBSERVATION_FIELDS[-2:] == (
        "legacy_gate_pose_confidence",
        "legacy_elapsed_time_fraction",
    )


def test_hybrid_prefix_confidence_requires_onehot_adapter():
    with pytest.raises(ValueError, match="requires the gate-phase one-hot"):
        smoke.build_policy_observation(
            smoke.TelemetryState(),
            gate_detection=None,
            gate_pose=None,
            elapsed_fraction=0.0,
            last_cmd_norm=(0.0, 0.0, 0.0, 0.0),
            hybrid_prefix_confidence_observation=True,
        )


def test_policy_observation_appends_native_parity_phase_adapter_values():
    assert smoke.PHASE_ADAPTER_OBSERVATION_FIELDS == (
        "official_gate_one_active",
        "official_gate_two_active",
        "official_gate_three_active",
        "gate_two_right_norm",
        "gate_two_right_rate_norm",
        "gate_two_down_norm",
        "gate_two_down_rate_norm",
        "gate_three_right_norm",
        "gate_three_right_rate_norm",
    )
    telemetry = smoke.TelemetryState()
    pose = SimpleNamespace(
        body_vector_ned_m=(8.0, -1.25, 0.75),
        yaw_error_rad=-0.15,
        range_camera_m=8.13,
        confidence=0.9,
    )
    base = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
    )
    gate_two = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        phase_adapter_gate_index=2,
    )
    gate_three = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
        phase_adapter_gate_index=3,
    )

    assert len(base) == 23
    assert len(gate_two) == 32
    assert gate_two[:23] == pytest.approx(base)
    assert gate_two[23:32] == pytest.approx(
        (0.0, 1.0, 0.0, base[12], base[1], base[13], base[2], 0.0, 0.0)
    )
    assert gate_three[:23] == pytest.approx(base)
    assert gate_three[23:32] == pytest.approx(
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, base[12], base[1])
    )


def test_policy_observation_holds_associated_gate_across_missing_frame():
    telemetry = smoke.TelemetryState()
    pose = SimpleNamespace(
        body_vector_ned_m=(5.0, -1.0, 0.25),
        yaw_error_rad=-0.2,
        range_camera_m=5.105,
        confidence=0.8,
    )
    motion = smoke.ObservableGateMotionFilter()
    smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=pose,
        elapsed_fraction=0.0,
        last_cmd_norm=(0.0, 0.0, 0.0, 0.0),
        observation_time_s=1.0,
        gate_motion_state=motion,
        gate_identity=2,
    )
    held = smoke.build_policy_observation(
        telemetry,
        gate_detection=None,
        gate_pose=None,
        elapsed_fraction=0.01,
        last_cmd_norm=(0.0, 0.0, 0.0, 0.0),
        observation_time_s=1.1,
        gate_motion_state=motion,
        gate_identity=2,
    )

    assert held[10] == pytest.approx(1.0)
    assert held[11] == pytest.approx(math.tanh(0.5))
    assert held[12] == pytest.approx(math.tanh(-0.2))


def test_load_policy_callable_from_python_file(tmp_path):
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "def infer(observation):\n"
        "    return [0.5, -0.5, 0.25, -0.25]\n"
    )
    infer = smoke.load_policy_callable(f"{policy_path}:infer")

    result = infer(tuple([0.0] * smoke.OBSERVATION_SIZE))
    assert result == [0.5, -0.5, 0.25, -0.25]


def test_parse_policy_action_json_clamps_inputs():
    action = smoke.maybe_parse_action_json("[2.0, -2.0, 0.5, 0.0]")
    assert action == pytest.approx((1.0, -1.0, 0.5, 0.0))


def test_update_approach_diagnostics_tracks_closest_and_samples():
    diagnostics = smoke.ApproachDiagnostics()
    far_detection = smoke.GateDetection(
        corners=((290.0, 150.0), (350.0, 150.0), (350.0, 210.0), (290.0, 210.0)),
        area_px=3600.0,
        bounding_width_px=60.0,
        bounding_height_px=60.0,
        fill_ratio=0.9,
        confidence=0.4,
    )
    near_detection = smoke.GateDetection(
        corners=((250.0, 110.0), (390.0, 110.0), (390.0, 250.0), (250.0, 250.0)),
        area_px=19600.0,
        bounding_width_px=140.0,
        bounding_height_px=140.0,
        fill_ratio=0.9,
        confidence=0.8,
    )
    far_pose = smoke.estimate_gate_pose_from_corners(far_detection.corners)
    near_pose = smoke.estimate_gate_pose_from_corners(near_detection.corners)

    smoke.update_approach_diagnostics(
        diagnostics,
        elapsed_s=1.2345678,
        sim_time_ns=100,
        gate_pose=far_pose,
        detection=far_detection,
        visual_servo_target=smoke.LocalNedSetpoint(vx=0.1, vy=0.2, vz=-0.3, yaw_rate=0.4),
        max_samples=1,
    )
    smoke.update_approach_diagnostics(
        diagnostics,
        elapsed_s=2.0,
        sim_time_ns=200,
        gate_pose=near_pose,
        detection=near_detection,
        visual_servo_target=smoke.LocalNedSetpoint(vx=0.5, vy=-0.25, vz=0.1, yaw_rate=-0.2),
        actual_command={
            "kind": "visual_servo_attitude_target",
            "body_pitch_rate": -0.3,
            "body_yaw_rate": 0.1,
            "thrust": 0.58,
        },
        max_samples=1,
    )

    assert diagnostics.detections_sampled == 2
    assert diagnostics.closest_range_m == pytest.approx(round(near_pose.range_camera_m, 6))
    assert diagnostics.closest_range_elapsed_s == pytest.approx(2.0)
    assert diagnostics.highest_confidence == pytest.approx(0.8)
    assert diagnostics.latest_command["kind"] == "visual_servo_attitude_target"
    assert diagnostics.latest_command["body_pitch_rate"] == pytest.approx(-0.3)
    assert diagnostics.closest_range_command["thrust"] == pytest.approx(0.58)
    assert len(diagnostics.samples) == 1
    assert diagnostics.samples[0]["sim_time_ns"] == 200
    assert diagnostics.samples[0]["actual_command"]["body_yaw_rate"] == pytest.approx(0.1)


def test_attitude_servo_command_uses_gate_pose_for_pitch_yaw_and_thrust():
    pose = type(
        "Pose",
        (),
        {
            "body_vector_ned_m": (3.0, 0.5, -0.5),
            "yaw_error_rad": 0.25,
        },
    )()
    args = SimpleNamespace(
        attitude_servo_desired_standoff_m=0.0,
        attitude_servo_max_pitch_rate_rad_s=0.5,
        attitude_servo_max_roll_rate_rad_s=0.4,
        attitude_servo_max_yaw_rate_rad_s=0.7,
        attitude_servo_hover_thrust=0.58,
        attitude_servo_min_thrust=0.35,
        attitude_servo_max_thrust=0.75,
        attitude_servo_k_pitch=0.16,
        attitude_servo_k_roll=0.0,
        attitude_servo_k_yaw=1.2,
        attitude_servo_k_thrust=0.08,
    )

    target = smoke.attitude_servo_command_from_args(pose, args)

    assert target.body_pitch_rate < 0.0
    assert target.body_yaw_rate > 0.0
    assert target.body_roll_rate == pytest.approx(0.0)
    assert target.thrust > args.attitude_servo_hover_thrust


def test_attitude_servo_can_hold_forward_until_gate_centered():
    pose = type(
        "Pose",
        (),
        {
            "body_vector_ned_m": (5.0, 0.0, -2.0),
            "yaw_error_rad": 0.5,
        },
    )()
    args = SimpleNamespace(
        attitude_servo_desired_standoff_m=0.0,
        attitude_servo_max_pitch_rate_rad_s=0.5,
        attitude_servo_max_roll_rate_rad_s=0.4,
        attitude_servo_max_yaw_rate_rad_s=0.7,
        attitude_servo_hover_thrust=0.58,
        attitude_servo_min_thrust=0.35,
        attitude_servo_max_thrust=0.75,
        attitude_servo_k_pitch=0.16,
        attitude_servo_k_roll=0.0,
        attitude_servo_k_yaw=1.2,
        attitude_servo_k_thrust=0.08,
        attitude_servo_forward_yaw_tolerance_rad=0.2,
        attitude_servo_forward_z_tolerance_m=0.7,
        attitude_servo_uncentered_forward_scale=0.0,
    )

    target = smoke.attitude_servo_command_from_args(pose, args)

    assert target.body_pitch_rate == pytest.approx(0.0)
    assert target.body_yaw_rate == pytest.approx(0.6)
    assert target.thrust == pytest.approx(0.74)


def test_attitude_control_pose_filter_rejects_tiny_or_far_targets():
    detection = smoke.GateDetection(
        corners=((314.0, 348.0), (325.0, 348.0), (325.0, 358.0), (314.0, 358.0)),
        area_px=66.5,
        bounding_width_px=11.0,
        bounding_height_px=10.0,
        fill_ratio=0.6,
        confidence=0.56,
    )
    pose = smoke.estimate_gate_pose_from_corners(detection.corners)
    args = SimpleNamespace(
        attitude_servo_min_control_size_px=30.0,
        attitude_servo_max_control_range_m=20.0,
        attitude_servo_min_control_confidence=0.3,
    )

    assert smoke.attitude_control_pose_allowed(pose, detection, args) is False

    near_detection = smoke.GateDetection(
        corners=((293.0, 263.0), (368.0, 263.0), (368.0, 338.0), (293.0, 338.0)),
        area_px=2906.0,
        bounding_width_px=49.0,
        bounding_height_px=75.0,
        fill_ratio=0.79,
        confidence=0.62,
    )
    near_pose = smoke.estimate_gate_pose_from_corners(near_detection.corners)

    assert smoke.attitude_control_pose_allowed(near_pose, near_detection, args) is True


def test_attitude_final_approach_latches_bounded_command():
    # Vertically centered gate: with the level v3379 camera this is the
    # aligned-approach geometry (body z ~ 0) that should arm final approach.
    detection = smoke.GateDetection(
        corners=((293.0, 143.0), (368.0, 143.0), (368.0, 218.0), (293.0, 218.0)),
        area_px=2906.0,
        bounding_width_px=49.0,
        bounding_height_px=75.0,
        fill_ratio=0.79,
        confidence=0.62,
    )
    pose = smoke.estimate_gate_pose_from_corners(detection.corners)
    args = SimpleNamespace(
        attitude_servo_final_approach=True,
        attitude_servo_final_max_activations=1,
        attitude_servo_final_duration_s=1.2,
        attitude_servo_final_trigger_range_m=7.0,
        attitude_servo_final_min_size_px=40.0,
        attitude_servo_final_min_confidence=0.45,
        attitude_servo_final_max_yaw_error_rad=0.25,
        attitude_servo_final_max_abs_z_m=1.0,
        attitude_servo_final_pitch_rate_rad_s=0.22,
        attitude_servo_final_max_yaw_rate_rad_s=0.35,
        attitude_servo_final_thrust=0.63,
        attitude_servo_min_thrust=0.35,
        attitude_servo_max_thrust=0.78,
        attitude_servo_k_yaw=2.0,
    )
    state = smoke.AttitudeFinalApproachState()

    assert smoke.should_start_attitude_final_approach(
        pose,
        detection,
        args,
        state,
        now_s=10.0,
    ) is True

    command = smoke.attitude_final_approach_command_from_args(pose, args)
    state.activate(now_s=10.0, started_s=9.0, duration_s=1.2, pose=pose, command=command)
    summary = state.to_summary(now_s=10.5, started_s=9.0)

    assert command.body_pitch_rate == pytest.approx(-0.22)
    assert command.thrust == pytest.approx(0.63)
    assert state.is_active(10.5) is True
    assert summary["activations"] == 1
    assert summary["last_activation_elapsed_s"] == pytest.approx(1.0)
    assert smoke.should_start_attitude_final_approach(
        pose,
        detection,
        args,
        state,
        now_s=10.5,
    ) is False


def test_vision_gate_pass_tracker_emits_ordered_pass_and_completion():
    cfg = smoke.GatePassConfig(
        arm_range_m=3.0,
        pass_range_m=1.2,
        rearm_range_m=1.6,
        min_consecutive_pass_frames=2,
        max_missed_detection_frames=3,
        pass_cooldown_s=0.3,
        confidence_arm_min=0.35,
        confidence_pass_min=0.5,
    )
    tracker = smoke.VisionGatePassTracker(config=cfg, target_gate_count=1)
    pose_far = type("Pose", (), {"range_camera_m": 3.6})()
    pose_near = type("Pose", (), {"range_camera_m": 1.0})()

    assert tracker.observe(now_s=10.0, gate_pose=pose_far, detection_confidence=0.7, detection_present=True) is False
    assert tracker.armed is True
    assert tracker.observe(now_s=10.2, gate_pose=pose_near, detection_confidence=0.8, detection_present=True) is False
    assert tracker.observe(now_s=10.24, gate_pose=pose_near, detection_confidence=0.8, detection_present=True) is True
    assert tracker.pass_count == 1
    assert tracker.completion_time_s == pytest.approx(10.24)

    summary = tracker.to_summary(started_s=10.0)
    assert summary["pass_count"] == 1
    assert summary["completion_time_s"] == pytest.approx(0.24)
    assert summary["events"][0]["confidence"] == pytest.approx(0.8)


def test_vision_gate_pass_tracker_requires_rearm_and_cooldown():
    cfg = smoke.GatePassConfig(
        arm_range_m=3.0,
        pass_range_m=1.2,
        rearm_range_m=1.6,
        min_consecutive_pass_frames=2,
        max_missed_detection_frames=3,
        pass_cooldown_s=0.3,
        confidence_arm_min=0.35,
        confidence_pass_min=0.5,
    )
    tracker = smoke.VisionGatePassTracker(config=cfg, target_gate_count=2)
    pose_far = type("Pose", (), {"range_camera_m": 3.5})()
    pose_near = type("Pose", (), {"range_camera_m": 1.1})()

    tracker.observe(now_s=0.0, gate_pose=pose_far, detection_confidence=0.7, detection_present=True)
    tracker.observe(now_s=0.1, gate_pose=pose_near, detection_confidence=0.7, detection_present=True)
    assert tracker.observe(now_s=0.14, gate_pose=pose_near, detection_confidence=0.7, detection_present=True) is True
    assert tracker.observe(now_s=0.2, gate_pose=pose_near, detection_confidence=0.7, detection_present=True) is False
    assert tracker.observe(now_s=0.6, gate_pose=pose_near, detection_confidence=0.7, detection_present=True) is False
    tracker.observe(now_s=0.8, gate_pose=pose_far, detection_confidence=0.7, detection_present=True)
    tracker.observe(now_s=0.9, gate_pose=pose_near, detection_confidence=0.7, detection_present=True)
    assert tracker.observe(now_s=0.94, gate_pose=pose_near, detection_confidence=0.7, detection_present=True) is True
    assert tracker.pass_count == 2


def test_vision_gate_pass_tracker_debounces_across_missed_detections():
    cfg = smoke.GatePassConfig(min_consecutive_pass_frames=2, max_missed_detection_frames=2)
    tracker = smoke.VisionGatePassTracker(config=cfg, target_gate_count=1)
    pose_far = type("Pose", (), {"range_camera_m": 3.3})()
    pose_near = type("Pose", (), {"range_camera_m": 1.1})()

    tracker.observe(now_s=0.0, gate_pose=pose_far, detection_confidence=0.8, detection_present=True)
    assert tracker.observe(now_s=0.1, gate_pose=pose_near, detection_confidence=0.8, detection_present=True) is False
    assert tracker.observe(now_s=0.12, gate_pose=None, detection_confidence=None, detection_present=False) is False
    assert tracker.observe(now_s=0.14, gate_pose=None, detection_confidence=None, detection_present=False) is False
    assert tracker.observe(now_s=0.16, gate_pose=pose_near, detection_confidence=0.8, detection_present=True) is True
    assert tracker.pass_count == 1


def test_evaluate_acceptance_reports_blockers():
    sitl = smoke.SitlRunReport(
        endpoint="udpout:127.0.0.1:14540",
        mode="competition-smoke",
        duration_s=1.0,
        heartbeat_hz=2.0,
        command_hz=50.0,
        command_kind="local_ned_velocity",
    )
    sitl.telemetry.messages_seen = 0
    report = smoke.CompetitionSmokeReport(
        sitl=sitl,
        control_mode="visual-servo",
        policy_source="visual_servo",
        ordered_gate_passes=0,
        vision=smoke.SmokeVisionMetrics(frames_seen=0),
    )
    ok, blockers = smoke.evaluate_acceptance(
        report,
        require_telemetry=True,
        require_camera=True,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=1,
        min_camera_frames=1,
        max_telemetry_dropouts=2,
    )
    assert ok is False
    assert "no_telemetry_messages" in blockers
    assert "no_camera_frames" in blockers
    assert "insufficient_gate_passes:0<1" in blockers


def test_evaluate_acceptance_passes_when_requirements_met():
    sitl = smoke.SitlRunReport(
        endpoint="udpout:127.0.0.1:14540",
        mode="competition-smoke",
        duration_s=1.0,
        heartbeat_hz=2.0,
        command_hz=50.0,
        command_kind="local_ned_velocity",
    )
    sitl.telemetry.messages_seen = 10
    report = smoke.CompetitionSmokeReport(
        sitl=sitl,
        control_mode="visual-servo",
        policy_source="visual_servo",
        ordered_gate_passes=1,
        vision=smoke.SmokeVisionMetrics(frames_seen=5),
    )
    ok, blockers = smoke.evaluate_acceptance(
        report,
        require_telemetry=True,
        require_camera=True,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=1,
        min_camera_frames=1,
        max_telemetry_dropouts=2,
    )
    assert ok is True
    assert blockers == []


def test_evaluate_acceptance_rejects_slow_commands_and_telemetry_backlog():
    sitl = smoke.SitlRunReport(
        endpoint="udpout:127.0.0.1:14540",
        mode="competition-smoke",
        duration_s=10.0,
        heartbeat_hz=2.0,
        command_hz=60.0,
        effective_command_hz=49.0,
        command_kind="body_rates_course_fsm_target",
    )
    sitl.telemetry.messages_seen = 10
    sitl.telemetry.drain_limit_hits = 1
    report = smoke.CompetitionSmokeReport(
        sitl=sitl,
        control_mode="course-fsm",
        policy_source="cyan_guidance_gate_fsm",
        ordered_gate_passes=1,
        vision=smoke.SmokeVisionMetrics(frames_seen=5),
    )

    ok, blockers = smoke.evaluate_acceptance(
        report,
        require_telemetry=True,
        require_camera=True,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=1,
        min_camera_frames=1,
        max_telemetry_dropouts=2,
        min_effective_command_hz=50.0,
        max_telemetry_drain_limit_hits=0,
    )

    assert ok is False
    assert "effective_command_rate_too_low:49.000<50.000" in blockers
    assert "telemetry_drain_limit_hits_exceeded:1>0" in blockers


def test_effective_command_rate_excludes_reset_and_calibration_lead_time():
    rate, interval = smoke.calculate_effective_command_rate(
        commands_sent=423,
        first_command_sent_s=3.2,
        last_command_sent_s=10.233333333333333,
        fallback_duration_s=10.266,
    )

    assert interval == pytest.approx(7.033333333333333)
    assert rate == pytest.approx(60.0)


def test_evaluate_acceptance_can_require_official_race_progress():
    sitl = smoke.SitlRunReport(
        endpoint="udpout:127.0.0.1:14540",
        mode="competition-smoke",
        duration_s=1.0,
        heartbeat_hz=2.0,
        command_hz=50.0,
        command_kind="local_ned_velocity",
    )
    sitl.telemetry.messages_seen = 10
    sitl.telemetry.race_statuses = 1
    report = smoke.CompetitionSmokeReport(
        sitl=sitl,
        control_mode="visual-servo",
        policy_source="visual_servo",
        ordered_gate_passes=1,
        official_active_gate_index=0,
        official_last_gate_race_time=-1,
        vision=smoke.SmokeVisionMetrics(frames_seen=5),
    )
    ok, blockers = smoke.evaluate_acceptance(
        report,
        require_telemetry=True,
        require_camera=True,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=1,
        min_camera_frames=1,
        max_telemetry_dropouts=2,
        require_official_race_progress=True,
    )
    assert ok is False
    assert "insufficient_official_gate_progress:0<1" in blockers

    report.official_active_gate_index = 1
    ok, blockers = smoke.evaluate_acceptance(
        report,
        require_telemetry=True,
        require_camera=True,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=1,
        min_camera_frames=1,
        max_telemetry_dropouts=2,
        require_official_race_progress=True,
    )
    assert ok is True
    assert blockers == []


class FakeTelemetry:
    def __init__(self):
        self.metrics = sitl_adapter.TelemetryMetrics()
        self.state = sitl_adapter.TelemetryState()

    def update_ages(self):
        pass


class FakeAdapter:
    instances = []

    def __init__(self, endpoint, dropout_after_s=1.0):
        self.endpoint = endpoint
        self.dropout_after_s = dropout_after_s
        self.telemetry = FakeTelemetry()
        self.attitude_targets = []
        self.local_ned_targets = []
        self.heartbeats = 0
        self.arm_commands = 0
        self.master = type("Master", (), {"close": lambda _self: None})()
        self.__class__.instances.append(self)

    def send_heartbeat(self):
        self.heartbeats += 1

    def send_arm_command(self):
        self.arm_commands += 1

    def send_attitude_setpoint(self, target, *, mode="body_rates"):
        self.attitude_targets.append((target, mode))

    def send_local_ned_setpoint(self, target, *, frame="local_ned", yaw_mode="yaw_and_rate"):
        self.local_ned_targets.append((target, frame, yaw_mode))

    def poll_telemetry(self, *, timeout_s=0.0):
        return None

    def drain_telemetry(self, *, timeout_s=0.0, max_messages=512):
        return 0


def test_fixed_rate_publisher_sustains_compliant_wire_rate():
    adapter = FakeAdapter("udpout:127.0.0.1:14540")
    publisher = smoke.FixedRateAttitudePublisher(
        adapter,
        command_hz=60.0,
        initial_target=smoke.AttitudeSetpoint(thrust=0.27),
        send_lock=threading.Lock(),
    )

    publisher.start()
    time.sleep(0.75)
    publisher.stop()
    rate, interval = smoke.calculate_effective_command_rate(
        commands_sent=publisher.commands_sent,
        first_command_sent_s=publisher.first_command_sent_s,
        last_command_sent_s=publisher.last_command_sent_s,
        fallback_duration_s=0.75,
    )

    assert interval >= 0.6
    assert 50.0 <= rate < 100.0
    assert publisher.command_rate_violations == 0


class FakePolicyShadowAdapter(FakeAdapter):
    def __init__(self, endpoint, dropout_after_s=1.0):
        super().__init__(endpoint, dropout_after_s=dropout_after_s)
        self.disarm_commands = 0
        self.telemetry.metrics.heartbeats = 1
        self.telemetry.state.race_status = sitl_adapter.RaceStatus(
            sim_boot_time_ms=10_000,
            race_start_boot_time_ms=1_000,
            race_finish_time_ns=-1,
            active_gate_index=0,
            last_gate_race_time=-1,
        )
        self.telemetry.state.xacc = 0.0
        self.telemetry.state.yacc = 0.0
        self.telemetry.state.zacc = -9.81
        self.telemetry.state.xgyro = 0.0
        self.telemetry.state.ygyro = 0.0
        self.telemetry.state.zgyro = 0.0
        self.telemetry.state.imu_time_usec = 0

    def drain_telemetry(self, *, timeout_s=0.0, max_messages=512):
        self.telemetry.state.imu_time_usec += 10_000
        return 1

    def send_disarm_command(self):
        self.disarm_commands += 1


class FakeOfficialResetClock:
    def __init__(self):
        self.now_s = 0.0

    def monotonic(self):
        return self.now_s

    def sleep(self, duration_s):
        self.now_s += max(0.001, float(duration_s))


class FakeOfficialResetAdapter:
    def __init__(self, clock):
        self.clock = clock
        self.telemetry = FakeTelemetry()
        self.telemetry.metrics.heartbeats = 1
        self.telemetry.state.race_status = sitl_adapter.RaceStatus(
            sim_boot_time_ms=10_000,
            race_start_boot_time_ms=1_000,
            race_finish_time_ns=-1,
            active_gate_index=0,
            last_gate_race_time=-1,
        )
        self.reset_at_s = None
        self.reset_commands = 0
        self.disarm_commands = 0
        self.arm_commands = 0
        self.heartbeats = 0
        self.imu_seq = 0
        self.command_order = []

    def send_heartbeat(self):
        self.heartbeats += 1
        self.command_order.append("heartbeat")

    def send_disarm_command(self):
        self.disarm_commands += 1
        self.command_order.append("disarm")

    def send_sim_reset_command(self):
        self.reset_commands += 1
        self.command_order.append("reset")
        self.reset_at_s = self.clock.now_s

    def send_arm_command(self):
        self.arm_commands += 1

    def drain_telemetry(self, *, timeout_s=0.0, max_messages=512):
        if self.reset_at_s is None:
            return 1
        elapsed_s = self.clock.now_s - self.reset_at_s
        self.imu_seq += 1
        self.telemetry.state.imu_time_usec = self.imu_seq * 10_000
        self.telemetry.state.xacc = 0.0
        self.telemetry.state.yacc = 0.0
        self.telemetry.state.zacc = -9.81
        self.telemetry.state.xgyro = 0.0
        self.telemetry.state.ygyro = 0.0
        self.telemetry.state.zgyro = 0.0
        self.telemetry.state.race_status = sitl_adapter.RaceStatus(
            sim_boot_time_ms=int(elapsed_s * 1000.0),
            race_start_boot_time_ms=700,
            race_finish_time_ns=-1,
            active_gate_index=0,
            last_gate_race_time=-1,
        )
        return 1


def test_official_policy_reset_waits_for_actual_scheduled_start_boundary():
    clock = FakeOfficialResetClock()
    adapter = FakeOfficialResetAdapter(clock)
    estimator = smoke.DeadReckoningEstimator()

    result = smoke.reset_and_wait_for_official_policy_start(
        adapter,
        estimator,
        heartbeat_hz=2.0,
        timeout_s=2.0,
        policy_lead_s=0.05,
        min_calibration_samples=60,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )

    assert adapter.reset_commands == 1
    assert adapter.disarm_commands == 1
    assert adapter.command_order[:3] == ["heartbeat", "disarm", "reset"]
    assert adapter.arm_commands >= 1
    assert estimator.state.calibrated is True
    assert result.report["reset_detected"] is True
    assert result.report["calibration_samples"] == 60
    assert 0.0 <= result.report["policy_lead_s"] <= 0.05
    assert result.race_start_monotonic_s == pytest.approx(0.8, abs=0.01)


def smoke_args_for_attitude(tmp_path):
    return SimpleNamespace(
        acceptance_config=str(ROOT / "config" / "sitl_competition_acceptance.json"),
        endpoint="udpin:0.0.0.0:14550",
        heartbeat_hz=2.0,
        command_hz=50.0,
        duration=0.04,
        telemetry_timeout_s=0.0,
        telemetry_dropout_s=1.0,
        idle_sleep_s=0.001,
        arm_on_start=True,
        arm_attempts=3,
        prearm_heartbeat_timeout_s=0.0,
        control_mode="attitude-rates",
        command_frame="local_ned",
        command_yaw_mode="ignore",
        attitude_mode="body_rates",
        attitude_roll_rad=0.0,
        attitude_pitch_rad=0.0,
        attitude_yaw_rad=0.0,
        body_roll_rate_rad_s=0.1,
        body_pitch_rate_rad_s=-0.3,
        body_yaw_rate_rad_s=0.2,
        attitude_thrust=0.6,
        attitude_servo_desired_standoff_m=0.0,
        attitude_servo_max_pitch_rate_rad_s=0.5,
        attitude_servo_max_roll_rate_rad_s=0.4,
        attitude_servo_max_yaw_rate_rad_s=0.7,
        attitude_servo_hover_thrust=0.58,
        attitude_servo_min_thrust=0.35,
        attitude_servo_max_thrust=0.75,
        attitude_servo_k_pitch=0.16,
        attitude_servo_k_roll=0.0,
        attitude_servo_k_yaw=1.2,
        attitude_servo_k_thrust=0.08,
        attitude_servo_search_pitch_rate_rad_s=0.0,
        attitude_servo_search_yaw_rate_rad_s=0.0,
        attitude_servo_search_thrust=None,
        attitude_servo_forward_yaw_tolerance_rad=None,
        attitude_servo_forward_z_tolerance_m=None,
        attitude_servo_uncentered_forward_scale=1.0,
        policy_action_json="[0.0, 0.0, 0.0, 0.0]",
        policy_callable="",
        camera_host="0.0.0.0",
        camera_port=5600,
        camera_timeout_s=0.0,
        camera_max_packets_per_loop=512,
        no_camera=True,
        max_detection_age_s=0.25,
        visual_servo_desired_standoff_m=1.0,
        visual_servo_max_forward_m_s=1.0,
        visual_servo_max_lateral_m_s=0.5,
        visual_servo_max_vertical_m_s=0.4,
        visual_servo_max_yaw_rate_rad_s=0.6,
        visual_servo_k_forward=0.45,
        visual_servo_k_lateral=0.7,
        visual_servo_k_vertical=0.7,
        visual_servo_k_yaw=1.2,
        detector_min_area_px=1200.0,
        detector_max_aspect_error=0.5,
        detector_min_fill_ratio=0.15,
        max_approach_diagnostic_samples=12,
        target_gate_count=1,
        require_official_race_progress=False,
        gate_confidence_arm_min=None,
        gate_confidence_pass_min=None,
        gate_pass_arm_range_m=None,
        gate_pass_range_m=None,
        gate_pass_rearm_range_m=None,
        gate_pass_min_consecutive_frames=None,
        gate_pass_max_missed_frames=None,
        gate_pass_cooldown_s=None,
        require_telemetry=False,
        require_camera=False,
        min_gate_passes=1,
        max_command_rate_violations=0,
        min_telemetry_messages=0,
        min_camera_frames=0,
        max_telemetry_dropouts=2,
        json_path=str(tmp_path / "smoke.json"),
        csv_path=str(tmp_path / "smoke.csv"),
    )


def test_policy_shadow_runs_callable_but_never_arms_resets_or_sends_setpoints(
    monkeypatch, tmp_path
):
    FakePolicyShadowAdapter.instances.clear()
    monkeypatch.setattr(smoke, "MavlinkSitlAdapter", FakePolicyShadowAdapter)
    monkeypatch.setattr(
        smoke,
        "load_policy_callable",
        lambda _spec: (lambda observation: (0.1, -0.2, 0.3, -0.4)),
    )
    args = smoke_args_for_attitude(tmp_path)
    args.control_mode = "policy-attitude"
    args.arm_on_start = False
    args.policy_shadow_only = True
    args.official_reset_on_start = False
    args.policy_callable = "shadow_policy.py:infer"
    args.policy_gate_phase_onehot_adapter_observation = True
    args.policy_race_phase_denominator = 6
    args.policy_trace_hz = 1000.0
    args.policy_trace_max_samples = 100

    report = smoke.run_smoke(args)
    adapter = FakePolicyShadowAdapter.instances[-1]

    assert report.acceptance_passed is True
    assert report.acceptance_blockers == []
    assert report.control_inputs["policy_shadow_only"] is True
    assert report.control_inputs["disarm_commands_sent"] == 0
    assert report.sitl.commands_sent == 0
    assert report.sitl.command_kind == "shadow_policy_attitude_no_setpoint"
    assert adapter.arm_commands == 0
    assert adapter.attitude_targets == []
    assert adapter.local_ned_targets == []
    assert adapter.disarm_commands == 0
    assert report.control_inputs["official_reset_start"]["reset_sent"] is False
    assert report.policy_trace["samples"]
    assert len(report.policy_trace["samples"][0]["observation"]) == 32


def test_run_smoke_attitude_rates_sends_attitude_targets(monkeypatch, tmp_path):
    FakeAdapter.instances = []
    monkeypatch.setattr(smoke, "MavlinkSitlAdapter", FakeAdapter)

    report = smoke.run_smoke(smoke_args_for_attitude(tmp_path))

    adapter = FakeAdapter.instances[0]
    assert adapter.arm_commands == 3
    assert adapter.local_ned_targets == []
    assert len(adapter.attitude_targets) >= 1
    target, mode = adapter.attitude_targets[0]
    assert mode == "body_rates"
    assert target.body_pitch_rate == pytest.approx(-0.3)
    assert target.thrust == pytest.approx(0.6)
    assert report.control_mode == "attitude-rates"
    assert report.sitl.command_kind == "body_rates_attitude_target"
    assert report.control_inputs["attitude_target"]["body_pitch_rate"] == pytest.approx(-0.3)


def test_run_smoke_visual_servo_attitude_searches_when_gate_missing(monkeypatch, tmp_path):
    FakeAdapter.instances = []
    monkeypatch.setattr(smoke, "MavlinkSitlAdapter", FakeAdapter)
    args = smoke_args_for_attitude(tmp_path)
    args.control_mode = "visual-servo-attitude"
    args.body_pitch_rate_rad_s = 0.0
    args.body_yaw_rate_rad_s = 0.0
    args.attitude_thrust = 0.5
    args.attitude_servo_search_pitch_rate_rad_s = -0.02
    args.attitude_servo_search_yaw_rate_rad_s = 0.45
    args.attitude_servo_search_thrust = 0.57

    report = smoke.run_smoke(args)

    adapter = FakeAdapter.instances[0]
    assert adapter.local_ned_targets == []
    assert len(adapter.attitude_targets) >= 1
    target, mode = adapter.attitude_targets[0]
    assert mode == "body_rates"
    assert target.body_pitch_rate == pytest.approx(-0.02)
    assert target.body_yaw_rate == pytest.approx(0.45)
    assert target.thrust == pytest.approx(0.57)
    assert report.control_inputs["attitude_servo"]["search_yaw_rate_rad_s"] == pytest.approx(0.45)
    assert report.sitl.command_kind == "body_rates_visual_servo_attitude_target"


def test_official_gate_progress_retargets_control_state():
    from drone_state_estimator import DeadReckoningEstimator

    angle = smoke.AttitudeAngleServoState(
        yaw_cmd_rad=0.5,
        filt_vec=(1.0, 2.0, 3.0),
        filt_time_s=10.0,
        control_state="track",
        blind_command=smoke.AttitudeSetpoint(roll=0.0, pitch=-0.1, yaw=0.5, thrust=0.6),
    )
    map_guidance = smoke.MapGuidanceState(carry_bearing=(1.0, 0.0))
    tracker = smoke.VisionGatePassTracker()
    tracker.armed = True
    tracker.awaiting_rearm = True
    tracker.consecutive_pass_frames = 2
    estimator = DeadReckoningEstimator()
    estimator.state.calibrated = True
    estimator.state.position_ned_m = (25.0, 0.0, 0.0)
    estimator.landmarks[0] = (20.0, 0.0, 0.0)
    progress = smoke.OfficialGateProgressState(last_official_gate_index=0)

    retargeted = progress.observe(
        elapsed_s=12.0,
        official_gate_index=1,
        estimator=estimator,
        angle_servo=angle,
        map_guidance=map_guidance,
        pass_tracker=tracker,
    )

    assert retargeted is True
    assert progress.advance_count == 1
    assert angle.filt_vec is None
    assert angle.control_state == "search"
    assert angle.blind_command is None
    assert angle.yaw_cmd_rad == pytest.approx(0.0)
    assert map_guidance.carry_bearing is None
    assert tracker.pass_count == 1
    assert tracker.armed is False
    assert tracker.awaiting_rearm is False
    assert tracker.consecutive_pass_frames == 0


def test_grounded_limp_triggers_after_recovery_timeout():
    limp = smoke.GroundedLimpState()

    assert (
        limp.observe(
            now_s=1.0,
            elapsed_s=1.0,
            recovery_active=True,
            collision_count=0,
            velocity_ned_m_s=(0.0, 0.0, 0.0),
            recovery_timeout_s=3.0,
            max_speed_m_s=0.15,
            collision_grace_s=2.0,
        )
        is False
    )
    assert (
        limp.observe(
            now_s=4.5,
            elapsed_s=4.5,
            recovery_active=True,
            collision_count=0,
            velocity_ned_m_s=(0.01, 0.0, 0.0),
            recovery_timeout_s=3.0,
            max_speed_m_s=0.15,
            collision_grace_s=2.0,
        )
        is True
    )
    assert limp.active is True
    assert limp.reason == "recovery_timeout_near_zero_velocity"


def test_grounded_limp_immediate_trigger_latches_first_reason():
    limp = smoke.GroundedLimpState()

    assert limp.trigger("official_collision_message", 1.25) is True
    assert limp.trigger("later_reason", 2.0) is False
    assert limp.active is True
    assert limp.limp_events == 1
    assert limp.reason == "official_collision_message"
    assert limp.elapsed_s_at_trigger == pytest.approx(1.25)


def test_angle_servo_reset_for_gate_retarget_keeps_thrust_bias():
    angle = smoke.AttitudeAngleServoState(thrust_bias=0.04, filt_vec=(1.0, 0.0, 0.0))
    angle.reset_for_gate_retarget(yaw_cmd_rad=1.2)
    assert angle.filt_vec is None
    assert angle.yaw_cmd_rad == pytest.approx(1.2)
    assert angle.thrust_bias == pytest.approx(0.04)
