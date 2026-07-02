import importlib.util
import sys
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
    obs = smoke.build_policy_observation(
        telemetry,
        gate_detection=detection,
        gate_pose=pose,
        elapsed_fraction=0.25,
        last_cmd_norm=(0.1, -0.2, 0.3, -0.4),
    )

    assert len(obs) == smoke.OBSERVATION_SIZE
    assert max(obs) <= 1.0
    assert min(obs) >= -1.0


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
        max_samples=1,
    )

    assert diagnostics.detections_sampled == 2
    assert diagnostics.closest_range_m == pytest.approx(round(near_pose.range_camera_m, 6))
    assert diagnostics.closest_range_elapsed_s == pytest.approx(2.0)
    assert diagnostics.highest_confidence == pytest.approx(0.8)
    assert diagnostics.latest_command["vx"] == pytest.approx(0.5)
    assert len(diagnostics.samples) == 1
    assert diagnostics.samples[0]["sim_time_ns"] == 200


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
