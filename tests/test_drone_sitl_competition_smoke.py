import importlib.util
import sys
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
