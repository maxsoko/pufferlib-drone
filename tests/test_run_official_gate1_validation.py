import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_official_gate1_validation.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("run_official_gate1_validation", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        probe_duration=0.1,
        race_start_check_s=0.0,
        probe_json_path=str(tmp_path / "probe.json"),
        send_sim_reset=False,
        policy_ready_reset=False,
        post_reset_sleep_s=0.0,
        official_reset_start_timeout_s=12.0,
        official_policy_lead_s=0.05,
        acceptance_config="config/sitl_competition_acceptance.json",
        endpoint="udpin:0.0.0.0:14540",
        control_mode="visual-servo",
        command_frame="local_ned",
        command_yaw_mode="yaw_and_rate",
        attitude_mode="body_rates",
        attitude_roll_rad=0.0,
        attitude_pitch_rad=0.0,
        attitude_yaw_rad=0.0,
        body_roll_rate_rad_s=0.0,
        body_pitch_rate_rad_s=0.0,
        body_yaw_rate_rad_s=0.0,
        attitude_thrust=0.5,
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
        policy_callable="",
        policy_state_hz=0.0,
        policy_action_json="[0,0,0,0]",
        policy_stop_before_gate_index=-1,
        policy_stop_forward_m=0.0,
        policy_race_phase_observation=False,
        policy_race_phase_denominator=3,
        policy_phase_adapter_observation=False,
        policy_gate_progress_adapter_observation=False,
        policy_gate_phase_onehot_adapter_observation=False,
        smoke_duration=1.0,
        heartbeat_hz=2.0,
        command_hz=50.0,
        telemetry_timeout_s=0.0,
        telemetry_dropout_s=1.0,
        idle_sleep_s=0.001,
        arm_on_start=True,
        arm_attempts=3,
        prearm_heartbeat_timeout_s=2.0,
        camera_host="0.0.0.0",
        camera_timeout_s=0.0,
        camera_max_packets_per_loop=512,
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
        smoke_json_path=str(tmp_path / "smoke.json"),
        smoke_csv_path=str(tmp_path / "smoke.csv"),
        summary_json_path=str(tmp_path / "summary.json"),
    )


def test_validation_blocks_when_probe_requirements_fail(monkeypatch, tmp_path):
    args = _args(tmp_path)
    probe_report = module.probe.ProbeReport(
        duration_s=0.1,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        mavlink=module.probe.MavlinkProbeStats(),
        camera=module.probe.CameraProbeStats(),
        requirements_met=True,
        blockers=[],
    )

    monkeypatch.setattr(module.probe, "run_probe", lambda **kwargs: probe_report)
    monkeypatch.setattr(
        module.probe,
        "evaluate_probe_requirements",
        lambda report, **kwargs: (False, ["no_mavlink_packets"]),
    )

    called = {"smoke": 0}

    def fake_smoke_run(_args):
        called["smoke"] += 1
        raise AssertionError("smoke should not run when probe fails")

    monkeypatch.setattr(module.smoke, "run_smoke", fake_smoke_run)
    summary = module.run_validation(args)

    assert summary.status == "blocked_no_traffic"
    assert called["smoke"] == 0
    assert Path(args.probe_json_path).exists()
    with open(args.probe_json_path) as f:
        payload = json.load(f)
    assert payload["requirements_met"] is False
    assert payload["blockers"] == ["no_mavlink_packets"]


def test_smoke_args_defer_gate_requirement_to_acceptance_config(tmp_path):
    args = _args(tmp_path)
    args.acceptance_config = "config/sitl_multigate_acceptance.json"
    args.control_mode = "course-fsm"
    args.course_controller_config = "config/course_fsm_defaults.json"

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.min_gate_passes is None
    assert smoke_args.control_mode == "course-fsm"
    assert smoke_args.course_controller_config.endswith("course_fsm_defaults.json")


def test_build_smoke_args_allows_bounded_gate_target_override(tmp_path):
    args = _args(tmp_path)
    args.min_gate_passes = 1

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.target_gate_count == 1
    assert smoke_args.min_gate_passes == 1


def test_build_smoke_args_forwards_policy_race_phase(tmp_path):
    args = _args(tmp_path)
    args.policy_race_phase_observation = True
    args.policy_race_phase_denominator = 5

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_race_phase_observation is True
    assert smoke_args.policy_race_phase_denominator == 5


def test_build_smoke_args_forwards_fixed_policy_state_cadence(tmp_path):
    args = _args(tmp_path)
    args.policy_state_hz = 60.0

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_state_hz == 60.0


def test_build_smoke_args_forwards_policy_phase_adapter(tmp_path):
    args = _args(tmp_path)
    args.policy_phase_adapter_observation = True

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_phase_adapter_observation is True


def test_build_smoke_args_forwards_policy_gate_progress_adapter(tmp_path):
    args = _args(tmp_path)
    args.policy_gate_progress_adapter_observation = True
    args.policy_race_phase_denominator = 6

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_gate_progress_adapter_observation is True
    assert smoke_args.policy_race_phase_denominator == 6


def test_build_smoke_args_forwards_policy_gate_phase_onehot_adapter(tmp_path):
    args = _args(tmp_path)
    args.policy_gate_phase_onehot_adapter_observation = True
    args.policy_race_phase_denominator = 6

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_gate_phase_onehot_adapter_observation is True


def test_build_smoke_args_forwards_hybrid_prefix_confidence(tmp_path):
    args = _args(tmp_path)
    args.policy_gate_phase_onehot_adapter_observation = True
    args.policy_hybrid_prefix_confidence_observation = True
    args.policy_race_phase_denominator = 6

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_hybrid_prefix_confidence_observation is True
    assert smoke_args.policy_race_phase_denominator == 6


def test_build_smoke_args_forwards_policy_stop_guard(tmp_path):
    args = _args(tmp_path)
    args.policy_stop_before_gate_index = 3
    args.policy_stop_forward_m = 20.0

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.policy_stop_before_gate_index == 3
    assert smoke_args.policy_stop_forward_m == 20.0


def test_build_smoke_args_forwards_official_gate_index_stop(tmp_path):
    args = _args(tmp_path)
    args.stop_after_official_gate_index = 4

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.stop_after_official_gate_index == 4


def test_build_smoke_args_forwards_policy_ready_reset(tmp_path):
    args = _args(tmp_path)
    args.control_mode = "policy-attitude"
    args.send_sim_reset = True
    args.policy_ready_reset = True

    smoke_args = module._build_smoke_args(args)

    assert smoke_args.official_reset_on_start is True
    assert smoke_args.official_reset_start_timeout_s == 12.0
    assert smoke_args.official_policy_lead_s == 0.05


def test_policy_ready_reset_skips_stale_camera_and_race_prechecks(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.control_mode = "policy-attitude"
    args.send_sim_reset = True
    args.policy_ready_reset = True
    probe_report = module.probe.ProbeReport(
        duration_s=0.1,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        mavlink=module.probe.MavlinkProbeStats(packets_seen=1),
        camera=module.probe.CameraProbeStats(),
        requirements_met=True,
        blockers=[],
    )
    monkeypatch.setattr(module.probe, "run_probe", lambda **kwargs: probe_report)
    requirements = {}

    def evaluate(_report, **kwargs):
        requirements.update(kwargs)
        return True, []

    monkeypatch.setattr(module.probe, "evaluate_probe_requirements", evaluate)
    monkeypatch.setattr(
        module,
        "_maybe_send_sim_reset",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("reset must be delegated to the loaded policy runner")
        ),
    )
    monkeypatch.setattr(
        module,
        "_check_race_started",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("the policy-owned reset establishes race start")
        ),
    )
    smoke_report = module.smoke.CompetitionSmokeReport(
        sitl=module.smoke.SitlRunReport(
            mode="competition-smoke",
            endpoint=args.endpoint,
            heartbeat_hz=2.0,
            command_hz=60.0,
            command_kind="body_rates_policy_attitude_target",
            heartbeats_sent=2,
            commands_sent=60,
            duration_s=1.0,
        ),
        control_mode="policy-attitude",
        policy_source="checkpoint",
        control_inputs={
            "official_reset_start": {"reset_sent": True, "reset_detected": True}
        },
        ordered_gate_passes=1,
        acceptance_passed=True,
    )
    monkeypatch.setattr(module.smoke, "run_smoke", lambda _args: smoke_report)

    summary = module.run_validation(args)

    assert requirements["require_mavlink"] is True
    assert requirements["require_camera"] is False
    assert requirements["require_ts002_header"] is False
    assert summary.reset_sent is True
    assert summary.status == "smoke_passed"
    assert summary.race_start_check["reason"] == "policy_ready_reset_owns_race_start"


def test_validation_runs_smoke_when_probe_passes(monkeypatch, tmp_path):
    args = _args(tmp_path)
    probe_report = module.probe.ProbeReport(
        duration_s=0.1,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        mavlink=module.probe.MavlinkProbeStats(packets_seen=1),
        camera=module.probe.CameraProbeStats(packets_seen=1, ts002_header_packets=1),
        requirements_met=True,
        blockers=[],
    )
    monkeypatch.setattr(module.probe, "run_probe", lambda **kwargs: probe_report)
    monkeypatch.setattr(
        module.probe,
        "evaluate_probe_requirements",
        lambda report, **kwargs: (True, []),
    )

    smoke_report = module.smoke.CompetitionSmokeReport(
        sitl=module.smoke.SitlRunReport(
            mode="competition-smoke",
            endpoint="udpin:0.0.0.0:14540",
            heartbeat_hz=2.0,
            command_hz=50.0,
            command_kind="local_ned_velocity",
            heartbeats_sent=2,
            commands_sent=50,
            duration_s=1.0,
        ),
        control_mode="visual-servo",
        policy_source="visual_servo",
        ordered_gate_passes=1,
        acceptance_passed=True,
    )
    monkeypatch.setattr(module.smoke, "run_smoke", lambda _args: smoke_report)
    summary = module.run_validation(args)

    assert summary.status == "smoke_passed"
    assert summary.smoke is not None
    assert summary.smoke["acceptance_passed"] is True
    assert Path(args.probe_json_path).exists()
    assert Path(args.smoke_json_path).exists()


def test_validation_reports_race_not_started_when_camera_is_menu_stream(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.race_start_check_s = 1.0
    probe_report = module.probe.ProbeReport(
        duration_s=0.1,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        mavlink=module.probe.MavlinkProbeStats(packets_seen=1),
        camera=module.probe.CameraProbeStats(packets_seen=1, ts002_header_packets=1),
        requirements_met=True,
        blockers=[],
    )
    monkeypatch.setattr(module.probe, "run_probe", lambda **kwargs: probe_report)
    monkeypatch.setattr(
        module.probe,
        "evaluate_probe_requirements",
        lambda report, **kwargs: (True, []),
    )

    monkeypatch.setattr(
        module,
        "_check_race_started",
        lambda _args: {
            "skipped": False,
            "duration_s": 0.1,
            "race_started": False,
            "race_status": {
                "sim_boot_time_ms": 10,
                "race_start_boot_time_ms": -1,
                "race_finish_time_ns": -1,
                "active_gate_index": 0,
                "last_gate_race_time": -1,
            },
            "heartbeats_seen": 1,
            "messages_seen": 1,
        },
    )
    monkeypatch.setattr(
        module.smoke,
        "run_smoke",
        lambda _args: (_ for _ in ()).throw(AssertionError("smoke should wait for race start")),
    )

    summary = module.run_validation(args)

    assert summary.status == "blocked_race_not_started"
    assert summary.smoke is None
