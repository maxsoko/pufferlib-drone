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
        probe_json_path=str(tmp_path / "probe.json"),
        send_sim_reset=False,
        post_reset_sleep_s=0.0,
        acceptance_config="config/sitl_competition_acceptance.json",
        endpoint="udpin:0.0.0.0:14540",
        control_mode="visual-servo",
        command_frame="local_ned",
        policy_callable="",
        policy_action_json="[0,0,0,0]",
        smoke_duration=1.0,
        heartbeat_hz=2.0,
        command_hz=50.0,
        telemetry_timeout_s=0.0,
        telemetry_dropout_s=1.0,
        idle_sleep_s=0.001,
        camera_host="0.0.0.0",
        camera_timeout_s=0.0,
        camera_max_packets_per_loop=512,
        max_detection_age_s=0.25,
        detector_min_area_px=1200.0,
        detector_max_aspect_error=0.5,
        detector_min_fill_ratio=0.15,
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
