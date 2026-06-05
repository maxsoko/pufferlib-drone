import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "watch_official_gate1_validation.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("watch_official_gate1_validation", MODULE_PATH)
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
        attempt_json_template=str(tmp_path / "attempt_{attempt:03d}.json"),
        poll_interval_s=0.01,
        max_wait_s=0.03,
        watch_summary_json_path=str(tmp_path / "watch_summary.json"),
    )


def _validation_result(status: str) -> SimpleNamespace:
    payload = {
        "status": status,
        "next_commands": ["cmd1", "cmd2"],
    }
    return SimpleNamespace(
        status=status,
        to_dict=lambda: payload,
    )


def test_watch_times_out_when_traffic_never_appears(tmp_path):
    args = _args(tmp_path)
    timeline = iter([0.0, 0.0, 0.01, 0.02, 0.03, 0.04])

    def now_fn():
        return next(timeline)

    calls = {"n": 0}

    def run_once(_validation_args):
        calls["n"] += 1
        return _validation_result("blocked_no_traffic")

    summary = module.run_watch(
        args,
        run_once=run_once,
        now_fn=now_fn,
        sleep_fn=lambda _s: None,
    )
    assert summary.final_status == "watch_timeout_no_traffic"
    assert summary.attempts >= 1
    assert calls["n"] == summary.attempts
    attempt_path = Path(args.attempt_json_template.format(attempt=1))
    assert attempt_path.exists()
    with open(attempt_path) as f:
        payload = json.load(f)
    assert payload["status"] == "blocked_no_traffic"


def test_watch_stops_when_smoke_passes(tmp_path):
    args = _args(tmp_path)
    args.max_wait_s = 1.0
    timeline = iter([0.0, 0.0, 0.2, 0.25])
    statuses = iter(["blocked_no_traffic", "smoke_passed"])
    validation_args_seen = []

    def now_fn():
        return next(timeline)

    def run_once(validation_args):
        validation_args_seen.append(validation_args)
        return _validation_result(next(statuses))

    summary = module.run_watch(
        args,
        run_once=run_once,
        now_fn=now_fn,
        sleep_fn=lambda _s: None,
    )
    assert summary.final_status == "smoke_passed"
    assert summary.attempts == 2
    assert summary.attempt_statuses == ["blocked_no_traffic", "smoke_passed"]
    assert validation_args_seen[-1].camera_max_packets_per_loop == 512
