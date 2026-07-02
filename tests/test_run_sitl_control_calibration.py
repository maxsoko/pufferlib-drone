import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "run_sitl_control_calibration.py"
SPEC = importlib.util.spec_from_file_location("run_sitl_control_calibration", MODULE_PATH)
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def test_default_cases_cover_velocity_and_attitude_primitives():
    cases = calibration.default_cases()

    assert any(case.control_mode == "policy" and case.command_frame == "local_ned" for case in cases)
    assert any(case.control_mode == "policy" and case.command_frame == "body_ned" for case in cases)
    assert any(case.control_mode == "attitude-rates" and case.body_pitch_rate_rad_s < 0.0 for case in cases)
    assert any(case.control_mode == "visual-servo-attitude" for case in cases)


def test_run_calibration_filters_cases_and_summarizes_official_pass(monkeypatch, tmp_path):
    seen_args = []

    def fake_run_smoke(args):
        seen_args.append(args)
        sitl = calibration.smoke.SitlRunReport(
            endpoint=args.endpoint,
            mode="competition-smoke",
            duration_s=args.duration,
            heartbeat_hz=args.heartbeat_hz,
            command_hz=args.command_hz,
            command_kind="body_rates_attitude_target",
        )
        sitl.telemetry.messages_seen = 10
        report = calibration.smoke.CompetitionSmokeReport(
            sitl=sitl,
            control_mode=args.control_mode,
            policy_source="constant_attitude_rates",
            ordered_gate_passes=1,
            official_active_gate_index=1,
            official_last_gate_race_time=123,
            vision=calibration.smoke.SmokeVisionMetrics(frames_seen=4),
        )
        report.acceptance_passed = True
        report.approach_diagnostics.closest_range_m = 0.8
        report.approach_diagnostics.closest_range_elapsed_s = 2.5
        report.approach_diagnostics.highest_confidence = 0.9
        return report

    monkeypatch.setattr(calibration.smoke, "run_smoke", fake_run_smoke)
    args = calibration.build_parser().parse_args(
        [
            "--skip-probe",
            "--no-reset-between-cases",
            "--case-filter",
            "attitude_rates_pitch_negative_t060",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "unit",
            "--case-duration",
            "0.01",
            "--require-official-race-progress",
        ]
    )

    summary = calibration.run_calibration(args)

    assert len(seen_args) == 1
    assert seen_args[0].control_mode == "attitude-rates"
    assert seen_args[0].body_pitch_rate_rad_s == -0.3
    assert summary.cases_run == 1
    assert summary.official_passing_cases == ["attitude_rates_pitch_negative_t060"]
    assert summary.best_by_closest_range == "attitude_rates_pitch_negative_t060"
    assert (tmp_path / "control_calibration_unit_01_attitude_rates_pitch_negative_t060.json").exists()
