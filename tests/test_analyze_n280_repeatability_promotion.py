from pathlib import Path

import pytest

from scripts import analyze_n280_repeatability_promotion as module


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl")


def test_real_n280_n282_promotion():
    if not all((EVIDENCE / filename).exists() for filename, _ in module.EXPECTED_RUNS.values()):
        pytest.skip("N280--N282 Windows evidence is not mounted")
    report = module.build_report(
        EVIDENCE,
        ROOT / "logs/sitl/n270_v3391_cross7p5_repeatability_promotion.json",
        ROOT / "logs/sitl/n278_n277_equal_timestamp_ack_diagnosis.json",
        EVIDENCE / "n279_v3391_n259_cross9_xgain2_ack350_strict_gate4_bounded_001.json",
        ROOT
        / "checkpoints/drone_race_vq1_v3391_telemetry/n259_aperture_fullphase_bc_6gate.bin",
        ROOT / "scripts/run_v3391_telemetry_policy.py",
        ROOT / "config/drone_race_vq1_v3391_telemetry.ini",
    )
    promotion = report["promotion"]
    assert promotion["promoted"]
    assert promotion["official_valid_laps"] == "3/3"
    assert promotion["best_official_finish_time_s"] == pytest.approx(23.832376480)
    assert promotion["worst_official_finish_time_s"] < 24.5
    assert promotion["finish_time_range_s"] < 0.05
    assert report["comparison_to_n270"]["every_promoted_lap_faster_than_n270_best"]
    assert all(lap["collision"] is None for lap in report["official_full_laps"])
