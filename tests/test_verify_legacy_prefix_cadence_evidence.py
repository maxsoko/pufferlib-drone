import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.verify_legacy_prefix_cadence_evidence import (
    inference_rate,
    motion_update_rate,
    scheduler_contract,
)


def test_rates_use_actual_attempt_duration():
    report = {
        "sitl": {"duration_s": 5.0},
        "policy_trace": {
            "inference_ticks": 30,
            "motion_filter": {"accepted_samples": 20, "rejected_samples": 5},
        },
    }

    assert inference_rate(report) == pytest.approx(6.0)
    assert motion_update_rate(report) == pytest.approx(5.0)


def test_current_runner_releases_gil_before_windows_hot_deadline():
    contract = scheduler_contract(SCRIPTS / "drone_sitl_competition_smoke.py")

    assert contract["passed"] is True
    assert contract["windows_hot_deadline_s"] == pytest.approx(0.001)
