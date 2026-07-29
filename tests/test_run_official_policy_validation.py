import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_official_policy_validation as validation


def test_official_finish_time_prefers_top_level_smoke_field():
    assert validation._official_finish_time_ns(
        {
            "official_race_finish_time_ns": 123,
            "race_status": {"race_finish_time_ns": 456},
        }
    ) == 123


def test_official_finish_time_falls_back_safely():
    assert validation._official_finish_time_ns(
        {"race_status": {"race_finish_time_ns": 456}}
    ) == 456
    assert validation._official_finish_time_ns({}) == -1
    assert validation._official_finish_time_ns(
        {"official_race_finish_time_ns": "invalid"}
    ) == -1


def test_clean_gate_progress_requires_progress_and_acceptance():
    clean = {
        "acceptance_passed": True,
        "crash_detected": False,
        "invalid_run": False,
    }
    assert validation._clean_gate_progress_passed(
        smoke=clean, official_index=3, minimum_index=3, child_exit_code=0
    )
    assert not validation._clean_gate_progress_passed(
        smoke=clean, official_index=2, minimum_index=3, child_exit_code=0
    )


def test_clean_gate_progress_fails_closed_on_child_or_smoke_failure():
    clean = {
        "acceptance_passed": True,
        "crash_detected": False,
        "invalid_run": False,
    }
    assert not validation._clean_gate_progress_passed(
        smoke=clean, official_index=3, minimum_index=3, child_exit_code=1
    )
    for field in ("acceptance_passed", "crash_detected", "invalid_run"):
        smoke = dict(clean)
        smoke[field] = False if field == "acceptance_passed" else True
        assert not validation._clean_gate_progress_passed(
            smoke=smoke, official_index=3, minimum_index=3, child_exit_code=0
        )


def test_fixed_policy_state_cadence_is_forwarded_to_child_validation():
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--policy-state-hz", type=float, default=0.0)' in source
    assert '"--policy-state-hz",\n        str(args.policy_state_hz),' in source
