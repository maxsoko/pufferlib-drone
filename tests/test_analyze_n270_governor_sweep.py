from pathlib import Path

import pytest

from scripts import analyze_n270_governor_sweep as module


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_LOGS = Path(
    "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl"
)


def test_candidate_grid_contains_baseline_and_near_24_rung():
    candidates = module.candidate_grid()
    assert module.BASELINE in candidates
    assert module.replay_model.GovernorParameters(
        crossing_speed_m_s=9.0,
        maximum_along_speed_m_s=9.0,
        slowdown_distance_m=10.0,
        cross_track_gain_s_inv=2.0,
        maximum_cross_track_correction_m_s=4.0,
    ) in candidates
    assert len(candidates) == 30


def test_minimal_selection_prefers_fewer_changes_then_slower_safe_rung():
    candidates = [
        {
            "safe_offline": True,
            "meets_near_24_target": True,
            "parameter_change_count": 3,
            "robust_worst_predicted_finish_time_s": 23.8,
            "robust_maximum_crossing_error_m": 0.3,
            "parameters": {"crossing": 9.1},
        },
        {
            "safe_offline": True,
            "meets_near_24_target": True,
            "parameter_change_count": 3,
            "robust_worst_predicted_finish_time_s": 24.1,
            "robust_maximum_crossing_error_m": 0.4,
            "parameters": {"crossing": 9.0},
        },
    ]
    selected = module.choose_minimal_robust_candidate(candidates)
    assert selected["parameters"]["crossing"] == 9.0


def test_real_three_trace_sweep_selects_source_locked_n272():
    import json

    evidence = ROOT / "logs/sitl/n271_n270_three_trace_governor_sweep.json"
    assert module.sha256(evidence) == (
        "a5d3e09625ae8883f19b111d61c8c691310d361fffba593d6db6ee47ccbe1278"
    )
    report = json.loads(evidence.read_text())
    recommendation = report["minimal_robust_recommendation"]
    assert recommendation["parameters"] == {
        "crossing_speed_m_s": 9.0,
        "maximum_along_speed_m_s": 9.0,
        "slowdown_distance_m": 10.0,
        "cross_track_gain_s_inv": 2.0,
        "maximum_cross_track_correction_m_s": 4.0,
    }
    assert recommendation["parameter_change_count"] == 3
    assert recommendation["robust_worst_predicted_finish_time_s"] < 24.25
    assert recommendation["robust_maximum_crossing_error_m"] < 0.45
    assert report["preregistered_next_action"]["experiment"] == "N272"
    assert report["preregistered_next_action"]["authoritative_stop_gate_index"] == 3
