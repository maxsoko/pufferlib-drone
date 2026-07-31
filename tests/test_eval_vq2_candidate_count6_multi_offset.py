from __future__ import annotations

import inspect

import scripts.eval_vq2_candidate_count6_multi_offset as screen


def test_candidate_screen_uses_four_real_offsets_and_six_gates() -> None:
    assert screen.OFFSETS == (0, 8, 16, 24)
    assert screen.AGENTS == screen.EPISODES == 64
    assert screen.THREADS == 4
    assert screen.MAX_STEPS == 3072
    source = inspect.getsource(screen.run)
    assert "num_gates=6" in source
    assert "evaluation_episode_offset" in inspect.getsource(screen.offset_config)


def test_candidate_qualification_requires_safety_and_downstream_gain() -> None:
    baseline = {
        "gate_reach": {"1": 256, "2": 241, "3": 52, "4": 5, "5": 0, "6": 0},
        "successes": 0,
        "crashes": 17,
        "mean_gates_passed": 2.1640625,
    }
    candidate = {
        "gate_reach": {"1": 256, "2": 241, "3": 52, "4": 6, "5": 0, "6": 0},
        "successes": 0,
        "crashes": 17,
        "mean_gates_passed": 2.17,
        "all_hard_transport_pass": True,
    }
    assert all(screen.qualification(baseline, candidate).values())
    candidate["crashes"] = 18
    assert not all(screen.qualification(baseline, candidate).values())
    candidate["crashes"] = 17
    candidate["gate_reach"]["2"] = 240
    assert not all(screen.qualification(baseline, candidate).values())


def test_candidate_screen_is_offline_and_puffer_only() -> None:
    source = inspect.getsource(screen)
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in source
    assert '"teacher_plant_actions": 0' in source
    assert '"flight_sim_packets_sent": 0' in source
    assert '"submission_authorized": False' in source
