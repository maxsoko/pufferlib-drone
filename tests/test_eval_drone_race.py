import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_drone_race.py"
SPEC = importlib.util.spec_from_file_location("eval_drone_race", SCRIPT_PATH)
EVAL_DRONE_RACE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL_DRONE_RACE)


def test_resolve_eval_seeds_uses_incrementing_custom_seeds():
    seeds, suite_name = EVAL_DRONE_RACE.resolve_eval_seeds(episodes=4, seed=7, suite=None)
    assert seeds == [7, 8, 9, 10]
    assert suite_name is None


def test_resolve_eval_seeds_uses_fixed_suite_values():
    seeds, suite_name = EVAL_DRONE_RACE.resolve_eval_seeds(
        episodes=999,
        seed=0,
        suite="drone_race_v1_quick20",
    )
    assert suite_name == "drone_race_v1_quick20"
    assert len(seeds) == 20
    assert seeds[0] == 42
    assert seeds[-1] == 61


def test_list_seed_suites_reports_expected_full_suite_metadata():
    suites = EVAL_DRONE_RACE.list_seed_suites()
    assert "drone_race_v1_full100" in suites
    assert suites["drone_race_v1_full100"]["num_seeds"] == 100
    assert suites["drone_race_v1_full100"]["first_seed"] == 1000
    assert suites["drone_race_v1_full100"]["last_seed"] == 1099
