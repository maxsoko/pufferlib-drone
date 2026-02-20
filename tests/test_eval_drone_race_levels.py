import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_drone_race_levels.py"
SPEC = importlib.util.spec_from_file_location("eval_drone_race_levels", SCRIPT_PATH)
EVAL_LEVELS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL_LEVELS)


def test_evaluate_threshold_supports_relational_ops():
    assert EVAL_LEVELS.evaluate_threshold(0.9, ">=", 0.8)
    assert EVAL_LEVELS.evaluate_threshold(0.9, "<=", 1.0)
    assert EVAL_LEVELS.evaluate_threshold(5, ">", 4)
    assert EVAL_LEVELS.evaluate_threshold(5, "<", 6)
    assert EVAL_LEVELS.evaluate_threshold(5, "==", 5)


def test_compute_level_metrics_extracts_gate_and_consistency_metrics():
    rows = [
        {"success": 1, "valid_run": 1, "gates_passed": 3, "completion_time": 10.0, "progress": 0.9},
        {"success": 1, "valid_run": 1, "gates_passed": 1, "completion_time": 12.0, "progress": 0.8},
        {"success": 0, "valid_run": 0, "gates_passed": 0, "completion_time": float("nan"), "progress": 0.2},
        {"success": 1, "valid_run": 1, "gates_passed": 2, "completion_time": 11.0, "progress": 0.7},
    ]

    metrics = EVAL_LEVELS.compute_level_metrics(rows)

    assert metrics["success_rate"] == 0.75
    assert metrics["valid_rate"] == 0.75
    assert metrics["first_gate_pass_rate"] == 0.75
    assert metrics["two_gate_pass_rate"] == 0.5
    assert metrics["max_consecutive_successes"] == 2
    assert metrics["median_completion_time_valid"] == 11.0


def test_load_levels_config_parses_order_and_level_fields(tmp_path):
    cfg = tmp_path / "levels.ini"
    cfg.write_text(
        """
[meta]
schema_version = 1

[levels]
order = L1,L2

[L1]
description = level one
suite = drone_race_v1_quick20
episodes = 20
metric = valid_rate
operator = >=
threshold = 0.9
env_kwargs = {"num_gates": 1}

[L2]
description = level two
suite = drone_race_v1_full100
episodes = 100
metric = success_rate
operator = >=
threshold = 0.5
env_kwargs = {"num_gates": 4}
""".strip()
    )

    meta, levels = EVAL_LEVELS.load_levels_config(str(cfg))

    assert meta["schema_version"] == "1"
    assert [level.name for level in levels] == ["L1", "L2"]
    assert levels[0].episodes == 20
    assert levels[0].env_kwargs["num_gates"] == 1
    assert levels[1].suite == "drone_race_v1_full100"
