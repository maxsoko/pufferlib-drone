import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_full_policy_transfer_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("train_full_policy_transfer_curriculum", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_curriculum(path: Path, stages: list[dict]) -> Path:
    path.write_text(json.dumps({"version": 1, "stages": stages}), encoding="utf-8")
    return path


def test_default_curriculum_models_gate1_dropout_and_monotonic_jitter():
    stages = MODULE.load_curriculum(MODULE.DEFAULT_CURRICULUM)
    assert stages[0]["overrides"]["env.sitl-gate-obs-dropout-from-index"] == "1"
    assert stages[0]["overrides"]["env.num-gates"] == "2"
    assert stages[3]["overrides"]["env.num-gates"] == "3"
    assert stages[-1]["overrides"]["env.num-gates"] == "4"
    assert stages[-1]["overrides"]["env.gate-position-jitter-z"] == "2.50"
    assert [stage["name"] for stage in stages] == list(dict.fromkeys(stage["name"] for stage in stages))


def test_curriculum_rejects_late_dropout(tmp_path):
    path = write_curriculum(
        tmp_path / "bad.json",
        [{"name": "bad", "overrides": {"env.sitl-gate-obs-dropout-from-index": "2"}}],
    )
    with pytest.raises(ValueError, match="gate-1 camera loss"):
        MODULE.load_curriculum(path)


def test_curriculum_rejects_decreasing_jitter(tmp_path):
    path = write_curriculum(
        tmp_path / "bad.json",
        [
            {
                "name": "wide",
                "overrides": {
                    "env.sitl-gate-obs-dropout-from-index": "1",
                    "env.gate-position-jitter-y": "2.0",
                },
            },
            {
                "name": "narrow",
                "overrides": {
                    "env.sitl-gate-obs-dropout-from-index": "1",
                    "env.gate-position-jitter-y": "1.0",
                },
            },
        ],
    )
    with pytest.raises(ValueError, match="decreases"):
        MODULE.load_curriculum(path)


def test_evaluate_command_requires_exact_episode_count(tmp_path):
    metrics, report, command = MODULE.evaluate(
        python="python",
        checkpoint=Path("policy.bin"),
        env_name="drone_race_full_policy_stage_d_gate4",
        stage_name="dropout",
        settings={"env.sitl-gate-obs-dropout-from-index": "1"},
        output_dir=tmp_path,
        attempt_id="test",
        episodes=1024,
        horizon=32,
        max_rollouts=64,
        dry_run=True,
    )
    assert metrics is None
    assert report.name == "transfer_test_dropout.json"
    assert "--require-exact-episodes" in command
    assert command[-2:] == ["--env.sitl-gate-obs-dropout-from-index", "1"]


def test_zero_success_regression_uses_average_ordered_gates():
    parent = {"success_rate": 0.0, "gates_passed": 1.89}
    child = {"success_rate": 0.0, "gates_passed": 1.75}
    assert MODULE.progress_regressed(
        parent,
        child,
        maximum_success_drop=0.05,
        maximum_gates_drop=0.10,
    )
