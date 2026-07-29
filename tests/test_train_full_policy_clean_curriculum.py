import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_full_policy_clean_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("train_full_policy_clean_curriculum", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_curriculum(path: Path, stages: list[dict]) -> Path:
    path.write_text(json.dumps({"version": 1, "stages": stages}), encoding="utf-8")
    return path


def final_stage(name: str = "target") -> dict:
    return {
        "name": name,
        "exact_overrides": {
            "env.num-gates": "4",
            "env.gate-radius": "0.75",
            "env.sitl-gate-obs-dropout-from-index": "1",
            "env.sitl-gate-obs-dropout-range-m": "4.25",
        },
    }


def test_default_curriculum_is_full_course_geometry_annealing_schedule():
    defaults, stages = MODULE.load_curriculum(MODULE.DEFAULT_CURRICULUM)
    assert defaults["timesteps"] == 500_000
    assert stages[0]["name"] == "coursewide_radius_18p00"
    assert stages[0]["training"]["learning_rate"] == pytest.approx(3e-4)
    assert stages[0]["training"]["timesteps"] == 4_000_000
    assert all(stage["exact_overrides"]["env.num-gates"] == "4" for stage in stages)
    early_radii = [
        float(stage["exact_overrides"]["env.gate-radius"])
        for stage in stages[:7]
    ]
    assert early_radii == [18.0, 4.0, 4.0, 4.0, 3.0, 3.25, 3.25]
    assert stages[-1]["exact_overrides"]["env.gate-radius"] == "0.75"
    assert stages[-1]["exact_overrides"]["env.sitl-gate-obs-dropout-range-m"] == "4.25"
    geometry_scales = [
        float(stage["exact_overrides"]["env.course-geometry-scale"])
        for stage in stages[1:15]
    ]
    assert geometry_scales == [
        0.0, 0.25, 0.35, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.80, 0.85,
        0.90, 0.95, 1.0
    ]
    geometry_ranges = [
        (
            float(stage["training_overrides"]["env.course-geometry-scale-min"]),
            float(stage["training_overrides"]["env.course-geometry-scale-max"]),
        )
        for stage in stages[6:15]
    ]
    assert geometry_ranges == [
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.65),
        (0.65, 0.75),
        (0.75, 0.80),
        (0.80, 0.85),
        (0.85, 0.90),
        (0.90, 0.95),
        (0.95, 1.00),
    ]
    assert all(
        stage["training_overrides"]["env.course-geometry-scale-randomize"]
        == "1"
        for stage in stages[6:15]
    )
    assert all(
        stage["exact_overrides"]["env.course-geometry-scale-randomize"]
        == "0"
        for stage in stages[6:15]
    )
    aperture_radii = [
        float(stage["exact_overrides"]["env.gate-radius"])
        for stage in stages[15:25]
    ]
    assert aperture_radii == [
        3.00, 2.95, 2.90, 2.89, 2.85625, 2.85, 2.84625, 2.84, 2.80, 2.75
    ]
    assert stages[19]["training"]["timesteps"] == 65_536
    assert stages[19]["training"]["learning_rate"] == pytest.approx(1e-6)
    assert stages[20]["training"]["timesteps"] == 65_536
    assert stages[20]["training"]["learning_rate"] == pytest.approx(1e-6)
    assert stages[21]["training"]["timesteps"] == 65_536
    assert stages[21]["training"]["learning_rate"] == pytest.approx(1e-6)
    assert stages[22]["training"]["timesteps"] == 65_536
    assert stages[22]["training"]["learning_rate"] == pytest.approx(1e-6)
    assert all(
        stage["training_overrides"]["env.gate-radius"]
        == stage["exact_overrides"]["env.gate-radius"]
        for stage in stages[17:25]
    )


def test_compact_numeric_ladder_expands_previous_value_training(tmp_path):
    payload = {
        "training_defaults": {"ent_coef": 0.0},
        "ladders": [
            {
                "name_prefix": "radius",
                "parameter": "env.gate-radius",
                "values": ["1.00", "0.75"],
                "train_on_previous_value": True,
                "exact_overrides": {
                    "env.num-gates": "4",
                    "env.sitl-gate-obs-dropout-range-m": "4.25",
                },
                "training_overrides": {"env.w-cross-track": "50"},
            }
        ],
    }
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    defaults, stages = MODULE.load_curriculum(path)

    assert defaults["ent_coef"] == 0.0
    assert [stage["name"] for stage in stages] == ["radius_1p00", "radius_0p75"]
    assert stages[0]["training_overrides"]["env.gate-radius"] == "1.00"
    assert stages[1]["training_overrides"]["env.gate-radius"] == "1.00"
    assert stages[1]["exact_overrides"]["env.gate-radius"] == "0.75"


def test_compact_numeric_ladder_materializes_current_value_training(tmp_path):
    payload = {
        "training_defaults": {"ent_coef": 0.0},
        "ladders": [
            {
                "name_prefix": "radius",
                "parameter": "env.gate-radius",
                "values": ["1.00", "0.75"],
                "train_on_previous_value": False,
                "exact_overrides": {
                    "env.num-gates": "4",
                    "env.sitl-gate-obs-dropout-range-m": "4.25",
                },
            }
        ],
    }
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, stages = MODULE.load_curriculum(path)

    assert stages[0]["training_overrides"]["env.gate-radius"] == "1.00"
    assert stages[1]["training_overrides"]["env.gate-radius"] == "0.75"


def test_coursewide_stages_preserve_recurrent_full_start_distribution():
    _, stages = MODULE.load_curriculum(MODULE.DEFAULT_CURRICULUM)
    for stage in stages[:9]:
        overrides = stage["training_overrides"]
        assert "env.mixed-start-curriculum" not in overrides
        assert "env.start-gate-index" not in overrides
    for stage in stages[:5]:
        assert stage["training_overrides"]["env.w-gate-exit-velocity"] == "10.0"
        assert stage["training_overrides"]["env.cross-track-from-gate-index"] == "0"
    assert stages[1]["training_overrides"]["env.w-gate-crossing-error"] == "0.25"
    assert (
        stages[1]["training_overrides"]["env.gate-crossing-error-from-gate-index"]
        == "0"
    )


def test_curriculum_rejects_training_on_different_gate_count(tmp_path):
    stage = final_stage()
    stage["training_overrides"] = {"env.num-gates": "3"}
    path = write_curriculum(tmp_path / "bad.json", [stage])
    with pytest.raises(ValueError, match="different gate counts"):
        MODULE.load_curriculum(path)


def test_curriculum_rejects_non_target_final_stage(tmp_path):
    stage = final_stage()
    stage["exact_overrides"]["env.gate-radius"] = "1.0"
    path = write_curriculum(tmp_path / "bad.json", [stage])
    with pytest.raises(ValueError, match="0.75 m"):
        MODULE.load_curriculum(path)


def test_fresh_train_does_not_load_checkpoint(tmp_path):
    checkpoints, _, command = MODULE.train(
        python="python",
        source=None,
        env_name="drone_race_full_policy_stage_d_gate4",
        settings={"env.num-gates": "1"},
        checkpoint_root=tmp_path,
        training={
            "timesteps": 100,
            "max_attempts": 1,
            "learning_rate": 3e-4,
            "reward_scale": 0.005,
            "reward_clip": 1.0,
            "exploration_log_std": None,
        },
        seed=123,
        dry_run=True,
    )
    assert checkpoints == []
    assert "--load-model-path" not in command
    assert command[command.index("--checkpoint-interval") + 1] == "5"
    assert command[command.index("--train.seed") + 1] == "123"
    assert command[-2:] == ["--env.num-gates", "1"]
    assert command[command.index("--train.ent-coef") + 1] == "0.001"


def test_exact_evaluation_is_distinct_and_requires_exact_episode_count(tmp_path):
    metrics, report, command = MODULE.evaluate(
        python="python",
        checkpoint=Path("policy.bin"),
        env_name="drone_race_full_policy_stage_d_gate4",
        stage_name="official_gate0",
        settings={"env.gate-radius": "0.75"},
        output_dir=tmp_path,
        run_id="test",
        episodes=1024,
        horizon=32,
        max_rollouts=96,
        dry_run=True,
    )
    assert metrics is None
    assert report.name == "clean_test_official_gate0.json"
    assert "--require-exact-episodes" in command
    assert command[-2:] == ["--env.gate-radius", "0.75"]


def test_training_timestep_cap_is_local_and_non_mutating():
    configured = {"timesteps": 1_048_576, "learning_rate": 1e-4}

    capped = MODULE.cap_training_timesteps(configured, 262_144)

    assert capped["timesteps"] == 262_144
    assert capped["learning_rate"] == configured["learning_rate"]
    assert configured["timesteps"] == 1_048_576
    assert MODULE.cap_training_timesteps(configured, None) == configured
    assert MODULE.cap_training_timesteps(configured, 2_000_000) == configured


def test_screen_pass_consolidation_only_selects_better_passing_child():
    assert MODULE.should_select_trained_child(
        child_accepted=True,
        child_improved=True,
        consolidating_screen_pass=True,
    )
    assert not MODULE.should_select_trained_child(
        child_accepted=True,
        child_improved=False,
        consolidating_screen_pass=True,
    )
    assert not MODULE.should_select_trained_child(
        child_accepted=False,
        child_improved=True,
        consolidating_screen_pass=True,
    )
    assert MODULE.should_select_trained_child(
        child_accepted=True,
        child_improved=False,
        consolidating_screen_pass=False,
    )


def test_progress_score_rejects_crash_heavy_shortcut():
    safe = {
        "success_rate": 0.0,
        "crash_rate": 0.0,
        "gates_passed": 1.5,
        "missed_gate_rate": 0.1,
    }
    crashing = {
        "success_rate": 0.05,
        "crash_rate": 0.9,
        "gates_passed": 1.8,
        "missed_gate_rate": 0.0,
    }
    assert MODULE.progress_score(safe, 0.1) > MODULE.progress_score(crashing, 0.1)


def test_pause_after_working_screen_preserves_retained_parent():
    retained = "/tmp/retained.bin"
    candidate = Path("/tmp/candidate.bin")
    metrics = {
        "success_rate": 0.88,
        "crash_rate": 0.0,
        "gates_passed": 3.8,
        "missed_gate_rate": 0.12,
    }
    state = {
        "retained_checkpoint": retained,
        "status": "stage_exhausted",
        "exhausted_stage": "radius_2p85",
        "exhausted_stage_index": 20,
    }

    MODULE.pause_after_working_screen(
        state,
        checkpoint=candidate,
        metrics=metrics,
        stage_index=20,
    )

    assert state["retained_checkpoint"] == retained
    assert state["working_checkpoint"] == str(candidate)
    assert state["working_metrics"] == metrics
    assert state["working_stage_index"] == 20
    assert state["status"] == "paused_after_screen"
    assert "exhausted_stage" not in state
    assert "exhausted_stage_index" not in state


def test_progress_score_detects_smaller_gate_miss():
    parent = {
        "success_rate": 0.0,
        "crash_rate": 0.0,
        "gates_passed": 1.0,
        "terminal_crossing_radial": 3.35,
        "missed_gate_rate": 1.0,
    }
    child = {**parent, "terminal_crossing_radial": 3.29}
    assert MODULE.progress_score(child, 0.1) > MODULE.progress_score(parent, 0.1)


def test_material_progress_rejects_centimetre_only_radial_change():
    parent = {
        "success_rate": 0.0,
        "crash_rate": 0.0,
        "gates_passed": 1.0,
        "terminal_crossing_radial": 3.20,
        "missed_gate_rate": 1.0,
    }
    tiny = {**parent, "terminal_crossing_radial": 3.15}
    useful = {**parent, "terminal_crossing_radial": 3.05}
    assert not MODULE.materially_improves(
        tiny, parent, 0.1, min_radial_improvement=0.10
    )
    assert MODULE.materially_improves(
        useful, parent, 0.1, min_radial_improvement=0.10
    )


def test_material_progress_accepts_discrete_gate_progress():
    parent = {
        "success_rate": 0.0,
        "crash_rate": 0.0,
        "gates_passed": 1.0,
        "terminal_crossing_radial": 3.20,
        "missed_gate_rate": 1.0,
    }
    child = {
        **parent,
        "success_rate": 0.1,
        "gates_passed": 1.1,
        "missed_gate_rate": 0.9,
    }
    assert MODULE.materially_improves(
        child, parent, 0.1, min_radial_improvement=0.10
    )


def test_refresh_pending_curriculum_preserves_accepted_prefix(tmp_path):
    accepted = final_stage("accepted")
    revised_accepted = {**accepted, "training_overrides": {"env.reward": "new"}}
    old_pending = final_stage("old_pending")
    new_pending = final_stage("new_pending")
    state = {
        "stages": [accepted, old_pending],
        "curriculum": "old.json",
        "training_defaults": {},
    }
    MODULE.refresh_pending_curriculum(
        state=state,
        stages=[revised_accepted, new_pending],
        defaults={"timesteps": 500_000},
        curriculum_path=tmp_path / "new.json",
        start_stage=1,
    )
    assert state["stages"] == [accepted, new_pending]
    assert state["curriculum_revisions"][-1]["from_stage_index"] == 1


def test_refresh_pending_curriculum_rejects_accepted_stage_change(tmp_path):
    accepted = final_stage("accepted")
    changed = final_stage("changed")
    state = {"stages": [accepted]}
    with pytest.raises(ValueError, match="already accepted"):
        MODULE.refresh_pending_curriculum(
            state=state,
            stages=[changed],
            defaults={},
            curriculum_path=tmp_path / "new.json",
            start_stage=1,
        )


def test_pending_curriculum_match_ignores_preserved_accepted_metadata():
    old_accepted = final_stage("accepted")
    new_accepted = {**old_accepted, "training_overrides": {"env.reward": "new"}}
    pending = final_stage("pending")

    assert MODULE.pending_curriculum_matches(
        [old_accepted, pending], [new_accepted, pending], 1
    )
    assert not MODULE.pending_curriculum_matches(
        [old_accepted, final_stage("old_pending")],
        [new_accepted, pending],
        1,
    )
