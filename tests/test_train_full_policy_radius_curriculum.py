from pathlib import Path

import numpy as np
import pytest

from scripts.train_full_policy_radius_curriculum import (
    cli_overrides,
    evaluation_run_id,
    parse_override,
    radius_schedule,
    refined_radius_schedule,
    resume_source,
    resume_radius_schedule,
    select_new_checkpoint,
    success_regressed,
    write_checkpoint_with_log_std,
)


def test_radius_schedule_is_inclusive_without_float_drift():
    assert radius_schedule(13.25, 13.0, 0.125) == [13.25, 13.125, 13.0]
    assert radius_schedule(1.6, 1.5, 0.125) == [1.6, 1.5]


def test_radius_schedule_rejects_invalid_direction_and_step():
    with pytest.raises(ValueError):
        radius_schedule(1.0, 1.5, 0.1)
    with pytest.raises(ValueError):
        radius_schedule(2.0, 1.5, 0.0)


def test_refined_radius_schedule_replaces_failed_tail_with_half_steps():
    refined = refined_radius_schedule(9.25, 9.125, 9.0, 0.03125)
    assert refined == ([9.1875, 9.125, 9.0625, 9.0], 0.0625)


def test_refined_radius_schedule_stops_at_configured_floor():
    assert refined_radius_schedule(9.25, 9.1875, 9.0, 0.0625) is None


def test_refined_radius_schedule_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        refined_radius_schedule(9.25, 9.25, 9.0, 0.01)
    with pytest.raises(ValueError):
        refined_radius_schedule(9.25, 9.125, 9.0, 0.0)


def test_override_conversion_preserves_puffer_cli_keys():
    assert parse_override("env.gate2-radius=2.0") == ("env.gate2-radius", "2.0")
    assert cli_overrides({"env.gate2-radius": "2.0"}) == ["--env.gate2-radius", "2.0"]


def test_evaluation_run_id_keeps_revisited_radius_reports_immutable():
    assert evaluation_run_id("1784011865881", 43) == "1784011865881_a0043"
    assert evaluation_run_id("1784011865881", 48) == "1784011865881_a0048"
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluation_run_id("1784011865881", -1)


def test_success_regression_stops_bad_child_chaining():
    baseline = {"success_rate": 0.8996}
    assert success_regressed(baseline, {"success_rate": 0.8363}, 0.03)
    assert not success_regressed(baseline, {"success_rate": 0.8916}, 0.03)
    with pytest.raises(ValueError, match="cannot be negative"):
        success_regressed(baseline, {"success_rate": 0.9}, -0.01)


def test_resume_source_uses_only_last_accepted_parent(tmp_path: Path):
    checkpoint = tmp_path / "accepted.bin"
    checkpoint.write_bytes(b"weights")
    source, radius = resume_source(
        {"retained_checkpoint": str(checkpoint), "retained_radius": 12.25}
    )
    assert source == checkpoint.resolve()
    assert radius == 12.25


def test_resume_source_rejects_state_without_accepted_radius():
    with pytest.raises(ValueError):
        resume_source({"retained_checkpoint": "candidate.bin", "retained_radius": None})


def test_resume_schedule_retries_interrupted_refined_radius(tmp_path: Path):
    checkpoint = tmp_path / "accepted.bin"
    checkpoint.write_bytes(b"weights")
    state = {
        "retained_checkpoint": str(checkpoint),
        "retained_radius": 4.25,
        "target_radius": 0.75,
        "schedule": [4.75, 4.5, 4.25, 4.0, 4.125, 4.0, 3.875, 0.75],
        "schedule_refinements": [
            {
                "accepted_radius": 4.25,
                "failed_radius": 4.0,
                "refined_step": 0.125,
            }
        ],
        "attempts": [{"radius": 4.125, "mode": "screen"}],
    }
    schedule, target = resume_radius_schedule(state)
    assert schedule == [4.25, 4.125, 4.0, 3.875, 0.75]
    assert target == 0.75


def test_resume_schedule_uses_refinement_after_failed_jump(tmp_path: Path):
    checkpoint = tmp_path / "accepted.bin"
    checkpoint.write_bytes(b"weights")
    state = {
        "retained_checkpoint": str(checkpoint),
        "retained_radius": 4.25,
        "schedule": [4.75, 4.5, 4.25, 4.0, 4.125, 4.0, 0.75],
        "schedule_refinements": [
            {
                "accepted_radius": 4.25,
                "failed_radius": 4.0,
                "refined_step": 0.125,
            }
        ],
        "attempts": [{"radius": 4.0, "mode": "fine_tune", "accepted": False}],
    }
    schedule, target = resume_radius_schedule(state)
    assert schedule == [4.25, 4.125, 4.0, 0.75]
    assert target == 0.75


def test_resume_schedule_rejects_target_change(tmp_path: Path):
    checkpoint = tmp_path / "accepted.bin"
    checkpoint.write_bytes(b"weights")
    state = {
        "retained_checkpoint": str(checkpoint),
        "retained_radius": 4.25,
        "target_radius": 0.75,
        "schedule": [4.25, 4.0, 0.75],
    }
    with pytest.raises(ValueError, match="does not match"):
        resume_radius_schedule(state, 1.5)


def test_select_new_checkpoint_ignores_existing_files(tmp_path: Path):
    old = tmp_path / "old" / "0000000001000000.bin"
    old.parent.mkdir()
    old.write_bytes(b"old")
    before = {old.resolve()}

    run = tmp_path / "new"
    run.mkdir()
    first = run / "0000000000032768.bin"
    final = run / "0000000001998848.bin"
    first.write_bytes(b"first")
    final.write_bytes(b"final")

    assert select_new_checkpoint(before, tmp_path) == final.resolve()


def test_write_checkpoint_with_log_std_changes_only_exploration_values(tmp_path: Path):
    input_dim = 3
    hidden_dim = 4
    num_actions = 2
    encoder_end = (hidden_dim * input_dim + 7) & ~7
    decoder_end = (encoder_end + (num_actions + 1) * hidden_dim + 7) & ~7
    weights = np.arange(decoder_end + num_actions + 9, dtype=np.float32)
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    weights.tofile(source)

    before = write_checkpoint_with_log_std(
        source,
        output,
        -4.0,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    )
    changed = np.fromfile(output, dtype=np.float32)

    assert before == weights[decoder_end : decoder_end + num_actions].tolist()
    np.testing.assert_array_equal(changed[:decoder_end], weights[:decoder_end])
    np.testing.assert_array_equal(
        changed[decoder_end : decoder_end + num_actions],
        np.full(num_actions, -4.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(changed[decoder_end + num_actions :], weights[decoder_end + num_actions :])


def test_write_checkpoint_with_log_std_rejects_invalid_inputs(tmp_path: Path):
    source = tmp_path / "short.bin"
    np.zeros(4, dtype=np.float32).tofile(source)
    with pytest.raises(ValueError, match="finite"):
        write_checkpoint_with_log_std(source, tmp_path / "bad.bin", float("nan"))
    with pytest.raises(ValueError, match="ended before"):
        write_checkpoint_with_log_std(source, tmp_path / "bad.bin", -4.0)
