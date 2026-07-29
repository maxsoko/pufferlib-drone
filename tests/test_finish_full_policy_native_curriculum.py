from pathlib import Path

from scripts.finish_full_policy_native_curriculum import (
    accepted_target_parent,
    build_promotion_command,
    promotion_passes,
)


def test_accepted_target_parent_requires_complete_target_state(tmp_path: Path):
    checkpoint = tmp_path / "policy.bin"
    checkpoint.write_bytes(b"weights")
    complete = {
        "status": "complete",
        "retained_radius": 0.75,
        "retained_checkpoint": str(checkpoint),
    }
    assert accepted_target_parent(complete, 0.75) == checkpoint.resolve()
    assert accepted_target_parent({**complete, "status": "running"}, 0.75) is None
    assert accepted_target_parent({**complete, "retained_radius": 1.5}, 0.75) is None
    assert accepted_target_parent({**complete, "retained_radius": 0.5}, 0.75) is None


def test_promotion_requires_thresholds_and_episode_count():
    passing = {
        "success_rate": 0.91,
        "crash_rate": 0.09,
        "gates_passed": 3.95,
        "episodes": 4096.0,
    }
    assert promotion_passes(
        passing, success_threshold=0.9, crash_limit=0.1, required_episodes=4096
    )
    assert not promotion_passes(
        {**passing, "episodes": 4095.0},
        success_threshold=0.9,
        crash_limit=0.1,
        required_episodes=4096,
    )
    assert not promotion_passes(
        {**passing, "episodes": 4097.0},
        success_threshold=0.9,
        crash_limit=0.1,
        required_episodes=4096,
    )
    assert not promotion_passes(
        {**passing, "success_rate": 0.89},
        success_threshold=0.9,
        crash_limit=0.1,
        required_episodes=4096,
    )
    assert not promotion_passes(
        {**passing, "out_of_order_rate": 0.001},
        success_threshold=0.9,
        crash_limit=0.1,
        required_episodes=4096,
    )


def test_promotion_does_not_add_undocumented_average_gate_threshold():
    assert promotion_passes(
        {
            "success_rate": 0.90,
            "crash_rate": 0.0,
            "gates_passed": 3.6,
            "episodes": 4096.0,
            "out_of_order_rate": 0.0,
        },
        success_threshold=0.9,
        crash_limit=0.1,
        required_episodes=4096,
    )


def test_promotion_command_locks_exact_true_aperture_geometry(tmp_path: Path):
    command = build_promotion_command(
        checkpoint=tmp_path / "policy.bin",
        target_radius=0.75,
        promotion_episodes=4096,
        promotion_json=tmp_path / "promotion.json",
        promotion_csv=tmp_path / "promotion.csv",
    )
    pairs = dict(zip(command[3::2], command[4::2]))
    assert pairs["--eval-episodes"] == "4096"
    assert pairs["--max-rollouts"] == "160"
    assert pairs["--env.gate-radius"] == "0.75"
    assert pairs["--env.gate2-radius"] == "0.75"
    assert pairs["--env.gate3-radius"] == "0.75"
    assert "--require-exact-episodes" in command
