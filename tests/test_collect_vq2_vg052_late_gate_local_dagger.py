from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.collect_vq2_vg052_late_gate_local_dagger as vg052


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/vq2_vg052_late_gate_local_dagger_manifest_2026-07-31.json"
RUNNER = ROOT / "scripts/run_vq2_vg052_vast.sh"


def test_vg052_manifest_and_environment_are_late_gate_only() -> None:
    manifest = vg052.load_manifest(MANIFEST)
    environment = vg052.late_gate_environment(manifest)
    assert environment["num_gates"] == 6
    assert environment["num_gates_per_env_randomize"] == 0
    assert environment["gate_local_start_curriculum"] == 1
    assert environment["gate_local_start_probability"] == 1.0
    assert environment["gate_local_start_gate_min"] == 3
    assert environment["gate_local_start_gate_max_exclusive"] == 5
    assert environment["gate_local_start_offset_min"] == 2.0
    assert environment["gate_local_start_offset_max"] == 5.0
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_roll_until_gate_index"] == 0
    assert environment["w_action_teacher"] == 0.0
    assert environment["observable_gate_index_denominator"] == 16.0


def test_vg052_binds_parent_baseline_and_oracle() -> None:
    manifest = vg052.load_manifest(MANIFEST)
    required = (
        ROOT / manifest["checkpoint"],
        ROOT / manifest["train_report"],
        ROOT / manifest["candidate_admission"],
        ROOT / manifest["six_gate_baseline"],
        ROOT / manifest["six_gate_report"],
        ROOT / manifest["oracle_report"],
    )
    if not all(path.is_file() for path in required):
        pytest.skip("retained remote evidence is not mirrored in this checkout")
    vg052.verify_bound_inputs(manifest)


def test_vg052_runner_is_bounded_source_locked_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "vq2_vg052_late_gate_local_dagger_vg033_visited_512" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
