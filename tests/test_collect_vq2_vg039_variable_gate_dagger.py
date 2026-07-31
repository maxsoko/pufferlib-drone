from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.collect_vq2_vg039_variable_gate_dagger as vg039


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "docs/vq2_vg039_variable_gate_dagger_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg039_vast.sh"


def test_vg039_manifest_binds_floor_only_rejection_and_fresh_seed() -> None:
    manifest = vg039.load_manifest(MANIFEST)
    assert manifest["seed"] == 429152
    assert manifest["minimum_gate3_rate"] == 0.20
    assert manifest["minimum_gate4_rate"] == 0.05
    assert manifest["previous_collection_rejection_sha256"].startswith(
        "7f91cb84"
    )
    required = (
        ROOT / manifest["oracle_report"],
        ROOT / manifest["checkpoint"],
        ROOT / manifest["screen_evidence"],
        ROOT / manifest["previous_collection_rejection"],
    )
    if not all(path.is_file() for path in required):
        pytest.skip("retained remote evidence is not mirrored in this checkout")
    vg039.verify_bound_inputs(manifest)


def test_vg039_runner_is_source_locked_bounded_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "vq2_vg039_variable_gate_dagger_round9_vg033_visited_512" in text
    assert text.count("OMP_NUM_THREADS=4 MKL_NUM_THREADS=1") >= 2
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
