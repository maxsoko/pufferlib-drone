from __future__ import annotations

import os
from pathlib import Path

import scripts.eval_vq2_candidate_count6_multi_offset as screen


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/vq2_vg054_count6_multi_offset_manifest_2026-07-31.json"
RUNNER = ROOT / "scripts/run_vq2_vg054_vast.sh"


def test_vg054_manifest_binds_vg053_and_vg051() -> None:
    manifest = screen.load_manifest(MANIFEST)
    assert manifest["tag"] == "vq2_vg054_vg053_count6_multi_offset_001"
    assert manifest["checkpoint_sha256"].startswith("f0a9812f")
    assert manifest["parent_baseline_sha256"].startswith("88e56a5b")
    assert manifest["offsets"] == [0, 8, 16, 24]
    screen.verify_inputs(manifest)


def test_vg054_runner_is_resumable_source_locked_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "--resume" in text
    assert text.index('VQ2_OUTPUT="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
