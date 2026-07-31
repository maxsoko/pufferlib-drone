from __future__ import annotations

import os
import subprocess
import sys

import scripts.compare_vq2_vg050_offset24_paired_count5 as vg050
import scripts.eval_vq2_staged_count5_component as component


ROOT = component.ROOT
PARENT = ROOT / "docs/vq2_vg050_parent_count5_manifest_2026-07-31.json"
CANDIDATE = (
    ROOT / "docs/vq2_vg050_candidate_count5_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg050_vast.sh"


def test_vg050_manifests_bind_offset24_fixture() -> None:
    parent = component.load_manifest(PARENT)
    candidate = component.load_manifest(CANDIDATE)
    component.verify_actor_evidence(parent)
    component.verify_actor_evidence(candidate)
    for key in (
        "num_gates",
        "agents",
        "episodes",
        "num_threads",
        "max_steps",
        "seed",
        "episode_offset",
    ):
        assert parent[key] == candidate[key]
    assert parent["episode_offset"] == 24


def test_vg050_wrapper_executes_directly() -> None:
    result = subprocess.run(
        [sys.executable, str(vg050.Path(vg050.__file__)), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vg050_runner_is_parallel_resumable_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert "parent_pid=$!" in text
    assert "candidate_pid=$!" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
