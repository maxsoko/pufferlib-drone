from __future__ import annotations

import os
import subprocess
import sys

import scripts.compare_vq2_vg048_offset16_paired_count5 as vg048
import scripts.eval_vq2_staged_count5_component as component


ROOT = component.ROOT
PARENT_MANIFEST = (
    ROOT / "docs/vq2_vg048_parent_count5_manifest_2026-07-31.json"
)
CANDIDATE_MANIFEST = (
    ROOT / "docs/vq2_vg048_candidate_count5_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg048_vast.sh"


def test_vg048_contract_uses_distinct_offset16() -> None:
    assert vg048.EPISODE_OFFSET == 16
    assert (
        vg048.PRIOR_PARENT_BEHAVIOR_SHA256
        == "faa021347a723c9353249cc72b145cddb27ffef7d9a443fbe5ae12606d4cc69c"
    )
    assert (
        vg048.PRIOR_CANDIDATE_BEHAVIOR_SHA256
        == "b742e6029976d0fdeac091bcc87b2270ee8d00f40eb3b1ac32f3fcf11c3c07f2"
    )


def test_vg048_manifests_bind_offset16_identical_fixture() -> None:
    parent = component.load_manifest(PARENT_MANIFEST)
    candidate = component.load_manifest(CANDIDATE_MANIFEST)
    component.verify_actor_evidence(parent)
    component.verify_actor_evidence(candidate)
    for name in (
        "num_gates",
        "agents",
        "episodes",
        "num_threads",
        "max_steps",
        "seed",
        "episode_offset",
    ):
        assert parent[name] == candidate[name]
    assert parent["episode_offset"] == 16
    assert parent["agents"] == parent["episodes"] == 64


def test_vg048_wrapper_executes_directly() -> None:
    result = subprocess.run(
        [sys.executable, str(vg048.Path(vg048.__file__)), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vg048_runner_is_parallel_resumable_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "compare_vq2_vg048_offset16_paired_count5.py" in text
    assert "parent_pid=$!" in text
    assert "candidate_pid=$!" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
