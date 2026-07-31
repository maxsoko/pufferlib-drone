from __future__ import annotations

import os

import scripts.eval_vq2_staged_count5_component as component


ROOT = component.ROOT
PARENT_MANIFEST = ROOT / "docs/vq2_vg041_parent_count5_manifest_2026-07-31.json"
CANDIDATE_MANIFEST = (
    ROOT / "docs/vq2_vg041_candidate_count5_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg041_vast.sh"


def test_vg041_manifests_bind_same_fresh_count5_fixture() -> None:
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
    ):
        assert parent[name] == candidate[name]
    assert parent["role"] == "parent"
    assert candidate["role"] == "candidate"
    assert parent["checkpoint_sha256"] != candidate["checkpoint_sha256"]
    assert parent["num_gates"] == 5
    assert parent["seed"] == 429154
    assert parent["agents"] == parent["episodes"] == 32
    assert parent["num_threads"] == 4
    assert parent["max_steps"] == 2560


def test_vg041_runner_is_parallel_resumable_source_locked_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "eval_vq2_staged_count5_component.py" in text
    assert "compare_vq2_staged_count5_diagnostic.py" in text
    assert "--num-gates 5" in text
    assert "tests/test_vq2_vg041_staged_count5_diagnostic.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "parent_pid=$!" in text
    assert "candidate_pid=$!" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
