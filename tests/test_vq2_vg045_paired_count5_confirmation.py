from __future__ import annotations

import os

import scripts.compare_vq2_vg045_paired_count5_confirmation as comparator
import scripts.eval_vq2_staged_count5_component as component


ROOT = component.ROOT
PARENT_MANIFEST = (
    ROOT / "docs/vq2_vg045_parent_count5_manifest_2026-07-31.json"
)
CANDIDATE_MANIFEST = (
    ROOT / "docs/vq2_vg045_candidate_count5_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg045_vast.sh"


def _summary(
    *,
    reach: tuple[int, int, int],
    crashes: int,
) -> dict[str, object]:
    return {
        "episodes": 64,
        "seed": 429159,
        "hard_transport_pass": True,
        "gate_reach": {
            "1": reach[0],
            "2": reach[1],
            "3": reach[2],
            "4": 0,
            "5": 0,
        },
        "crashes": crashes,
        "successes": 0,
    }


def test_vg045_confirmation_requires_gate2_and_crash_preservation() -> None:
    parent = _summary(reach=(64, 61, 10), crashes=4)
    candidate = _summary(reach=(64, 62, 12), crashes=4)
    assert comparator.confirmation_passes(parent, candidate)
    candidate["gate_reach"]["2"] = 60
    assert not comparator.confirmation_passes(parent, candidate)
    candidate["gate_reach"]["2"] = 62
    candidate["crashes"] = 5
    assert not comparator.confirmation_passes(parent, candidate)


def test_vg045_manifests_bind_fresh_identical_fixture() -> None:
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
    assert parent["seed"] == 429159
    assert parent["agents"] == parent["episodes"] == 64
    assert parent["checkpoint_sha256"] != candidate["checkpoint_sha256"]


def test_vg045_runner_is_parallel_resumable_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "eval_vq2_staged_count5_component.py" in text
    assert "compare_vq2_vg045_paired_count5_confirmation.py" in text
    assert "parent_pid=$!" in text
    assert "candidate_pid=$!" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
