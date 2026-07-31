from __future__ import annotations

import os

import scripts.compare_vq2_vg046_offset8_paired_count5 as comparator
import scripts.eval_vq2_staged_count5_component as component


ROOT = component.ROOT
PARENT_MANIFEST = (
    ROOT / "docs/vq2_vg046_parent_count5_manifest_2026-07-31.json"
)
CANDIDATE_MANIFEST = (
    ROOT / "docs/vq2_vg046_candidate_count5_manifest_2026-07-31.json"
)
RUNNER = ROOT / "scripts/run_vq2_vg046_vast.sh"


def _summary(
    *,
    reach: tuple[int, int, int],
    crashes: int,
) -> dict[str, object]:
    return {
        "episodes": 64,
        "seed": 429160,
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


def test_vg046_requires_offset_distinctness_and_rollout_gain() -> None:
    parent = _summary(reach=(64, 61, 10), crashes=4)
    candidate = _summary(reach=(64, 62, 12), crashes=4)
    kwargs = {
        "parent_offset_pass": True,
        "candidate_offset_pass": True,
        "parent_distinct": True,
        "candidate_distinct": True,
    }
    assert comparator.confirmation_passes(parent, candidate, **kwargs)
    assert not comparator.confirmation_passes(
        parent,
        candidate,
        **(kwargs | {"candidate_distinct": False}),
    )
    candidate["gate_reach"]["2"] = 60
    assert not comparator.confirmation_passes(parent, candidate, **kwargs)


def test_vg046_offset_parser_is_exact() -> None:
    report = {
        "loader_overrides": [
            "--seed",
            "429160",
            "--env.evaluation-episode-offset",
            "8",
        ]
    }
    assert comparator.report_uses_offset(report, 8)
    assert not comparator.report_uses_offset(report, 7)


def test_vg046_manifests_bind_offset8_identical_fixture() -> None:
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
    assert parent["role"] == "parent"
    assert candidate["role"] == "candidate"
    assert parent["episode_offset"] == 8
    assert parent["agents"] == parent["episodes"] == 64


def test_vg046_component_applies_offset8_to_native_config() -> None:
    manifest = component.load_manifest(PARENT_MANIFEST)
    changed_names = (
        "TAG",
        "SCHEMA",
        "COUNTS",
        "AGENTS",
        "EPISODES_PER_COUNT",
        "TOTAL_EPISODES",
        "SEEDS",
        "MINIMUM_SUCCESS_RATE",
        "MAX_EXECUTED_ACTION_ERROR",
        "REQUIRE_ZERO_CROSSING_MARGIN",
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
        "PREREGISTRATION",
        "RUNNER",
        "EXTRA_SOURCE_PATHS",
        "teacher_free_config",
    )
    original = {
        name: getattr(component.evaluator, name) for name in changed_names
    }

    def base_config(_module, *, num_gates):
        assert num_gates == 5
        return {"vec": {}, "env": {}}, []

    component.evaluator.teacher_free_config = base_config
    try:
        component.configure_evaluator(PARENT_MANIFEST, manifest)
        config, overrides = component.evaluator.teacher_free_config(
            None, num_gates=5
        )
        assert config["env"]["evaluation_episode_offset"] == 8
        index = overrides.index("--env.evaluation-episode-offset")
        assert overrides[index + 1] == "8"
    finally:
        for name, value in original.items():
            setattr(component.evaluator, name, value)


def test_vg046_runner_is_parallel_resumable_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "eval_vq2_staged_count5_component.py" in text
    assert "compare_vq2_vg046_offset8_paired_count5.py" in text
    assert "parent_pid=$!" in text
    assert "candidate_pid=$!" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
