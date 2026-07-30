from __future__ import annotations

import json
import os

import numpy as np

import scripts.recover_vq2_vg012_postcollection as recovery


def _passing_replay() -> dict[str, int | float]:
    return {
        "active_mismatches": 0,
        "mask_max_byte_error": 0,
        "tail_max_error": 0.0,
        "actor_recorded_action_max_error": 0.0,
        "executed_action_max_error": 0.0,
        "terminal_mismatches": 0,
        "episodes_replayed": 512,
        "replayed_actions": 171_903,
        "derived_terminal_actions": 512,
        "recorded_nonterminal_actions": 171_391,
        "stored_phase_records_match": True,
        "actor_nonfinite": 0,
        "actor_envelope_violations": 0,
    }


def test_vg013_json_ready_converts_numpy_booleans_strictly() -> None:
    payload = recovery.json_ready({"passed": np.bool_(True), "count": np.int64(3)})
    assert payload == {"passed": True, "count": 3}
    assert type(payload["passed"]) is bool
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_vg013_recovery_predicates_fail_closed() -> None:
    original = {"layout": True, "transport": True}
    predicates = recovery.recovery_predicates(
        replay=_passing_replay(), original_predicates=original
    )
    assert all(predicates.values())
    assert all(type(value) is bool for value in predicates.values())

    for field in (
        "active_mismatches",
        "mask_max_byte_error",
        "tail_max_error",
        "actor_recorded_action_max_error",
        "executed_action_max_error",
        "terminal_mismatches",
        "actor_nonfinite",
        "actor_envelope_violations",
    ):
        changed = _passing_replay()
        changed[field] = 1
        assert not all(recovery.recovery_predicates(
            replay=changed, original_predicates=original
        ).values())
    for field in (
        "episodes_replayed",
        "replayed_actions",
        "derived_terminal_actions",
        "recorded_nonterminal_actions",
    ):
        changed = _passing_replay()
        changed[field] = 0
        assert not all(recovery.recovery_predicates(
            replay=changed, original_predicates=original
        ).values())
    changed = _passing_replay()
    changed["stored_phase_records_match"] = False
    assert not all(recovery.recovery_predicates(
        replay=changed, original_predicates=original
    ).values())
    assert not all(recovery.recovery_predicates(
        replay=_passing_replay(), original_predicates={"layout": False}
    ).values())


def test_vg013_binds_exact_failed_vg012_artifacts() -> None:
    assert recovery.ORIGINAL_SOURCE_COMMIT == (
        "107963410014d8bb8009e1fb4ded2dd79adeab11"
    )
    assert recovery.ORIGINAL_METADATA_SHA256 == (
        "e7bfdba11d1a1c65187c820203bb3a675d51b93cdff9270498ce8280aa25caac"
    )
    assert recovery.ORIGINAL_STATE_SHA256 == (
        "75a89042b24c0d3be4e13c529ecf1119df24e07972f1cf06ae802970e6214520"
    )
    assert recovery.ORIGINAL_ARCHIVE_SHA256 == (
        "c2f868dcb401700d39119c4a5bfd4b94316fc9a73c7cd8ea329780223be506a0"
    )
    assert set(recovery.ORIGINAL_ARRAY_SHA256) == {
        "action.npy", "mask.npy", "tail.npy", "terminal.npy", "valid.npy"
    }


def test_vg013_runner_is_replay_only_source_locked_and_resumable() -> None:
    text = recovery.RUNNER.read_text()
    assert os.access(recovery.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert recovery.TAG in text
    assert "scripts/recover_vq2_vg012_postcollection.py" in text
    assert "tests/test_recover_vq2_vg012_postcollection.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
