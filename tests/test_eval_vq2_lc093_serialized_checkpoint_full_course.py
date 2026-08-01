from __future__ import annotations

import torch

import scripts.eval_vq2_lc093_serialized_checkpoint_full_course as lc093
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256


def test_lc093_saved_checkpoint_contract() -> None:
    assert lc093.GROUP_SIZE == 128
    assert lc093.TARGET_RAW_INDEX == 7
    assert lc093.MINIMUM_TARGET_PASS_GAIN == 1
    assert lc093.MINIMUM_MEAN_GATE_GAIN == 1.0 / 128.0


def test_lc093_inputs_and_candidate_state_are_exact() -> None:
    parent = lc093.verify_inputs()
    selected = lc093.build_selected_candidate_state(parent["model_state"])
    candidate = torch.load(
        lc093.CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    assert state_sha256(selected) == state_sha256(candidate["model_state"])


def test_lc093_selection_requires_exact_progress_and_safety() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.38,
        "promotion_target_passes": 1, "crash_rate": 0.1,
        "maximum_raw_index": 7,
    }
    candidate = {
        "transport_pass": True, "mean_gates_passed": 3.40,
        "promotion_target_passes": 2, "crash_rate": 0.1,
        "maximum_raw_index": 8,
    }
    assert lc093.choose_candidate(parent, candidate)
    candidate["crash_rate"] = 0.11
    assert not lc093.choose_candidate(parent, candidate)
