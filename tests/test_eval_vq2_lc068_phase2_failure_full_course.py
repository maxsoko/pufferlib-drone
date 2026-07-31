from __future__ import annotations

import torch

import scripts.eval_vq2_lc068_phase2_failure_full_course as lc068
from scripts.eval_vq2_variable_gate_oracle import sha256_path


def test_lc068_binds_confirmed_candidate_and_fresh_seed() -> None:
    parent = lc068.verify_inputs()
    assert lc068.SEED == 431680
    assert lc068.COEFFICIENT == (-0.01, 0.0, 0.0, 0.0)
    assert sha256_path(lc068.LC067_REPORT) == lc068.LC067_REPORT_SHA256
    assert parent["schema"] == "vq2_lc062_phase2_bias_full_course_checkpoint_v1"


def test_lc068_candidate_changes_only_phase2_decoder() -> None:
    parent = lc068.verify_inputs()["model_state"]
    candidate = lc068.build_selected_candidate_state(parent)
    changed = {
        name for name in parent
        if not torch.equal(parent[name], candidate[name])
    }
    assert changed == {
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    }
    for name in changed:
        keep = torch.arange(parent[name].shape[0]) != lc068.base.TARGET_PHASE
        assert torch.equal(parent[name][keep], candidate[name][keep])
