from __future__ import annotations

from pathlib import Path

import torch

import scripts.eval_vq2_lc124_phase6_success_rescue_milestone as lc124


def test_lc124_source_lock_and_candidate_contract() -> None:
    parent = lc124.verify_inputs()["model_state"]
    baseline = lc124.candidate_state_for_index(parent, 0)
    candidate = lc124.candidate_state_for_index(parent, 1)
    assert all(torch.equal(baseline[name], parent[name]) for name in parent)
    changed = {name for name in parent if not torch.equal(candidate[name], parent[name])}
    assert lc124.candidate_payload()["success_rescue_anchor"]["target_phase"] == 6
    assert changed == {
        "indexed_phase_residual_input",
        "indexed_phase_residual_input_bias",
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    }
    assert lc124.candidate_metadata_for_index(1)["endpoint_phases"] == [6]


def test_lc124_run_installs_callbacks_without_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_run(*, output: Path, device_name: str, resume: bool) -> dict[str, object]:
        seen["parent"] = lc124.base.verify_inputs()
        return {"diagnostic_valid": True}

    monkeypatch.setattr(lc124.base, "run", fake_run)
    report = lc124.run(output=tmp_path, device_name="cpu")
    assert report["diagnostic_valid"] is True
    assert isinstance(seen["parent"], dict)
