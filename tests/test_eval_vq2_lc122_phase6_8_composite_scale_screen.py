from __future__ import annotations

from pathlib import Path

import torch

import scripts.eval_vq2_lc122_phase6_8_composite_scale_screen as lc122


def test_lc122_source_lock_and_composite_contract() -> None:
    parent = lc122.verify_inputs()["model_state"]
    assert lc122.PHASE6_SCALE == 0.30
    assert lc122.PHASE8_SCALES == (0.0, 0.10, 0.30, 1.0)
    baseline = lc122.candidate_state_for_index(parent, 0)
    candidate = lc122.candidate_state_for_index(parent, 2)
    assert all(torch.equal(baseline[name], parent[name]) for name in lc122.PARAMETER_NAMES)
    for name in lc122.PARAMETER_NAMES:
        assert not torch.equal(candidate[name][6], parent[name][6])
        assert not torch.equal(candidate[name][8], parent[name][8])
        assert torch.equal(candidate[name][7], parent[name][7])
        assert torch.equal(candidate[name][9], parent[name][9])


def test_lc122_run_installs_callbacks_without_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_run(*, output: Path, device_name: str, resume: bool) -> dict[str, object]:
        seen["parent"] = lc122.base.verify_inputs()
        seen["output"] = output
        return {"diagnostic_valid": True}

    monkeypatch.setattr(lc122.base, "run", fake_run)
    result = lc122.run(output=tmp_path, device_name="cpu")
    assert result["diagnostic_valid"] is True
    assert seen["output"] == tmp_path
    assert isinstance(seen["parent"], dict)
