from __future__ import annotations

from pathlib import Path

import torch

import scripts.eval_vq2_lc128_phase6_onpolicy_ppo_milestone as lc128


def test_lc128_source_lock_and_candidate_contract() -> None:
    parent = lc128.verify_inputs()["model_state"]
    baseline = lc128.candidate_state_for_index(parent, 0)
    candidate = lc128.candidate_state_for_index(parent, 1)
    assert all(torch.equal(baseline[name], parent[name]) for name in parent)
    changed = {name for name in parent if not torch.equal(candidate[name], parent[name])}
    assert changed == set(lc128.PARAMETER_NAMES)
    assert lc128.candidate_metadata_for_index(1)["endpoint_phases"] == [6]


def test_lc128_run_installs_callbacks_without_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_run(*, output: Path, device_name: str, resume: bool) -> dict[str, object]:
        seen["parent"] = lc128.base.verify_inputs()
        return {"diagnostic_valid": True}

    monkeypatch.setattr(lc128.base, "run", fake_run)
    report = lc128.run(output=tmp_path, device_name="cpu")
    assert report["diagnostic_valid"] is True
    assert isinstance(seen["parent"], dict)
