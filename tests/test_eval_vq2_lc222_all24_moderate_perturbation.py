from __future__ import annotations

import scripts.eval_vq2_lc222_all24_moderate_perturbation as target


def test_lc222_source_lock_and_shadow_gate() -> None:
    parent = target.verify_inputs()
    assert parent["model"]["sequence_length"] == 2_329


def test_lc222_perturbation_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, ["locked"])
    )
    config, overrides = target.perturbed_load_config(object())
    assert all(config["env"][name] == value for name, value in target.PERTURBATIONS.items())
    assert overrides == ["locked"]
    assert target.GROUP_SIZE == 32
    assert target.ENV_SEED_INDEX_OFFSET == 143
