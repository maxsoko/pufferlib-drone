from __future__ import annotations

import scripts.eval_vq2_lc224_all24_fixed_camera_robustness as target


def test_lc224_source_lock_and_isolation_table() -> None:
    parent = target.verify_inputs()
    assert parent["model"]["sequence_length"] == 2_329


def test_lc224_perturbation_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, ["locked"])
    )
    config, overrides = target.perturbed_load_config(object())
    assert config["env"] == target.PERTURBATIONS
    assert not any("camera_roll_jitter" in name for name in target.PERTURBATIONS)
    assert not any("camera_pitch_jitter" in name for name in target.PERTURBATIONS)
    assert not any("camera_yaw_jitter" in name for name in target.PERTURBATIONS)
    assert overrides == ["locked"]
    assert target.GROUP_SIZE == 32
    assert target.ENV_SEED_INDEX_OFFSET == 287
