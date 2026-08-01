from __future__ import annotations

import scripts.train_vq2_lc131_phase8_dense_return_ppo as lc131


def test_lc131_source_lock_and_repeated_config(monkeypatch) -> None:
    payload = lc131.verify_inputs()
    assert payload["numerically_admitted"]
    monkeypatch.setattr(
        lc131,
        "BASE_LOAD_CONFIG",
        lambda *args, **kwargs: ({"vec": {}}, ["--base"]),
    )
    config, overrides = lc131.repeated_load_config(object())
    assert config["vec"]["env_seed_group_size"] == 128
    assert overrides[-2:] == ["--vec.env-seed-group-size", "128"]


def test_lc131_configuration_restores_base_module() -> None:
    before = (
        lc131.base.TAG,
        lc131.base.TOTAL_AGENTS,
        lc131.base.TARGET_PHASE,
        lc131.base.PHASE_RETURN_ADVANTAGE_WEIGHT,
    )
    originals = lc131.configure()
    try:
        assert lc131.base.TAG == lc131.TAG
        assert lc131.base.TOTAL_AGENTS == 256
        assert lc131.base.TARGET_PHASE == 8
        assert lc131.base.PHASE_RETURN_ADVANTAGE_WEIGHT == 1.0
    finally:
        lc131.restore(originals)
    after = (
        lc131.base.TAG,
        lc131.base.TOTAL_AGENTS,
        lc131.base.TARGET_PHASE,
        lc131.base.PHASE_RETURN_ADVANTAGE_WEIGHT,
    )
    assert after == before
