from __future__ import annotations

import scripts.train_vq2_lc130_phase8_repeated_frontier_ppo as lc130


def test_lc130_source_lock_and_repeated_config(monkeypatch) -> None:
    payload = lc130.verify_inputs()
    assert payload["numerically_admitted"]
    monkeypatch.setattr(
        lc130,
        "BASE_LOAD_CONFIG",
        lambda *args, **kwargs: ({"vec": {}}, ["--base"]),
    )
    config, overrides = lc130.repeated_load_config(
        object(), num_gates=24, agents=lc130.TOTAL_AGENTS,
        episodes=lc130.TOTAL_AGENTS, seed=432050, threads=32,
    )
    assert config["vec"]["env_seed_group_size"] == lc130.SEED_GROUP_SIZE
    assert overrides[-2:] == ["--vec.env-seed-group-size", "128"]
    assert lc130.TOTAL_AGENTS // lc130.SEED_GROUP_SIZE == 4


def test_lc130_configuration_restores_base_module() -> None:
    before = (
        lc130.base.TAG, lc130.base.TOTAL_AGENTS,
        lc130.base.TARGET_PHASE, lc130.base.milestone.load_actor,
    )
    originals = lc130.configure()
    try:
        assert lc130.base.TAG == lc130.TAG
        assert lc130.base.TOTAL_AGENTS == 512
        assert lc130.base.TARGET_PHASE == 8
        assert lc130.base.milestone.load_actor is lc130.split_actor_loader
    finally:
        lc130.restore(originals)
    after = (
        lc130.base.TAG, lc130.base.TOTAL_AGENTS,
        lc130.base.TARGET_PHASE, lc130.base.milestone.load_actor,
    )
    assert after == before
