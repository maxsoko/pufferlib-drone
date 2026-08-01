from __future__ import annotations

import scripts.collect_vq2_lc092_phase7_paired_seed_recapture as lc092


def test_lc092_exact_paired_mapping_contract() -> None:
    assert lc092.AGENTS == lc092.EPISODES == 256
    assert lc092.SEED_GROUP_SIZE == 128
    assert lc092.EXPECTED_QUERY_AGENTS == 6
    assert lc092.EXPECTED_SUCCESS_AGENTS == 4
    assert lc092.EXPECTED_FAILURE_AGENTS == 2


def test_lc092_inputs_are_source_locked() -> None:
    lc092.verify_inputs()


def test_lc092_sets_native_seed_group() -> None:
    class Dummy:  # pragma: no cover - only passed through the patched loader
        pass

    original = lc092.recapture.base.dagger_config
    try:
        lc092.recapture.base.dagger_config = lambda _: ({"vec": {}}, ["base"])
        config, overrides = lc092.paired_dagger_config(Dummy())
    finally:
        lc092.recapture.base.dagger_config = original
    assert config["vec"]["env_seed_group_size"] == 128
    assert overrides[-2:] == ["--vec.env-seed-group-size", "128"]
