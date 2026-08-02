from __future__ import annotations

import scripts.eval_vq2_lc226_raw3_interaction_isolation as target


def test_lc226_source_lock() -> None:
    assert target.verify_inputs()["model"]["sequence_length"] == 2_329


def test_lc226_profiles_and_bound(monkeypatch) -> None:
    assert target.PROFILES is target.base.PROFILES
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, [])
    )
    target.set_profile("combined_no_camera")
    config, _ = target.profiled_load_config(object())
    assert config["env"] == {
        **target.base.RESET, **target.base.PERCEPTION, **target.base.PLANT,
    }
    assert target.GROUP_SIZE == 8
    assert target.TARGET_RAW_INDEX == 3
    assert target.MAX_STEPS == 45_000
