from __future__ import annotations

import scripts.eval_vq2_lc225_raw2_interaction_isolation as target


def test_lc225_source_lock() -> None:
    assert target.verify_inputs()["model"]["sequence_length"] == 2_329


def test_lc225_profiles_cover_group_interactions(monkeypatch) -> None:
    assert set(target.PROFILES) == {
        "perception", "plant", "reset_perception", "reset_plant",
        "perception_plant", "combined_no_camera",
    }
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, [])
    )
    target.set_profile("combined_no_camera")
    config, _ = target.profiled_load_config(object())
    assert config["env"] == {**target.RESET, **target.PERCEPTION, **target.PLANT}
    assert target.TARGET_RAW_INDEX == 2
    assert target.MAX_STEPS == 5_000
