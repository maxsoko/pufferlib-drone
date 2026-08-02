from __future__ import annotations

import scripts.eval_vq2_lc223_raw2_perturbation_isolation as target


def test_lc223_source_lock() -> None:
    assert target.verify_inputs()["model"]["sequence_length"] == 2_329


def test_lc223_profiles_are_complete_and_isolated(monkeypatch) -> None:
    assert set(target.PROFILES) == {
        "control", "reset", "camera_jitter", "camera_dropout", "edge_dropout",
        "rolling_shutter", "plant_gain_hover", "plant_lag_drag", "combined",
    }
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, [])
    )
    target.set_profile("camera_jitter")
    config, _ = target.profiled_load_config(object())
    assert config["env"] == target.PROFILES["camera_jitter"]
    assert target.TARGET_RAW_INDEX == 2
    assert target.MAX_STEPS == 5_000
