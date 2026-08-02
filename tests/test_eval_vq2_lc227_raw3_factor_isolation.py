from __future__ import annotations

import scripts.eval_vq2_lc227_raw3_factor_isolation as target


def test_lc227_source_lock() -> None:
    assert target.verify_inputs()["model"]["sequence_length"] == 2_329


def test_lc227_profiles_and_bound(monkeypatch) -> None:
    assert set(target.PROFILES) == {
        "control", "reset", "camera_dropout", "edge_dropout",
        "rolling_shutter", "reset_gain_hover", "reset_lag_drag",
    }
    monkeypatch.setattr(
        target, "BASE_LOAD_CONFIG", lambda *args, **kwargs: ({"env": {}}, [])
    )
    target.set_profile("reset_gain_hover")
    config, _ = target.profiled_load_config(object())
    assert config["env"] == target.PROFILES["reset_gain_hover"]
    assert target.GROUP_SIZE == 8
    assert target.TARGET_RAW_INDEX == 3
    assert target.MAX_STEPS == 10_000
