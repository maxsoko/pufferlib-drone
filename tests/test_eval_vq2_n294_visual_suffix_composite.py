from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts import eval_vq2_n294_visual_suffix_composite as composite


def test_source_locks_match_retained_artifacts():
    for path, expected in (
        (composite.N294_CHECKPOINT, composite.N294_SHA256),
        (composite.SUFFIX_CHECKPOINT, composite.SUFFIX_SHA256),
        (composite.SUFFIX_REPORT, composite.SUFFIX_REPORT_SHA256),
        (composite.PROMPT, composite.PROMPT_SHA256),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_first_step_reports_missing_and_reached_rows():
    values = np.asarray(
        [[0.0, 1.0 / 6.0, 2.0 / 6.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    assert composite._first_step(values, 2.0 / 6.0) == [2, -1]


def test_config_is_exact_and_teacher_free():
    try:
        from pufferlib import _C, pufferl
    except ImportError:
        pytest.skip("native Puffer binding is unavailable")
    if getattr(_C, "env_name", None) != composite.BACKEND_ENV_NAME:
        pytest.skip("drone_race_vision binding is not loaded")
    config, _ = composite.composite_config(
        pufferl, agents=8, seed=43001, camera_pitch_rad=0.0
    )
    environment = config["env"]
    assert config["vec"]["total_agents"] == 8
    assert environment["teacher_action_blend"] == 0.0
    assert environment["teacher_alignment_governor"] == 0
    assert environment["reset_position_noise_xy"] == 0.0
    assert environment["reset_position_noise_z"] == 0.0
    assert environment["sitl_plant_domain_randomize"] == 0
    assert environment["gate0_x"] == 10.78
    assert environment["gate1_y"] == 8.96


def test_config_couples_longitudinal_acceleration_and_braking():
    try:
        from pufferlib import _C, pufferl
    except ImportError:
        pytest.skip("native Puffer binding is unavailable")
    if getattr(_C, "env_name", None) != composite.BACKEND_ENV_NAME:
        pytest.skip("drone_race_vision binding is not loaded")
    config, _ = composite.composite_config(
        pufferl,
        agents=1,
        seed=43002,
        camera_pitch_rad=0.0,
        longitudinal_accel_scale=12.0,
    )
    assert config["env"]["sitl_horizontal_accel_scale"] == 12.0
    assert config["env"]["sitl_braking_accel_scale"] == 12.0


def test_config_rejects_invalid_longitudinal_scale():
    try:
        from pufferlib import _C, pufferl
    except ImportError:
        pytest.skip("native Puffer binding is unavailable")
    if getattr(_C, "env_name", None) != composite.BACKEND_ENV_NAME:
        pytest.skip("drone_race_vision binding is not loaded")
    with pytest.raises(ValueError, match="finite and positive"):
        composite.composite_config(
            pufferl,
            agents=1,
            seed=43002,
            camera_pitch_rad=0.0,
            longitudinal_accel_scale=0.0,
        )
