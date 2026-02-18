import numpy as np
import pytest

from pufferlib.environments.drone_race.contracts import SCHEMA_VERSION
from pufferlib.environments.drone_race.environment import DroneRaceEnv


def test_unknown_perception_adapter_raises_value_error():
    with pytest.raises(ValueError, match="Unknown perception adapter"):
        DroneRaceEnv(perception_adapter="not_a_real_adapter")


def test_privileged_adapter_reports_valid_perception_info():
    env = DroneRaceEnv(
        perception_adapter="privileged_state",
        randomize=False,
        curriculum=False,
    )
    _, reset_info = env.reset(seed=0)

    assert reset_info["perception_source"] == "privileged_state"
    assert reset_info["perception_valid"] == 1
    assert np.isclose(reset_info["perception_confidence"], 1.0)
    assert reset_info["perception_schema_version"] == SCHEMA_VERSION

    _, _, _, _, step_info = env.step(np.zeros(4, dtype=np.float32))
    assert step_info["perception_source"] == "privileged_state"
    assert step_info["perception_valid"] == 1
    assert np.isclose(step_info["perception_confidence"], 1.0)


def test_camera_sensor_stub_reports_invalid_when_dropout_without_fallback():
    env = DroneRaceEnv(
        perception_adapter="camera_sensor_stub",
        camera_dropout_prob=1.0,
        allow_perception_fallback=False,
        randomize=False,
        curriculum=False,
    )
    _, reset_info = env.reset(seed=0)

    assert reset_info["perception_source"] == "camera_sensor_stub"
    assert reset_info["perception_valid"] == 0
    assert np.isclose(reset_info["perception_confidence"], 0.0)
    assert reset_info["perception_schema_version"] == SCHEMA_VERSION

    obs, _, _, _, step_info = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (22,)
    assert step_info["perception_source"] == "camera_sensor_stub"
    assert step_info["perception_valid"] == 0
