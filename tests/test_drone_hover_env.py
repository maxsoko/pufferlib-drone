import numpy as np

from pufferlib.environments.drone_hover.environment import DroneHoverEnv


def test_reset_observation_shape_and_bounds():
    env = DroneHoverEnv()
    obs, _ = env.reset(seed=123)
    assert obs.shape == (12,)
    assert obs.dtype == np.float32
    assert np.all(obs <= 1.0)
    assert np.all(obs >= -1.0)


def test_reset_determinism():
    env_a = DroneHoverEnv()
    env_b = DroneHoverEnv()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert np.allclose(obs_a, obs_b)


def test_out_of_bounds_termination():
    env = DroneHoverEnv()
    env.reset(seed=0)
    env.position = np.array([env.pos_bound + 1.0, 0.0, 1.0], dtype=np.float32)
    action = np.zeros(4, dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(action)
    assert terminated
    assert not truncated
