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


def test_curriculum_progress_accumulates_across_resets():
    env = DroneHoverEnv(curriculum=True, curriculum_episodes=10)
    progresses = []
    for seed in range(3):
        env.reset(seed=seed)
        progresses.append(env._curriculum_progress())

    assert progresses[0] < progresses[1] < progresses[2]
    assert np.isclose(progresses[0], 0.1)
    assert np.isclose(progresses[2], 0.3)


def test_hover_curriculum_target_seconds_accumulates():
    env = DroneHoverEnv(
        hover_curriculum=True,
        hover_curriculum_min_seconds=2.0,
        hover_curriculum_episodes=4,
        success_hold_seconds=30.0,
    )
    hold_seconds = []
    for seed in range(5):
        env.reset(seed=seed)
        hold_seconds.append(env._current_success_hold_seconds())

    assert hold_seconds[0] < hold_seconds[1] < hold_seconds[2] < hold_seconds[3]
    assert np.isclose(hold_seconds[3], 30.0)
    assert np.isclose(hold_seconds[4], 30.0)


def test_pd_assist_anneal_scale_accumulates():
    env = DroneHoverEnv(
        pd_assist=True,
        pd_assist_scale=0.3,
        pd_assist_scale_final=0.0,
        pd_assist_anneal_episodes=3,
    )
    scales = []
    for seed in range(5):
        env.reset(seed=seed)
        scales.append(env._current_pd_assist_scale())

    assert scales[0] > scales[1] > scales[2]
    assert np.isclose(scales[2], 0.0)
    assert np.isclose(scales[3], 0.0)
    assert np.isclose(scales[4], 0.0)
