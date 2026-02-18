import numpy as np

from pufferlib.environments.drone_race.environment import DroneRaceEnv


def _run_until_done(env, action, limit=256):
    info = None
    terminated = False
    truncated = False
    for _ in range(limit):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return terminated, truncated, info


def test_gate_order_validity_marks_out_of_order_run_invalid():
    env = DroneRaceEnv(
        gate_centers=np.array([[10.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        gate_normals=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_position=np.array([-1.05, 0.0, 1.0], dtype=np.float32),
        dt=0.1,
        max_speed_xy=2.0,
        velocity_response=1.0,
        linear_damping=0.0,
        init_position_jitter=0.0,
        init_velocity_jitter=0.0,
        init_yaw_jitter=0.0,
        crash_height=-1.0,
        pos_bound=100.0,
        randomize=False,
        curriculum=False,
    )
    env.reset(seed=0)

    terminated, truncated, info = _run_until_done(
        env,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        limit=32,
    )

    assert terminated
    assert not truncated
    assert info is not None
    assert info["valid_run"] == 0
    assert info["out_of_order"] == 1
    assert info["gates_passed"] == 0


def test_elapsed_time_is_monotonic_and_resets_on_reset():
    env = DroneRaceEnv(
        gate_centers=np.array([[100.0, 0.0, 1.0]], dtype=np.float32),
        gate_normals=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        start_position=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        dt=0.05,
        init_position_jitter=0.0,
        init_velocity_jitter=0.0,
        init_yaw_jitter=0.0,
        crash_height=-1.0,
        pos_bound=1000.0,
        randomize=False,
        curriculum=False,
    )
    env.reset(seed=0)

    elapsed = []
    for _ in range(3):
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        elapsed.append(info["elapsed_time"])

    assert np.allclose(elapsed, [0.05, 0.10, 0.15], atol=1e-6)

    env.reset(seed=1)
    assert np.isclose(env.elapsed_time, 0.0)

    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert np.isclose(info["elapsed_time"], 0.05)


def test_timeout_sets_truncated_without_termination():
    env = DroneRaceEnv(
        gate_centers=np.array([[100.0, 0.0, 1.0]], dtype=np.float32),
        gate_normals=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        start_position=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        dt=0.02,
        time_limit_seconds=0.04,
        max_steps=100,
        init_position_jitter=0.0,
        init_velocity_jitter=0.0,
        init_yaw_jitter=0.0,
        crash_height=-1.0,
        pos_bound=1000.0,
        randomize=False,
        curriculum=False,
    )
    env.reset(seed=0)

    terminated, truncated, info = _run_until_done(
        env,
        np.zeros(4, dtype=np.float32),
        limit=10,
    )

    assert not terminated
    assert truncated
    assert info is not None
    assert info["valid_run"] == 1


def test_scripted_baseline_completes_simple_course():
    env = DroneRaceEnv(
        num_gates=3,
        gate_spacing=3.0,
        gate_lateral_amplitude=0.0,
        gate_altitude=1.0,
        start_offset=1.0,
        dt=0.05,
        max_steps=300,
        time_limit_seconds=20.0,
        init_position_jitter=0.0,
        init_velocity_jitter=0.0,
        init_yaw_jitter=0.0,
        crash_height=-1.0,
        pos_bound=100.0,
        randomize=False,
        curriculum=False,
    )
    env.reset(seed=0)

    info = None
    terminated = False
    truncated = False
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(env.scripted_action())
        if terminated or truncated:
            break

    assert info is not None
    assert terminated
    assert not truncated
    assert info["valid_run"] == 1
    assert info["gates_passed"] == info["gates_total"]
