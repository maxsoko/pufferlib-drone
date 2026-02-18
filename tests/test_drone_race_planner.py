import numpy as np

from pufferlib.environments.drone_race.environment import DroneRaceEnv


def test_planner_output_respects_speed_and_yaw_limits():
    env = DroneRaceEnv(
        planner_speed_scale=0.9,
        randomize=False,
        curriculum=False,
    )
    env.reset(seed=0)

    # Force a large relative target to exercise clipping paths.
    env.perception_output = env.perception_output.__class__(
        schema_version=env.perception_output.schema_version,
        relative_gate_center_body=np.array([100.0, -100.0, 100.0], dtype=np.float32),
        gate_normal_body=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        confidence=1.0,
        valid=True,
        source=env.perception_output.source,
    )
    planner_output = env.planner.plan(env, env.perception_output)

    assert abs(planner_output.desired_velocity_body[0]) <= env.max_speed_xy + 1e-6
    assert abs(planner_output.desired_velocity_body[1]) <= env.max_speed_xy + 1e-6
    assert abs(planner_output.desired_velocity_body[2]) <= env.max_speed_z + 1e-6
    assert abs(planner_output.desired_yaw_rate) <= env.max_yaw_rate + 1e-6


def test_planner_fallback_activates_for_invalid_perception():
    env = DroneRaceEnv(
        perception_adapter="camera_sensor_stub",
        camera_dropout_prob=1.0,
        allow_perception_fallback=False,
        randomize=False,
        curriculum=False,
    )
    _, reset_info = env.reset(seed=0)

    assert reset_info["planner_fallback"] == 1
    assert env.last_planner_output.fallback_active

    _, _, _, _, step_info = env.step(np.zeros(4, dtype=np.float32))
    assert step_info["planner_fallback"] == 1


def test_planner_target_gate_index_is_monotonic_during_scripted_run():
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

    target_indices = []
    terminated = False
    truncated = False
    info = None
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(env.scripted_action())
        target_indices.append(info["planner_target_gate_index"])
        if terminated or truncated:
            break

    assert info is not None
    assert terminated and not truncated
    assert info["valid_run"] == 1
    assert target_indices == sorted(target_indices)
    assert target_indices[-1] == env.gates_total - 1
