import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Box

import pufferlib
import pufferlib.emulation
from .adapters import make_perception_adapter
from .planner import make_planner
from .gate_progress import is_valid_gate_crossing


def env_creator(name='drone_race'):
    return functools.partial(make, name)


def make(name='drone_race', render_mode=None, buf=None, seed=0, **kwargs):
    env = DroneRaceEnv(render_mode=render_mode, **kwargs)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError('Gate normal vector must be non-zero')
    return v / norm


def _yaw_rotation(yaw):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class DroneRaceEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt=0.02,
        num_gates=4,
        gate_spacing=5.0,
        gate_radius=1.0,
        gate_altitude=1.0,
        gate_lateral_amplitude=1.5,
        gate_centers=None,
        gate_normals=None,
        start_offset=2.0,
        start_position=None,
        max_speed_xy=6.0,
        max_speed_z=3.0,
        max_yaw_rate=2.0,
        velocity_response=0.35,
        linear_damping=0.05,
        crash_height=0.0,
        pos_bound=20.0,
        plane_cross_tolerance=1e-3,
        miss_tolerance=0.5,
        direction_min=0.05,
        strict_missed_gate=True,
        max_steps=1500,
        time_limit_seconds=30.0,
        init_position_jitter=0.1,
        init_velocity_jitter=0.1,
        init_yaw_jitter=0.2,
        randomize=False,
        curriculum=False,
        curriculum_episodes=1000,
        wind_std=0.3,
        wind_std_final=0.3,
        w_progress=20.0,
        w_gate=3.0,
        w_finish=30.0,
        w_time=1.0,
        w_ctrl=0.01,
        invalid_penalty=40.0,
        perception_adapter='privileged_state',
        camera_noise_std=0.0,
        camera_dropout_prob=0.0,
        allow_perception_fallback=True,
        planner='gate_setpoint',
        planner_speed_scale=0.85,
        planner_forward_gain=0.8,
        planner_lateral_gain=1.2,
        planner_vertical_gain=1.0,
        planner_yaw_gain=1.5,
        planner_smoothing=1.0,
        planner_confidence_threshold=0.25,
        planner_fallback_forward_scale=0.15,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.dt = float(dt)
        self.max_speed_xy = float(max_speed_xy)
        self.max_speed_z = float(max_speed_z)
        self.max_yaw_rate = float(max_yaw_rate)
        self.velocity_response = float(np.clip(velocity_response, 0.0, 1.0))
        self.linear_damping = float(max(linear_damping, 0.0))
        self.crash_height = float(crash_height)
        self.pos_bound = float(pos_bound)
        self.plane_cross_tolerance = float(max(plane_cross_tolerance, 0.0))
        self.miss_tolerance = float(max(miss_tolerance, 0.0))
        self.direction_min = float(direction_min)
        self.strict_missed_gate = bool(strict_missed_gate)
        self.max_steps = int(max_steps)
        self.time_limit_seconds = float(time_limit_seconds)
        self.init_position_jitter = float(max(init_position_jitter, 0.0))
        self.init_velocity_jitter = float(max(init_velocity_jitter, 0.0))
        self.init_yaw_jitter = float(max(init_yaw_jitter, 0.0))
        self.randomize = bool(randomize)
        self.curriculum = bool(curriculum)
        self.curriculum_episodes = int(curriculum_episodes)
        self.wind_std = float(max(wind_std, 0.0))
        self.wind_std_final = float(max(wind_std_final, 0.0))

        self.w_progress = float(w_progress)
        self.w_gate = float(w_gate)
        self.w_finish = float(w_finish)
        self.w_time = float(w_time)
        self.w_ctrl = float(w_ctrl)
        self.invalid_penalty = float(invalid_penalty)
        self.perception_adapter_name = str(perception_adapter)
        self.camera_noise_std = float(max(camera_noise_std, 0.0))
        self.camera_dropout_prob = float(np.clip(camera_dropout_prob, 0.0, 1.0))
        self.allow_perception_fallback = bool(allow_perception_fallback)
        self.planner_name = str(planner)

        self.gate_radius = float(gate_radius)
        self._build_course(
            num_gates=num_gates,
            gate_spacing=gate_spacing,
            gate_altitude=gate_altitude,
            gate_lateral_amplitude=gate_lateral_amplitude,
            gate_centers=gate_centers,
            gate_normals=gate_normals,
        )
        self.start_offset = float(start_offset)
        self.start_position = None if start_position is None else np.asarray(start_position, dtype=np.float32)

        self.observation_space = Box(low=-1.0, high=1.0, shape=(22,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._rng = None
        self.perception = make_perception_adapter(
            self.perception_adapter_name,
            camera_noise_std=self.camera_noise_std,
            camera_dropout_prob=self.camera_dropout_prob,
            allow_perception_fallback=self.allow_perception_fallback,
        )
        self.planner = make_planner(
            self.planner_name,
            speed_scale=planner_speed_scale,
            forward_gain=planner_forward_gain,
            lateral_gain=planner_lateral_gain,
            vertical_gain=planner_vertical_gain,
            yaw_gain=planner_yaw_gain,
            smoothing=planner_smoothing,
            confidence_threshold=planner_confidence_threshold,
            fallback_forward_scale=planner_fallback_forward_scale,
        )
        self.episode_count = 0
        self._reset_state()

    def _build_course(
        self,
        num_gates,
        gate_spacing,
        gate_altitude,
        gate_lateral_amplitude,
        gate_centers,
        gate_normals,
    ):
        if gate_centers is not None:
            centers = np.asarray(gate_centers, dtype=np.float32)
        else:
            centers = []
            for gate_idx in range(int(num_gates)):
                y = gate_lateral_amplitude if gate_idx % 2 == 0 else -gate_lateral_amplitude
                centers.append([gate_idx * gate_spacing, y, gate_altitude])
            centers = np.asarray(centers, dtype=np.float32)

        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            raise ValueError('gate_centers must be shaped [num_gates, 3] with at least one gate')

        if gate_normals is None:
            normals = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(centers), 1))
        else:
            normals = np.asarray(gate_normals, dtype=np.float32)
            if normals.shape != centers.shape:
                raise ValueError('gate_normals must have same shape as gate_centers')

        normals = np.stack([_normalize(n) for n in normals], axis=0).astype(np.float32)

        self.gate_centers = centers
        self.gate_normals = normals
        self.gates_total = int(len(centers))

        self._between_gate_distance = np.zeros(self.gates_total, dtype=np.float32)
        for i in range(self.gates_total - 1):
            self._between_gate_distance[i] = float(np.linalg.norm(centers[i + 1] - centers[i]))

        self._obs_scale = np.array(
            [
                self.pos_bound,
                self.pos_bound,
                max(2.0, self.pos_bound),
                self.max_speed_xy,
                self.max_speed_xy,
                self.max_speed_z,
                np.pi,
                np.pi,
                np.pi,
                self.max_yaw_rate,
                self.max_yaw_rate,
                self.max_yaw_rate,
                max(self.pos_bound, 5.0),
                max(self.pos_bound, 5.0),
                max(self.pos_bound, 5.0),
                1.0,
                1.0,
                1.0,
                max(self.gate_radius, 1.0),
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def _reset_state(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.angles = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.elapsed_time = 0.0
        self.current_gate_index = 0
        self.valid_run = True
        self.out_of_order = False
        self.missed_gate = False
        self.crash = False
        self.completion_time = None
        self.progress = 0.0
        self.wind = np.zeros(3, dtype=np.float32)
        self._initial_remaining_distance = 1.0
        self.perception_output = None
        self.last_planner_output = None

    def _curriculum_progress(self):
        if not self.curriculum:
            return 1.0
        if self.curriculum_episodes <= 0:
            return 1.0
        return min(1.0, self.episode_count / self.curriculum_episodes)

    def _current_wind_std(self):
        if not self.randomize and not self.curriculum:
            return 0.0
        progress = self._curriculum_progress()
        return self.wind_std + (self.wind_std_final - self.wind_std) * progress

    def _remaining_distance(self, position, gate_index):
        if gate_index >= self.gates_total:
            return 0.0

        remaining = float(np.linalg.norm(self.gate_centers[gate_index] - position))
        if gate_index < self.gates_total - 1:
            remaining += float(np.sum(self._between_gate_distance[gate_index:self.gates_total - 1]))
        return remaining

    def _compute_progress(self):
        if self.current_gate_index >= self.gates_total:
            return 1.0
        remaining = self._remaining_distance(self.position, self.current_gate_index)
        denom = max(self._initial_remaining_distance, 1e-6)
        return float(np.clip(1.0 - remaining / denom, 0.0, 1.0))

    def _relative_gate_pose_body(self):
        if self.current_gate_index >= self.gates_total:
            gate_center = self.gate_centers[-1]
            gate_normal = self.gate_normals[-1]
        else:
            gate_center = self.gate_centers[self.current_gate_index]
            gate_normal = self.gate_normals[self.current_gate_index]

        world_to_body = _yaw_rotation(self.angles[2]).T
        rel_center_body = world_to_body @ (gate_center - self.position)
        normal_body = world_to_body @ gate_normal
        return rel_center_body, normal_body

    def _observation(self):
        if self.perception_output is None:
            self.perception_output = self.perception.observe(self)

        rel_center_body = self.perception_output.relative_gate_center_body
        normal_body = self.perception_output.gate_normal_body
        remaining = self._remaining_distance(self.position, self.current_gate_index)
        remaining_norm = remaining / max(self._initial_remaining_distance, 1e-6)
        gate_index_norm = self.current_gate_index / max(self.gates_total, 1)
        elapsed_norm = self.elapsed_time / max(self.time_limit_seconds, self.dt)

        obs = np.concatenate(
            [
                self.position,
                self.velocity,
                self.angles,
                self.angular_velocity,
                rel_center_body,
                normal_body,
                np.array(
                    [
                        self.gate_radius,
                        gate_index_norm,
                        elapsed_norm,
                        remaining_norm,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        obs = obs / self._obs_scale
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _segment_crosses_gate(self, gate_index, prev_pos, curr_pos):
        return is_valid_gate_crossing(
            prev_position=prev_pos,
            position=curr_pos,
            gate_center=self.gate_centers[gate_index],
            gate_normal=self.gate_normals[gate_index],
            gate_radius=self.gate_radius,
            plane_cross_tolerance=self.plane_cross_tolerance,
            direction_min=self.direction_min,
        )

    def _update_gate_state(self, prev_pos, curr_pos):
        gate_passed = False
        if self.current_gate_index >= self.gates_total:
            return gate_passed

        if self._segment_crosses_gate(self.current_gate_index, prev_pos, curr_pos):
            self.current_gate_index += 1
            gate_passed = True

        if self.current_gate_index < self.gates_total:
            center = self.gate_centers[self.current_gate_index]
            normal = self.gate_normals[self.current_gate_index]
            d_curr = float(np.dot(curr_pos - center, normal))
            if d_curr > self.miss_tolerance and not gate_passed and self.strict_missed_gate:
                self.missed_gate = True
                self.valid_run = False

        for idx in range(self.gates_total):
            if idx < self.current_gate_index or idx == self.current_gate_index:
                continue
            if self._segment_crosses_gate(idx, prev_pos, curr_pos):
                self.out_of_order = True
                self.valid_run = False
                break

        return gate_passed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.episode_count += 1
        self.planner.reset()

        if self.start_position is not None:
            start = self.start_position.copy()
        else:
            start = self.gate_centers[0] - self.start_offset * self.gate_normals[0]
        if self.init_position_jitter > 0.0:
            start += self._rng.uniform(
                low=-self.init_position_jitter,
                high=self.init_position_jitter,
                size=3,
            ).astype(np.float32)

        self.position = start.astype(np.float32)
        if self.init_velocity_jitter > 0.0:
            self.velocity = self._rng.uniform(
                low=-self.init_velocity_jitter,
                high=self.init_velocity_jitter,
                size=3,
            ).astype(np.float32)
        else:
            self.velocity = np.zeros(3, dtype=np.float32)

        self.angles = np.zeros(3, dtype=np.float32)
        if self.init_yaw_jitter > 0.0:
            self.angles[2] = float(self._rng.uniform(-self.init_yaw_jitter, self.init_yaw_jitter))

        wind_std = self._current_wind_std()
        if wind_std > 0.0:
            self.wind = self._rng.normal(0.0, wind_std, size=3).astype(np.float32)
            self.wind[2] = 0.0

        self._initial_remaining_distance = self._remaining_distance(self.position, self.current_gate_index)
        self.progress = self._compute_progress()
        self.perception_output = self.perception.observe(self)
        self.last_planner_output = self.planner.plan(self, self.perception_output)
        info = {
            "perception_source": self.perception_output.source,
            "perception_confidence": float(self.perception_output.confidence),
            "perception_valid": int(bool(self.perception_output.valid)),
            "perception_schema_version": int(self.perception_output.schema_version),
            "planner_source": self.last_planner_output.source,
            "planner_fallback": int(bool(self.last_planner_output.fallback_active)),
            "planner_schema_version": int(self.last_planner_output.schema_version),
            "planner_target_gate_index": int(self.last_planner_output.target_gate_index),
            "planner_lookahead_gates": int(self.last_planner_output.lookahead_gates),
            "planner_setpoint_vx": float(self.last_planner_output.desired_velocity_body[0]),
            "planner_setpoint_vy": float(self.last_planner_output.desired_velocity_body[1]),
            "planner_setpoint_vz": float(self.last_planner_output.desired_velocity_body[2]),
            "planner_setpoint_yaw_rate": float(self.last_planner_output.desired_yaw_rate),
        }
        return self._observation(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_pos = self.position.copy()
        prev_progress = self.progress

        body_vel_target = np.array(
            [
                action[0] * self.max_speed_xy,
                action[1] * self.max_speed_xy,
                action[2] * self.max_speed_z,
            ],
            dtype=np.float32,
        )
        world_vel_target = _yaw_rotation(self.angles[2]) @ body_vel_target
        self.velocity = (
            (1.0 - self.velocity_response) * self.velocity
            + self.velocity_response * world_vel_target
        )
        self.velocity -= self.linear_damping * self.velocity * self.dt
        self.velocity += self.wind * self.dt

        yaw_rate = float(action[3]) * self.max_yaw_rate
        self.angular_velocity[:] = np.array([0.0, 0.0, yaw_rate], dtype=np.float32)
        self.angles[2] = float(self.angles[2] + yaw_rate * self.dt)
        self.position = self.position + self.velocity * self.dt

        self.step_count += 1
        self.elapsed_time += self.dt

        gate_passed = self._update_gate_state(prev_pos, self.position)

        self.crash = bool(
            self.position[2] <= self.crash_height
            or abs(self.position[0]) > self.pos_bound
            or abs(self.position[1]) > self.pos_bound
            or self.position[2] > self.pos_bound
        )
        if self.crash:
            self.valid_run = False

        success = self.valid_run and self.current_gate_index >= self.gates_total
        if success:
            self.completion_time = float(self.elapsed_time)

        terminated = bool(success or not self.valid_run or self.crash)
        truncated = bool(
            not terminated
            and (
                self.step_count >= self.max_steps
                or self.elapsed_time >= self.time_limit_seconds
            )
        )

        self.progress = self._compute_progress()
        delta_progress = self.progress - prev_progress
        reward = (
            self.w_progress * delta_progress
            + (self.w_gate if gate_passed else 0.0)
            + (self.w_finish if success else 0.0)
            - self.w_time * self.dt
            - self.w_ctrl * float(np.dot(action, action))
        )
        if terminated and not success:
            reward -= self.invalid_penalty

        self.perception_output = self.perception.observe(self)
        self.last_planner_output = self.planner.plan(self, self.perception_output)

        info = {
            "elapsed_time": float(self.elapsed_time),
            "gate_index": int(self.current_gate_index),
            "gates_passed": int(self.current_gate_index),
            "gates_total": int(self.gates_total),
            "valid_run": int(bool(self.valid_run)),
            "completion_time": None if self.completion_time is None else float(self.completion_time),
            "crash": int(bool(self.crash)),
            "out_of_order": int(bool(self.out_of_order)),
            "missed_gate": int(bool(self.missed_gate)),
            "progress": float(self.progress),
            "perception_source": self.perception_output.source,
            "perception_confidence": float(self.perception_output.confidence),
            "perception_valid": int(bool(self.perception_output.valid)),
            "perception_schema_version": int(self.perception_output.schema_version),
            "planner_source": self.last_planner_output.source,
            "planner_fallback": int(bool(self.last_planner_output.fallback_active)),
            "planner_schema_version": int(self.last_planner_output.schema_version),
            "planner_target_gate_index": int(self.last_planner_output.target_gate_index),
            "planner_lookahead_gates": int(self.last_planner_output.lookahead_gates),
            "planner_setpoint_vx": float(self.last_planner_output.desired_velocity_body[0]),
            "planner_setpoint_vy": float(self.last_planner_output.desired_velocity_body[1]),
            "planner_setpoint_vz": float(self.last_planner_output.desired_velocity_body[2]),
            "planner_setpoint_yaw_rate": float(self.last_planner_output.desired_yaw_rate),
        }

        self.last_action = action
        return self._observation(), float(reward), terminated, truncated, info

    def scripted_action(
        self,
    ):
        if self.perception_output is None:
            self.perception_output = self.perception.observe(self)
        self.last_planner_output = self.planner.plan(self, self.perception_output)

        max_xy = max(self.max_speed_xy, 1e-6)
        max_z = max(self.max_speed_z, 1e-6)
        action = np.array(
            [
                self.last_planner_output.desired_velocity_body[0] / max_xy,
                self.last_planner_output.desired_velocity_body[1] / max_xy,
                self.last_planner_output.desired_velocity_body[2] / max_z,
                self.last_planner_output.desired_yaw_rate / max(self.max_yaw_rate, 1e-6),
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)

    def render(self):
        return None

    def close(self):
        return None
