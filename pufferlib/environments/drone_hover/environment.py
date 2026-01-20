import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Box

import pufferlib
import pufferlib.emulation


def env_creator(name='drone_hover'):
    return functools.partial(make, name)


def make(name='drone_hover', render_mode=None, buf=None, seed=0, **kwargs):
    env = DroneHoverEnv(render_mode=render_mode, **kwargs)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def _rotation_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # ZYX rotation
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


class DroneHoverEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt=0.02,
        mass=1.0,
        gravity=9.81,
        arm_length=0.2,
        max_thrust_per_motor=15.0,
        linear_damping=0.05,
        angular_damping=0.02,
        randomize=False,
        mass_range=0.05,
        inertia_range=0.05,
        wind_std=0.2,
        sensor_noise_std=0.01,
        curriculum=False,
        curriculum_episodes=200,
        action_centered=False,
        action_centered_scale=0.3,
        pd_assist=False,
        pd_label=False,
        pd_assist_scale=0.3,
        pd_assist_scale_final=0.0,
        pd_assist_anneal_episodes=0,
        pd_kp_xy=0.7,
        pd_kd_xy=0.5,
        pd_kp_z=5.0,
        pd_kd_z=3.0,
        pd_kp_ang=8.0,
        pd_kd_ang=2.0,
        pd_kp_yaw=0.5,
        pd_kd_yaw=0.1,
        residual_penalty=0.0,
        action_smoothing=0.0,
        hover_bonus=0.0,
        hover_bonus_pos_tol=0.1,
        hover_bonus_angle_tol=10.0,
        success_bonus=0.0,
        hover_curriculum=False,
        hover_curriculum_min_seconds=2.0,
        hover_curriculum_episodes=1000,
        target_z=1.0,
        pos_bound=5.0,
        max_tilt_deg=45.0,
        hover_tolerance=0.25,
        hover_tolerance_deg=15.0,
        success_hold_seconds=30.0,
        max_steps=None,
        init_pos_range=0.1,
        init_vel_range=0.05,
        init_angle_deg=5.0,
        init_ang_vel=0.05,
        w_pos=2.0,
        w_vel=0.1,
        w_ang=0.5,
        w_angvel=0.05,
        w_act=0.01,
        w_act_delta=0.05,
        render_mode=None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.dt = float(dt)
        self.base_mass = float(mass)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.arm_length = float(arm_length)
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.linear_damping = float(linear_damping)
        self.angular_damping = float(angular_damping)

        self.randomize = bool(randomize)
        self.mass_range = float(mass_range)
        self.inertia_range = float(inertia_range)
        self.wind_std = float(wind_std)
        self.sensor_noise_std = float(sensor_noise_std)
        self.curriculum = bool(curriculum)
        self.curriculum_episodes = int(curriculum_episodes)
        self.action_centered = bool(action_centered)
        self.action_centered_scale = float(action_centered_scale)
        self.pd_assist = bool(pd_assist)
        self.pd_label = bool(pd_label)
        self.pd_assist_scale = float(pd_assist_scale)
        self.pd_assist_scale_final = float(pd_assist_scale_final)
        self.pd_assist_anneal_episodes = int(pd_assist_anneal_episodes)
        self.pd_kp_xy = float(pd_kp_xy)
        self.pd_kd_xy = float(pd_kd_xy)
        self.pd_kp_z = float(pd_kp_z)
        self.pd_kd_z = float(pd_kd_z)
        self.pd_kp_ang = float(pd_kp_ang)
        self.pd_kd_ang = float(pd_kd_ang)
        self.pd_kp_yaw = float(pd_kp_yaw)
        self.pd_kd_yaw = float(pd_kd_yaw)
        self.residual_penalty = float(residual_penalty)
        self.action_smoothing = float(action_smoothing)
        self.hover_bonus = float(hover_bonus)
        self.hover_bonus_pos_tol = float(hover_bonus_pos_tol)
        self.hover_bonus_angle_tol = np.deg2rad(hover_bonus_angle_tol)
        self.success_bonus = float(success_bonus)
        self.hover_curriculum = bool(hover_curriculum)
        self.hover_curriculum_min_seconds = float(hover_curriculum_min_seconds)
        self.hover_curriculum_episodes = int(hover_curriculum_episodes)
        self.wind = np.zeros(3, dtype=np.float32)

        self.target = np.array([0.0, 0.0, float(target_z)], dtype=np.float32)
        self.pos_bound = float(pos_bound)
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.hover_tolerance = float(hover_tolerance)
        self.hover_tolerance_angle = np.deg2rad(hover_tolerance_deg)
        self.success_hold_seconds = float(success_hold_seconds)
        self.max_steps = int(max_steps) if max_steps is not None else int(self.success_hold_seconds / self.dt)

        self.init_pos_range = float(init_pos_range)
        self.init_vel_range = float(init_vel_range)
        self.init_angle = np.deg2rad(init_angle_deg)
        self.init_ang_vel = float(init_ang_vel)

        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_ang = float(w_ang)
        self.w_angvel = float(w_angvel)
        self.w_act = float(w_act)
        self.w_act_delta = float(w_act_delta)

        self.base_inertia = np.array([0.02, 0.02, 0.04], dtype=np.float32)
        self.inertia = self.base_inertia.copy()
        self.yaw_torque_coeff = 0.02

        self.obs_scale = np.array([
            self.pos_bound, self.pos_bound, self.pos_bound,
            5.0, 5.0, 5.0,
            np.pi, np.pi, np.pi,
            10.0, 10.0, 10.0,
        ], dtype=np.float32)

        self.observation_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self._rng = None
        self._reset_state()

    def _reset_state(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.angles = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.hover_steps = 0
        self.hover_time = 0.0
        self.step_count = 0
        self.episode_count = 0
        self.hover_episode_count = 0

    def _sample_uniform(self, low, high, size):
        return self._rng.uniform(low=low, high=high, size=size).astype(np.float32)

    def _get_obs(self):
        pos_error = self.position - self.target
        obs = np.concatenate([
            pos_error,
            self.velocity,
            self.angles,
            self.angular_velocity,
        ], axis=0)
        noise_std = self._current_sensor_noise_std()
        if noise_std > 0.0:
            obs = obs + self._rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)
        obs = obs / self.obs_scale
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _curriculum_progress(self):
        if not self.curriculum:
            return 1.0
        if self.curriculum_episodes <= 0:
            return 1.0
        return min(1.0, self.episode_count / self.curriculum_episodes)

    def _current_mass_range(self):
        if not self.randomize and not self.curriculum:
            return 0.0
        return self.mass_range * self._curriculum_progress()

    def _current_inertia_range(self):
        if not self.randomize and not self.curriculum:
            return 0.0
        return self.inertia_range * self._curriculum_progress()

    def _current_wind_std(self):
        if not self.randomize and not self.curriculum:
            return 0.0
        return self.wind_std * self._curriculum_progress()

    def _current_sensor_noise_std(self):
        if not self.randomize and not self.curriculum:
            return 0.0
        return self.sensor_noise_std * self._curriculum_progress()

    def _current_success_hold_seconds(self):
        if not self.hover_curriculum:
            return self.success_hold_seconds
        min_seconds = min(self.hover_curriculum_min_seconds, self.success_hold_seconds)
        if self.hover_curriculum_episodes <= 0:
            progress = 1.0
        else:
            progress = min(1.0, self.hover_episode_count / self.hover_curriculum_episodes)
        return min_seconds + progress * (self.success_hold_seconds - min_seconds)

    def _current_pd_assist_scale(self):
        if self.pd_assist_anneal_episodes <= 0:
            return self.pd_assist_scale
        progress = min(1.0, self.episode_count / self.pd_assist_anneal_episodes)
        return self.pd_assist_scale + progress * (self.pd_assist_scale_final - self.pd_assist_scale)

    def _pd_base_action(self):
        pos_error = self.position - self.target
        vel = self.velocity
        desired_acc = np.array([
            -self.pd_kp_xy * pos_error[0] - self.pd_kd_xy * vel[0],
            -self.pd_kp_xy * pos_error[1] - self.pd_kd_xy * vel[1],
            -self.pd_kp_z * pos_error[2] - self.pd_kd_z * vel[2],
        ], dtype=np.float32)
        thrust_vec = desired_acc + np.array([0.0, 0.0, self.gravity], dtype=np.float32)
        thrust_z = float(max(thrust_vec[2], 1e-3))
        desired_roll = np.clip(np.arctan2(-thrust_vec[1], thrust_z), -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(np.arctan2(thrust_vec[0], thrust_z), -self.max_tilt, self.max_tilt)
        roll_error = desired_roll - self.angles[0]
        pitch_error = desired_pitch - self.angles[1]
        yaw_error = -self.angles[2]

        roll_torque = self.pd_kp_ang * roll_error - self.pd_kd_ang * self.angular_velocity[0]
        pitch_torque = self.pd_kp_ang * pitch_error - self.pd_kd_ang * self.angular_velocity[1]
        yaw_torque = self.pd_kp_yaw * yaw_error - self.pd_kd_yaw * self.angular_velocity[2]

        total_thrust = np.clip(self.mass * float(np.linalg.norm(thrust_vec)), 0.0, 4.0 * self.max_thrust_per_motor)
        mix = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ], dtype=np.float32)
        b = np.array([
            total_thrust,
            roll_torque / max(self.arm_length, 1e-6),
            pitch_torque / max(self.arm_length, 1e-6),
            yaw_torque / max(self.yaw_torque_coeff, 1e-6),
        ], dtype=np.float32)
        motor_thrusts = np.linalg.solve(mix, b)
        return np.clip(motor_thrusts / self.max_thrust_per_motor, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.episode_count += 1
        self.hover_episode_count += 1

        if self.randomize or self.curriculum:
            mass_range = self._current_mass_range()
            inertia_range = self._current_inertia_range()
            wind_std = self._current_wind_std()
            mass_scale = self._rng.uniform(1.0 - mass_range, 1.0 + mass_range)
            inertia_scale = self._rng.uniform(1.0 - inertia_range, 1.0 + inertia_range, size=3)
            self.mass = float(self.base_mass * mass_scale)
            self.inertia = (self.base_inertia * inertia_scale).astype(np.float32)
            self.wind = self._rng.normal(0.0, wind_std, size=3).astype(np.float32)
            self.wind[2] = 0.0
        else:
            self.mass = float(self.base_mass)
            self.inertia = self.base_inertia.copy()
            self.wind = np.zeros(3, dtype=np.float32)

        self.position = self.target + self._sample_uniform(-self.init_pos_range, self.init_pos_range, 3)
        self.velocity = self._sample_uniform(-self.init_vel_range, self.init_vel_range, 3)
        self.angles = self._sample_uniform(-self.init_angle, self.init_angle, 3)
        self.angular_velocity = self._sample_uniform(-self.init_ang_vel, self.init_ang_vel, 3)

        hover_thrust = (self.mass * self.gravity) / (4.0 * self.max_thrust_per_motor)
        self.last_action = np.full(4, hover_thrust, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        pd_base_action = None
        if self.pd_assist or self.pd_label:
            pd_base_action = self._pd_base_action()
        if self.pd_assist:
            scale = np.clip(self._current_pd_assist_scale(), 0.0, 1.0)
            residual = (action - 0.5) * 2.0 * scale
            action = np.clip(pd_base_action + residual, 0.0, 1.0)
        elif self.action_centered:
            hover_thrust = (self.mass * self.gravity) / (4.0 * self.max_thrust_per_motor)
            scale = np.clip(self.action_centered_scale, 0.0, 1.0)
            centered = (action - 0.5) * 2.0 * scale
            action = np.clip(hover_thrust + centered, 0.0, 1.0)
        if self.action_smoothing > 0.0:
            alpha = np.clip(self.action_smoothing, 0.0, 1.0)
            action = (1.0 - alpha) * self.last_action + alpha * action

        thrusts = action * self.max_thrust_per_motor
        total_thrust = float(np.sum(thrusts))

        # Motor layout: [front_left, front_right, rear_right, rear_left]
        f_fl, f_fr, f_rr, f_rl = thrusts
        roll_torque = self.arm_length * ((f_fr + f_rr) - (f_fl + f_rl))
        pitch_torque = self.arm_length * ((f_fl + f_fr) - (f_rr + f_rl))
        yaw_torque = self.yaw_torque_coeff * (f_fl - f_fr + f_rr - f_rl)
        torque = np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float32)

        rotation = _rotation_matrix(self.angles[0], self.angles[1], self.angles[2])
        thrust_world = rotation @ np.array([0.0, 0.0, total_thrust], dtype=np.float32)
        acceleration = (thrust_world / self.mass) - np.array([0.0, 0.0, self.gravity], dtype=np.float32)
        acceleration -= self.linear_damping * self.velocity
        acceleration += self.wind

        angular_acc = (torque - self.angular_damping * self.angular_velocity) / self.inertia

        self.velocity = self.velocity + acceleration * self.dt
        self.position = self.position + self.velocity * self.dt
        self.angular_velocity = self.angular_velocity + angular_acc * self.dt
        self.angles = self.angles + self.angular_velocity * self.dt

        self.step_count += 1

        pos_error = self.position - self.target
        pos_error_norm = float(np.linalg.norm(pos_error))
        vel_norm = float(np.linalg.norm(self.velocity))
        angle_error = np.array([self.angles[0], self.angles[1], 0.1 * self.angles[2]], dtype=np.float32)
        ang_error_norm = float(np.linalg.norm(angle_error))
        ang_vel_norm = float(np.linalg.norm(self.angular_velocity))
        act_norm = float(np.linalg.norm(action))
        act_delta_norm = float(np.linalg.norm(action - self.last_action))

        reward = (
            -self.w_pos * pos_error_norm ** 2
            -self.w_vel * vel_norm ** 2
            -self.w_ang * ang_error_norm ** 2
            -self.w_angvel * ang_vel_norm ** 2
            -self.w_act * act_norm ** 2
            -self.w_act_delta * act_delta_norm ** 2
        )
        if self.residual_penalty > 0.0 and pd_base_action is not None:
            residual_norm = float(np.linalg.norm(action - pd_base_action))
            reward -= self.residual_penalty * residual_norm ** 2

        within_pos = pos_error_norm <= self.hover_tolerance
        within_angle = (abs(self.angles[0]) <= self.hover_tolerance_angle and abs(self.angles[1]) <= self.hover_tolerance_angle)
        within_bonus_pos = pos_error_norm <= self.hover_bonus_pos_tol
        within_bonus_angle = (abs(self.angles[0]) <= self.hover_bonus_angle_tol and abs(self.angles[1]) <= self.hover_bonus_angle_tol)
        if self.hover_bonus > 0.0 and within_bonus_pos and within_bonus_angle:
            reward += self.hover_bonus
        if within_pos and within_angle:
            self.hover_steps += 1
        else:
            self.hover_steps = 0
        self.hover_time = self.hover_steps * self.dt

        out_of_bounds = (
            abs(self.position[0]) > self.pos_bound
            or abs(self.position[1]) > self.pos_bound
            or self.position[2] < 0.0
        )
        too_tilted = (abs(self.angles[0]) > self.max_tilt or abs(self.angles[1]) > self.max_tilt)
        hold_seconds = self._current_success_hold_seconds()
        hold_steps = int(np.ceil(hold_seconds / self.dt - 1e-9))
        success = self.hover_steps >= hold_steps
        if success and self.success_bonus > 0.0:
            reward += self.success_bonus

        terminated = bool(out_of_bounds or too_tilted or success)
        truncated = self.step_count >= self.max_steps

        info = {
            "pos_error": pos_error_norm,
            "angle_error": ang_error_norm,
            "hover_time": self.hover_time,
            "hover_target_seconds": float(hold_seconds),
            "success": float(success),
            "mass": float(self.mass),
            "wind_norm": float(np.linalg.norm(self.wind)),
            "curriculum_progress": float(self._curriculum_progress()),
        }
        if pd_base_action is not None:
            info["pd_base_action"] = pd_base_action

        self.last_action = action

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
