import numpy as np

from .contracts import PlannerOutput, SCHEMA_VERSION


class Planner:
    name = "base"

    def reset(self):  # pragma: no cover - interface only
        raise NotImplementedError

    def plan(self, env, perception_output):  # pragma: no cover - interface only
        raise NotImplementedError


class GateSetpointPlanner(Planner):
    name = "gate_setpoint"

    def __init__(
        self,
        speed_scale=0.85,
        forward_gain=0.8,
        lateral_gain=1.2,
        vertical_gain=1.0,
        yaw_gain=1.5,
        smoothing=1.0,
        confidence_threshold=0.25,
        fallback_forward_scale=0.15,
    ):
        self.speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        self.forward_gain = float(max(forward_gain, 0.0))
        self.lateral_gain = float(max(lateral_gain, 0.0))
        self.vertical_gain = float(max(vertical_gain, 0.0))
        self.yaw_gain = float(max(yaw_gain, 0.0))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        self.fallback_forward_scale = float(max(fallback_forward_scale, 0.0))
        self._prev_velocity = np.zeros(3, dtype=np.float32)

    def reset(self):
        self._prev_velocity[:] = 0.0

    def _smooth(self, desired_velocity):
        smoothed = (1.0 - self.smoothing) * self._prev_velocity + self.smoothing * desired_velocity
        self._prev_velocity = smoothed.astype(np.float32)
        return self._prev_velocity

    def plan(self, env, perception_output):
        max_xy = max(float(env.max_speed_xy), 1e-6)
        max_z = max(float(env.max_speed_z), 1e-6)

        fallback_active = not bool(perception_output.valid) or float(perception_output.confidence) < self.confidence_threshold

        if fallback_active:
            desired_velocity = np.array(
                [self.fallback_forward_scale * max_xy, 0.0, 0.0],
                dtype=np.float32,
            )
            desired_yaw_rate = 0.0
        else:
            rel_center_body = perception_output.relative_gate_center_body.astype(np.float32)
            gate_normal_body = perception_output.gate_normal_body.astype(np.float32)

            target_xy = self.speed_scale * max_xy
            forward_target = float(rel_center_body[0])
            if abs(forward_target) < env.gate_radius:
                # Keep a small forward bias near the gate plane so we don't stall before crossing.
                forward_target += 0.5 * env.gate_radius

            vx_cmd = np.clip(self.forward_gain * forward_target, -target_xy, target_xy)
            vy_cmd = np.clip(self.lateral_gain * float(rel_center_body[1]), -target_xy, target_xy)
            vz_cmd = np.clip(
                self.vertical_gain * float(rel_center_body[2]),
                -self.speed_scale * max_z,
                self.speed_scale * max_z,
            )

            yaw_error = float(np.arctan2(gate_normal_body[1], gate_normal_body[0]))
            desired_yaw_rate = float(
                np.clip(self.yaw_gain * yaw_error, -float(env.max_yaw_rate), float(env.max_yaw_rate))
            )
            desired_velocity = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=np.float32)

        desired_velocity = self._smooth(desired_velocity)
        return PlannerOutput(
            schema_version=SCHEMA_VERSION,
            desired_velocity_body=desired_velocity,
            desired_yaw_rate=float(desired_yaw_rate),
            target_gate_index=int(min(env.current_gate_index, env.gates_total - 1)),
            lookahead_gates=1,
            fallback_active=fallback_active,
            source=self.name,
        )


def make_planner(
    name,
    speed_scale=0.85,
    forward_gain=0.8,
    lateral_gain=1.2,
    vertical_gain=1.0,
    yaw_gain=1.5,
    smoothing=1.0,
    confidence_threshold=0.25,
    fallback_forward_scale=0.15,
):
    if name == GateSetpointPlanner.name:
        return GateSetpointPlanner(
            speed_scale=speed_scale,
            forward_gain=forward_gain,
            lateral_gain=lateral_gain,
            vertical_gain=vertical_gain,
            yaw_gain=yaw_gain,
            smoothing=smoothing,
            confidence_threshold=confidence_threshold,
            fallback_forward_scale=fallback_forward_scale,
        )
    raise ValueError(f"Unknown planner: {name}")
