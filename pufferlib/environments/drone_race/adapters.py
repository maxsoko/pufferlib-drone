import numpy as np

from .contracts import PerceptionOutput, SCHEMA_VERSION


class PerceptionAdapter:
    name = "base"

    def observe(self, env):  # pragma: no cover - interface only
        raise NotImplementedError


class PrivilegedStateAdapter(PerceptionAdapter):
    name = "privileged_state"

    def observe(self, env):
        rel_center_body, normal_body = env._relative_gate_pose_body()
        return PerceptionOutput(
            schema_version=SCHEMA_VERSION,
            relative_gate_center_body=rel_center_body.astype(np.float32),
            gate_normal_body=normal_body.astype(np.float32),
            confidence=1.0,
            valid=True,
            source=self.name,
        )


class CameraSensorStubAdapter(PerceptionAdapter):
    name = "camera_sensor_stub"

    def __init__(self, noise_std=0.0, dropout_prob=0.0, allow_fallback=True):
        self.noise_std = float(max(noise_std, 0.0))
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.allow_fallback = bool(allow_fallback)

    def observe(self, env):
        rel_center_body, normal_body = env._relative_gate_pose_body()

        rng = env._rng if env._rng is not None else np.random.default_rng(0)
        dropped = bool(rng.random() < self.dropout_prob)

        if dropped and not self.allow_fallback:
            rel = np.zeros(3, dtype=np.float32)
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return PerceptionOutput(
                schema_version=SCHEMA_VERSION,
                relative_gate_center_body=rel,
                gate_normal_body=normal,
                confidence=0.0,
                valid=False,
                source=self.name,
            )

        rel = rel_center_body.astype(np.float32)
        normal = normal_body.astype(np.float32)
        if self.noise_std > 0.0:
            rel = rel + rng.normal(0.0, self.noise_std, size=3).astype(np.float32)
            normal = normal + rng.normal(0.0, self.noise_std, size=3).astype(np.float32)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1e-8:
                normal = normal / normal_norm

        confidence = 0.3 if dropped else 0.8
        return PerceptionOutput(
            schema_version=SCHEMA_VERSION,
            relative_gate_center_body=rel,
            gate_normal_body=normal,
            confidence=confidence,
            valid=True,
            source=self.name,
        )


def make_perception_adapter(name, camera_noise_std=0.0, camera_dropout_prob=0.0, allow_perception_fallback=True):
    if name == PrivilegedStateAdapter.name:
        return PrivilegedStateAdapter()
    if name == CameraSensorStubAdapter.name:
        return CameraSensorStubAdapter(
            noise_std=camera_noise_std,
            dropout_prob=camera_dropout_prob,
            allow_fallback=allow_perception_fallback,
        )
    raise ValueError(f"Unknown perception adapter: {name}")
