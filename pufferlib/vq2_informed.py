"""Hard observation boundary for the VQ2 informed visual policy.

The native ``drone_race_vision`` environment emits one flat tensor for Puffer's
vector ABI.  The leading slice is competition-legal and the trailing slice is
training-only information for the world-model decoder.  Deployed actor modules
accept the legal slice only; this module is the sole supported split point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


VISUAL_WIDTH = 64
VISUAL_HEIGHT = 64
MASK_SIZE = VISUAL_WIDTH * VISUAL_HEIGHT
OBSERVATION_SCHEMA = "vq2_visual_ctbr_v2"
ACTION_SCHEMA = "normalized_attitude_ctbr_v1"


def configure_full_start_collection(config: dict[str, Any]) -> dict[str, Any]:
    """Disable every local/segment reset while retaining open-ended collection."""

    environment = config.get("env")
    if not isinstance(environment, dict):
        raise RuntimeError("VQ2 collection config does not contain an env section")
    if int(environment.get("use_custom_start", 0)) and int(
        environment.get("start_gate_index", 0)
    ) > 0:
        raise RuntimeError("full-start collection refuses a custom later-gate start")
    environment["gate_local_start_curriculum"] = 0
    environment["gate_local_start_probability"] = 0.0
    environment["mixed_start_curriculum"] = 0
    environment["segment_start_probability"] = 0.0
    environment["evaluation_episode_limit"] = 0
    environment["evaluation_episode_offset"] = 0
    return config


def configure_full_start_evaluation(
    config: dict[str, Any],
    *,
    episodes_per_agent: int,
    episode_offset: int,
) -> dict[str, Any]:
    """Fail closed to uninterrupted course starts for admission screens.

    Training may use randomized gate-local or empirical segment resets. A
    deterministic evaluator must never inherit those curricula silently.
    Custom gate-zero starts are permitted because they can encode the true
    course spawn; a custom start at any later gate is not a full-course screen.
    """

    environment = config.get("env")
    if not isinstance(environment, dict):
        raise RuntimeError("VQ2 evaluation config does not contain an env section")
    if int(environment.get("use_custom_start", 0)) and int(
        environment.get("start_gate_index", 0)
    ) > 0:
        raise RuntimeError(
            "full-start evaluation refuses a custom start after Gate 1"
        )
    environment["gate_local_start_curriculum"] = 0
    environment["gate_local_start_probability"] = 0.0
    environment["mixed_start_curriculum"] = 0
    environment["segment_start_probability"] = 0.0
    environment["evaluation_episode_limit"] = episodes_per_agent
    environment["evaluation_episode_offset"] = episode_offset
    return config

BODY_RATE = slice(MASK_SIZE, MASK_SIZE + 3)
MOTOR_FEEDBACK = slice(BODY_RATE.stop, BODY_RATE.stop + 4)
ACTION_HISTORY = slice(MOTOR_FEEDBACK.stop, MOTOR_FEEDBACK.stop + 12)
NEW_FRAME = slice(ACTION_HISTORY.stop, ACTION_HISTORY.stop + 1)
FRAME_AGE = slice(NEW_FRAME.stop, NEW_FRAME.stop + 1)
STEP_DT = slice(FRAME_AGE.stop, FRAME_AGE.stop + 1)
LEGAL_OBS_SIZE = STEP_DT.stop

PRIVILEGED_NAMES = (
    "world_position_x",
    "world_position_y",
    "world_position_z",
    "gate_relative_body_x",
    "gate_relative_body_y",
    "gate_relative_body_z",
    "world_velocity_x",
    "world_velocity_y",
    "world_velocity_z",
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "attitude_qw",
    "attitude_qx",
    "attitude_qy",
    "attitude_qz",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
    "motor_rpm_0",
    "motor_rpm_1",
    "motor_rpm_2",
    "motor_rpm_3",
    "camera_roll",
    "camera_pitch",
    "camera_yaw",
    "ctbr_hover_thrust",
    "ctbr_vertical_accel_per_thrust",
    "ctbr_max_thrust",
    "ctbr_rate_lag_time_constant",
    "ctbr_gravity",
    "ctbr_linear_drag",
    "ordered_gate_phase",
    "active_gate_radius",
)
PRIVILEGED_SIZE = len(PRIVILEGED_NAMES)
ENV_OBS_SIZE = LEGAL_OBS_SIZE + PRIVILEGED_SIZE


@dataclass(frozen=True)
class InformedObservation:
    """Separated environment batch.

    ``legal`` is the only tensor permitted to reach the posterior encoder,
    recurrent state update, or actor. ``privileged`` is a decoder target only.
    """

    legal: torch.Tensor
    privileged: torch.Tensor


def split_environment_observation(observation: torch.Tensor) -> InformedObservation:
    """Split a native Puffer observation without copying it."""

    if observation.shape[-1] != ENV_OBS_SIZE:
        raise ValueError(
            f"expected a {ENV_OBS_SIZE}-value drone_race_vision observation, "
            f"received shape {tuple(observation.shape)}"
        )
    return InformedObservation(
        legal=observation[..., :LEGAL_OBS_SIZE],
        privileged=observation[..., LEGAL_OBS_SIZE:],
    )


class VQ2VisualEncoder(nn.Module):
    """Compact CNN/sensor encoder that structurally rejects privilege.

    Its input dimension is exactly ``LEGAL_OBS_SIZE``. A caller cannot pass the
    native legal-plus-information tensor by mistake; it must first go through
    :func:`split_environment_observation`.
    """

    def __init__(self, output_size: int = 256) -> None:
        super().__init__()
        self.output_size = output_size
        self.image = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 224),
            nn.LayerNorm(224),
            nn.SiLU(),
        )
        self.sensors = nn.Sequential(
            nn.Linear(LEGAL_OBS_SIZE - MASK_SIZE, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(224 + 64, output_size),
            nn.LayerNorm(output_size),
            nn.SiLU(),
        )

    def forward(self, legal_observation: torch.Tensor) -> torch.Tensor:
        if legal_observation.shape[-1] != LEGAL_OBS_SIZE:
            raise ValueError(
                "VQ2VisualEncoder accepts only the competition-legal slice; "
                f"expected {LEGAL_OBS_SIZE}, got {legal_observation.shape[-1]}"
            )
        leading = legal_observation.shape[:-1]
        flat = legal_observation.reshape(-1, LEGAL_OBS_SIZE)
        image = flat[:, :MASK_SIZE].reshape(-1, 1, VISUAL_HEIGHT, VISUAL_WIDTH)
        sensor = flat[:, MASK_SIZE:]
        encoded = self.fusion(torch.cat((self.image(image), self.sensors(sensor)), -1))
        return encoded.reshape(*leading, self.output_size)


class LegalActorAdapter(nn.Module):
    """Training adapter that removes information before invoking an actor.

    World-model training can retain the flat native tensor in replay. This
    adapter makes the privilege boundary explicit and testable at the actor
    call site. A deployed export omits this adapter and accepts legal inputs.
    """

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, environment_observation: torch.Tensor) -> torch.Tensor:
        separated = split_environment_observation(environment_observation)
        return self.actor(separated.legal)
