from dataclasses import dataclass

import numpy as np


SCHEMA_VERSION = 1


@dataclass
class PerceptionOutput:
    schema_version: int
    relative_gate_center_body: np.ndarray
    gate_normal_body: np.ndarray
    confidence: float
    valid: bool
    source: str


@dataclass
class PlannerOutput:
    schema_version: int
    desired_velocity_body: np.ndarray
    desired_yaw_rate: float
    fallback_active: bool
    source: str


@dataclass
class ControllerOutput:
    schema_version: int
    action: np.ndarray
    saturated: bool
    fallback_active: bool
