#!/usr/bin/env python3
"""N233: verified-once Gate-4 visual MPC with the proven 60 Hz prefix clock.

N232's planner was not reached in its official batch because the deployment
callable re-read and SHA-256 hashed the dynamics JSON on every inference. That
cost forced a live-rate prefix which regressed before Gate 4. N233 validates
the launcher-pinned immutable model once, reuses it for the process lifetime,
and otherwise delegates to the same N232 controller. Explicit 60 Hz logical
timestamps preserve the N230/N231 recurrent schedule and bound Gate-4 replans.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_mpc_n232 as n232


EXPECTED_N232_SHA256 = (
    "0c291b21ca49cd71281744a1cf1da770f42a36b81dec916691d965e91f5228cf"
)
EXPECTED_MODEL_SHA256 = n232.EXPECTED_MODEL_SHA256
POLICY_STATE_HZ = 60.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N232_PATH = Path(n232.__file__).resolve()
if _sha256(_N232_PATH) != EXPECTED_N232_SHA256:
    raise RuntimeError("N232 policy source hash mismatch")


_CONTROLLER: n232.Gate4MPCController | None = None
_CONTROLLER_MODEL_PATH: Path | None = None
_MODEL_VALIDATIONS = 0
_INFERENCE_CALLS = 0


def _load_verified_controller() -> n232.Gate4MPCController:
    global _CONTROLLER, _CONTROLLER_MODEL_PATH, _MODEL_VALIDATIONS
    if _CONTROLLER is not None:
        return _CONTROLLER
    value = os.getenv("PUFFER_POLICY_GATE4_DYNAMICS_PATH", "").strip()
    if not value:
        raise RuntimeError("PUFFER_POLICY_GATE4_DYNAMICS_PATH is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    _MODEL_VALIDATIONS += 1
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Gate-4 dynamics hash mismatch: "
            f"expected {EXPECTED_MODEL_SHA256}, got {actual}"
        )
    _CONTROLLER = n232.Gate4MPCController(
        n232.mpc.VisualDynamicsEnsemble.load(path)
    )
    _CONTROLLER_MODEL_PATH = path
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n232.parent.reset()
    _load_verified_controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _load_verified_controller().apply(
        values,
        parent_actions,
        now_s=logical_time_s,
    )


def controller_snapshot() -> dict:
    controller = _load_verified_controller()
    return {
        "n233_verified_once_fixed_clock": {
            "inference_calls": _INFERENCE_CALLS,
            "model_validations": _MODEL_VALIDATIONS,
            "model_path": None
            if _CONTROLLER_MODEL_PATH is None
            else str(_CONTROLLER_MODEL_PATH),
            "contract": {
                "policy_state_hz": POLICY_STATE_HZ,
                "model_validated_once_per_process": True,
                "launcher_also_hash_pins_model": True,
                "n232_controller_unchanged": True,
                "runtime_privileged_state": False,
            },
        },
        "n232_gate4_visual_mpc": controller.snapshot(),
        "parent": n232.parent.controller_snapshot(),
    }

