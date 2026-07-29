#!/usr/bin/env python3
"""N219: H12 official prefix plus the predictor-adapted recurrent tail."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import policy_callable_n209_unified_tail as n210_source


EXPECTED_N219_CHECKPOINT_SHA256 = (
    "20a44f9bddddec335bbbdd099aa0febe8041163b588a689586ca4cd9ca0110ec"
)

_SPEC = importlib.util.spec_from_file_location(
    "_policy_callable_n219_private_base", Path(n210_source.__file__).resolve()
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the N210 controller topology")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
base.EXPECTED_N209_CHECKPOINT_SHA256 = EXPECTED_N219_CHECKPOINT_SHA256


def reset() -> None:
    base.reset()


def infer(observation):
    return base.infer(observation)


def controller_snapshot() -> dict:
    snapshot = base.controller_snapshot()
    tail = snapshot.pop("n210_unified_tail")
    tail["parameters"]["tail_checkpoint"] = "n219_predictor_step_32768"
    tail["parameters"]["gate_motion_predict_dropout"] = True
    tail["parameters"]["teacher_intervention_at_deployment"] = 0.0
    snapshot["n219_unified_tail"] = tail
    return snapshot
