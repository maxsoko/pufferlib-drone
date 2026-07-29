#!/usr/bin/env python3
"""N215: H12 official prefix plus the promoted N214 recurrent tail.

The control topology and observation ABI are identical to N210.  Only the
late-tail checkpoint changes: N214 step 65,536 is the best zero-intervention
checkpoint selected on a common 128-episode cohort and confirmed on 256 held-
out measured Gate-4 starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import policy_callable_n209_unified_tail as n210_source


EXPECTED_N214_CHECKPOINT_SHA256 = (
    "999ab9061ca04f7d4b66d33991afd7fa14d1ca0049d9398a66f361419a15fa75"
)

# Load the already-tested N210 topology into a private module instance so the
# original N209 deployment module keeps its own immutable admission pin.
_SPEC = importlib.util.spec_from_file_location(
    "_policy_callable_n214_private_base", Path(n210_source.__file__).resolve()
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the N210 controller topology")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)
base.EXPECTED_N209_CHECKPOINT_SHA256 = EXPECTED_N214_CHECKPOINT_SHA256


def reset() -> None:
    base.reset()


def infer(observation):
    return base.infer(observation)


def controller_snapshot() -> dict:
    snapshot = base.controller_snapshot()
    tail = snapshot.pop("n210_unified_tail")
    tail["parameters"]["tail_checkpoint"] = "n214_step_65536"
    tail["parameters"]["teacher_intervention_at_deployment"] = 0.0
    snapshot["n215_unified_tail"] = tail
    return snapshot
