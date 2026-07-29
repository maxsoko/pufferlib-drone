#!/usr/bin/env python3
"""N200: exact N199 with the source-pinned 3.7 m terminal level restored."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import policy_callable_gate3_later_terminal_level_n195 as n196_module
import policy_callable_gate3_low_confidence_latched_recovery_n198 as n199


EXPECTED_N199_SHA256 = (
    "9deaecd3957d02b11dabdc7b1fb36ce649f1e5575fd11fc1e190332508742d30"
)
EXPECTED_N196_SHA256 = (
    "3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(n199, EXPECTED_N199_SHA256, "N199")
_assert_source_hash(n196_module, EXPECTED_N196_SHA256, "N196 terminal controller")


_RESTORED_TERMINAL = n196_module.Gate3LaterTerminalSafeLevel()


def reset() -> None:
    n199.reset()
    _RESTORED_TERMINAL.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _RESTORED_TERMINAL.apply(observation, n199.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate3_restored_terminal_safe_level": _RESTORED_TERMINAL.snapshot(),
        "n199": n199.controller_snapshot(),
    }
