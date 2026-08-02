#!/usr/bin/env python3
"""Pure-NumPy LC231 recurrent Puffer callable with the fitted phase-2 head."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.policy_callable_vq2_all24_sequence import NumpyLC216Policy


METADATA_SCHEMA = "vq2_lc231_phase2_full_residual_numpy_checkpoint_v1"


class NumpyLC231Policy(NumpyLC216Policy):
    def __init__(
        self, checkpoint: str | os.PathLike[str], *, expected_source_sha256: str,
    ) -> None:
        archive = np.load(checkpoint, allow_pickle=False)
        metadata = json.loads(str(archive["__metadata_json__"]))
        if (
            metadata.get("schema") != METADATA_SCHEMA
            or metadata.get("source_checkpoint_sha256") != expected_source_sha256
        ):
            archive.close()
            raise RuntimeError("LC231 NumPy checkpoint identity changed")
        self.values = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != "__metadata_json__"
        }
        archive.close()
        if self.values["phase_action_sequence"].shape != (9592, 4):
            raise RuntimeError("LC231 inherited action-sequence shape changed")
        self.reset()


_POLICY: NumpyLC231Policy | None = None
_CHECKPOINT: str | None = None
_SOURCE_SHA256: str | None = None


def _configuration() -> tuple[str, str]:
    checkpoint_value = os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "").strip()
    source_sha256 = os.getenv("PUFFER_POLICY_SOURCE_SHA256", "").strip()
    if not checkpoint_value or not source_sha256:
        raise RuntimeError(
            "PUFFER_POLICY_CHECKPOINT_PATH and PUFFER_POLICY_SOURCE_SHA256 are required"
        )
    checkpoint = str(Path(checkpoint_value).expanduser().resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, source_sha256


def reset() -> None:
    global _POLICY, _CHECKPOINT, _SOURCE_SHA256
    checkpoint, source_sha256 = _configuration()
    if (
        _POLICY is None or checkpoint != _CHECKPOINT
        or source_sha256 != _SOURCE_SHA256
    ):
        _POLICY = NumpyLC231Policy(
            checkpoint, expected_source_sha256=source_sha256
        )
        _CHECKPOINT, _SOURCE_SHA256 = checkpoint, source_sha256
    else:
        _POLICY.reset()


def policy(observation: Sequence[float]) -> tuple[float, ...]:
    global _POLICY, _CHECKPOINT, _SOURCE_SHA256
    checkpoint, source_sha256 = _configuration()
    if (
        _POLICY is None or checkpoint != _CHECKPOINT
        or source_sha256 != _SOURCE_SHA256
    ):
        _POLICY = NumpyLC231Policy(
            checkpoint, expected_source_sha256=source_sha256
        )
        _CHECKPOINT, _SOURCE_SHA256 = checkpoint, source_sha256
    return _POLICY(observation)


__all__ = ["NumpyLC231Policy", "policy", "reset"]
