#!/usr/bin/env python3
"""Pure-NumPy LC233 recurrent Puffer with checkpointed phase-2 rescue actions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scripts.policy_callable_vq2_all24_sequence import (
    LEGAL_SIZE,
    PROGRESS_SCALE,
    NumpyLC216Policy,
)


METADATA_SCHEMA = "vq2_lc233_phase2_action_sequence_numpy_checkpoint_v1"


class NumpyLC233Policy(NumpyLC216Policy):
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
            raise RuntimeError("LC233 NumPy checkpoint identity changed")
        self.values = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != "__metadata_json__"
        }
        archive.close()
        if self.values["phase_action_sequence"].shape != (9592, 4):
            raise RuntimeError("LC233 inherited late action sequence changed")
        if self.values["phase2_action_sequence"].shape != (1433, 4):
            raise RuntimeError("LC233 phase-2 action sequence changed")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.phase2_sequence_counter = 0

    def __call__(self, observation: np.ndarray) -> tuple[float, ...]:
        puffer_action = super().__call__(observation)
        values = np.asarray(observation, dtype=np.float32)
        raw_index = int(np.rint(float(values[LEGAL_SIZE]) * PROGRESS_SCALE))
        if raw_index != 2:
            return puffer_action
        index = min(
            self.phase2_sequence_counter,
            self.values["phase2_action_sequence"].shape[0] - 1,
        )
        action = self.values["phase2_action_sequence"][index]
        self.phase2_sequence_counter += 1
        return tuple(float(value) for value in action)


__all__ = ["NumpyLC233Policy"]
