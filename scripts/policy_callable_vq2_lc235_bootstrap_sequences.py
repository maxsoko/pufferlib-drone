#!/usr/bin/env python3
"""Pure-NumPy LC235 Puffer with source-locked per-phase sequence heads."""

from __future__ import annotations

import json
import os

import numpy as np

from scripts.policy_callable_vq2_all24_sequence import LEGAL_SIZE, PROGRESS_SCALE
from scripts.policy_callable_vq2_lc233_phase2_sequence import NumpyLC233Policy


METADATA_SCHEMA = "vq2_lc235_bootstrap_sequences_numpy_checkpoint_v1"


class NumpyLC235Policy(NumpyLC233Policy):
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
            raise RuntimeError("LC235 NumPy checkpoint identity changed")
        self.values = {}
        for name in archive.files:
            if name == "__metadata_json__":
                continue
            dtype = np.int32 if name == "bootstrap_sequence_offsets" else np.float32
            self.values[name] = np.ascontiguousarray(archive[name], dtype=dtype)
        archive.close()
        if self.values["phase_action_sequence"].shape != (9592, 4):
            raise RuntimeError("LC235 inherited late action sequence changed")
        if self.values["phase2_action_sequence"].shape != (1433, 4):
            raise RuntimeError("LC235 inherited phase-2 action sequence changed")
        offsets = self.values["bootstrap_sequence_offsets"]
        actions = self.values["bootstrap_sequence_actions"]
        if offsets.shape != (33, 2) or actions.ndim != 2 or actions.shape[1] != 4:
            raise RuntimeError("LC235 bootstrap sequence ABI changed")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.bootstrap_sequence_counters = np.zeros(33, dtype=np.int32)

    def __call__(self, observation: np.ndarray) -> tuple[float, ...]:
        puffer_action = super().__call__(observation)
        values = np.asarray(observation, dtype=np.float32)
        raw_index = int(np.rint(float(values[LEGAL_SIZE]) * PROGRESS_SCALE))
        if not 0 <= raw_index < 33:
            return puffer_action
        start, length = self.values["bootstrap_sequence_offsets"][raw_index]
        if start < 0 or length <= 0:
            return puffer_action
        counter = int(self.bootstrap_sequence_counters[raw_index])
        index = int(start) + min(counter, int(length) - 1)
        action = self.values["bootstrap_sequence_actions"][index]
        self.bootstrap_sequence_counters[raw_index] += 1
        return tuple(float(value) for value in action)


__all__ = ["NumpyLC235Policy"]
