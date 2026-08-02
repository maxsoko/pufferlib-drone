#!/usr/bin/env python3
"""Pure-NumPy LC216 recurrent Puffer callable for the Windows controller."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


OBSERVATION_SIZE = 4119
LEGAL_SIZE = 4118
MASK_SIZE = 4096
PROGRESS_SCALE = 6.0
EXPECTED_SOURCE_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _silu(value: np.ndarray) -> np.ndarray:
    return value * _sigmoid(value)


def _layer_norm(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    mean = np.mean(value, dtype=np.float32)
    centered = value - mean
    variance = np.mean(centered * centered, dtype=np.float32)
    return centered / np.sqrt(variance + np.float32(1e-5)) * weight + bias


def _linear(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    result = weight @ value
    if bias is not None:
        result = result + bias
    return np.asarray(result, dtype=np.float32)


def _conv2d(
    image_hwc: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int
) -> np.ndarray:
    kernel_h, kernel_w = weight.shape[2:]
    windows = np.lib.stride_tricks.sliding_window_view(
        image_hwc, (kernel_h, kernel_w), axis=(0, 1)
    )[::stride, ::stride]
    # windows: [out_h,out_w,in_channels,kernel_h,kernel_w]
    output = np.tensordot(
        windows, weight, axes=((2, 3, 4), (1, 2, 3))
    )
    output += bias
    return np.asarray(output, dtype=np.float32)


def _gru_cell(
    value: np.ndarray,
    hidden: np.ndarray,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> np.ndarray:
    input_gates = weight_ih @ value + bias_ih
    hidden_gates = weight_hh @ hidden + bias_hh
    ir, iz, inn = np.split(input_gates, 3)
    hr, hz, hn = np.split(hidden_gates, 3)
    reset = _sigmoid(ir + hr)
    update = _sigmoid(iz + hz)
    candidate = np.tanh(inn + reset * hn)
    return np.asarray((1.0 - update) * candidate + update * hidden, dtype=np.float32)


class NumpyLC216Policy:
    def __init__(self, checkpoint: str | os.PathLike[str]) -> None:
        archive = np.load(checkpoint, allow_pickle=False)
        metadata = json.loads(str(archive["__metadata_json__"]))
        if (
            metadata.get("schema") != "vq2_lc216_all24_numpy_checkpoint_v1"
            or metadata.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256
        ):
            raise RuntimeError("LC216 NumPy checkpoint identity changed")
        self.values = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != "__metadata_json__"
        }
        archive.close()
        if self.values["phase_action_sequence"].shape != (9592, 4):
            raise RuntimeError("LC216 action-sequence shape changed")
        self.reset()

    def reset(self) -> None:
        self.base_hidden = np.zeros(256, dtype=np.float32)
        self.adapter_hidden = np.zeros(64, dtype=np.float32)
        self.sequence_counter = 0

    def _encode(self, observation: np.ndarray) -> np.ndarray:
        value = observation[:MASK_SIZE].reshape(64, 64, 1)
        value = _silu(_conv2d(
            value,
            self.values["encoder.image.0.weight"],
            self.values["encoder.image.0.bias"],
            4,
        ))
        value = _silu(_conv2d(
            value,
            self.values["encoder.image.2.weight"],
            self.values["encoder.image.2.bias"],
            2,
        ))
        value = _silu(_conv2d(
            value,
            self.values["encoder.image.4.weight"],
            self.values["encoder.image.4.bias"],
            1,
        ))
        image = _linear(
            value.transpose(2, 0, 1).reshape(-1),
            self.values["encoder.image.7.weight"],
            self.values["encoder.image.7.bias"],
        )
        image = _silu(_layer_norm(
            image,
            self.values["encoder.image.8.weight"],
            self.values["encoder.image.8.bias"],
        ))
        sensor = _linear(
            observation[MASK_SIZE:LEGAL_SIZE],
            self.values["encoder.sensors.0.weight"],
            self.values["encoder.sensors.0.bias"],
        )
        sensor = _silu(_layer_norm(
            sensor,
            self.values["encoder.sensors.1.weight"],
            self.values["encoder.sensors.1.bias"],
        ))
        sensor = _silu(_linear(
            sensor,
            self.values["encoder.sensors.3.weight"],
            self.values["encoder.sensors.3.bias"],
        ))
        fused = _linear(
            np.concatenate((image, sensor)),
            self.values["encoder.fusion.0.weight"],
            self.values["encoder.fusion.0.bias"],
        )
        fused = _silu(_layer_norm(
            fused,
            self.values["encoder.fusion.1.weight"],
            self.values["encoder.fusion.1.bias"],
        ))
        progress = np.asarray([observation[LEGAL_SIZE]], dtype=np.float32)
        return fused + _linear(
            progress, self.values["phase_embedding.weight"]
        )

    def __call__(self, observation: Sequence[float]) -> tuple[float, ...]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"LC216 expects {OBSERVATION_SIZE} values, got {values.shape}"
            )
        if not bool(np.isfinite(values).all()) or float(values[LEGAL_SIZE]) < 0.0:
            raise ValueError("LC216 observation must be finite with nonnegative progress")
        encoded = self._encode(values)
        self.base_hidden = _gru_cell(
            encoded,
            self.base_hidden,
            self.values["recurrent.weight_ih_l0"],
            self.values["recurrent.weight_hh_l0"],
            self.values["recurrent.bias_ih_l0"],
            self.values["recurrent.bias_hh_l0"],
        )
        raw_index = int(np.rint(float(values[LEGAL_SIZE]) * PROGRESS_SCALE))
        tracked_index = min(max(raw_index, 0), 32)
        feature = np.tanh(
            self.values["indexed_phase_residual_input"][tracked_index]
            @ self.base_hidden
            + self.values["indexed_phase_residual_input_bias"][tracked_index]
        )
        residual = (
            self.values["indexed_phase_residual_output"][tracked_index] @ feature
            + self.values["indexed_phase_residual_output_bias"][tracked_index]
        )
        if raw_index > 32:
            residual.fill(0.0)
        pre_tanh = _linear(
            self.base_hidden,
            self.values["action_head.weight"],
            self.values["action_head.bias"],
        ) + residual
        if raw_index == 15:
            self.adapter_hidden = _gru_cell(
                self.base_hidden,
                self.adapter_hidden,
                self.values["phase_adapter_cell.weight_ih"],
                self.values["phase_adapter_cell.weight_hh"],
                self.values["phase_adapter_cell.bias_ih"],
                self.values["phase_adapter_cell.bias_hh"],
            )
            pre_tanh += _linear(
                self.adapter_hidden,
                self.values["phase_adapter_output.weight"],
                self.values["phase_adapter_output.bias"],
            )
        mean = np.tanh(pre_tanh)
        if 16 <= raw_index < 24:
            index = min(self.sequence_counter, 9591)
            mean = self.values["phase_action_sequence"][index]
            self.sequence_counter += 1
        return tuple(float(value) for value in mean)


_POLICY: NumpyLC216Policy | None = None
_CHECKPOINT: str | None = None


def _checkpoint_path() -> str:
    value = os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "").strip()
    if not value:
        raise RuntimeError("PUFFER_POLICY_CHECKPOINT_PATH is required")
    path = str(Path(value).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return path


def reset() -> None:
    global _POLICY, _CHECKPOINT
    checkpoint = _checkpoint_path()
    if _POLICY is None or checkpoint != _CHECKPOINT:
        # Load and validate the immutable archive during lifecycle setup, not
        # on the first timed recurrent tick.  Subsequent official resets retain
        # the arrays and clear only the actor's recurrent state/counter.
        _POLICY = NumpyLC216Policy(checkpoint)
        _CHECKPOINT = checkpoint
    else:
        _POLICY.reset()


def policy(observation: Sequence[float]) -> tuple[float, ...]:
    global _POLICY, _CHECKPOINT
    checkpoint = _checkpoint_path()
    if _POLICY is None or checkpoint != _CHECKPOINT:
        _POLICY = NumpyLC216Policy(checkpoint)
        _CHECKPOINT = checkpoint
    return _POLICY(observation)


__all__ = ["NumpyLC216Policy", "policy", "reset"]
