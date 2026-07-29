#!/usr/bin/env python3
"""Checkpoint-backed competition policy callable.

Usage:
  export PUFFER_POLICY_CHECKPOINT_PATH=checkpoints/drone_race_competition/<run>/<step>.bin
  python scripts/drone_sitl_competition_smoke.py \
    --control-mode policy \
    --policy-callable scripts/policy_callable_checkpoint.py:infer
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _align8(idx: int) -> int:
    return (idx + 7) & ~7


def _align(idx: int, precision_bytes: int) -> int:
    if precision_bytes not in (2, 4):
        raise ValueError("layout_precision_bytes must be 2 (BF16) or 4 (FP32)")
    alignment_elements = 16 // precision_bytes
    return (idx + alignment_elements - 1) & ~(alignment_elements - 1)


def _take_aligned(
    weights: np.ndarray, idx: int, count: int, *, precision_bytes: int
) -> tuple[np.ndarray, int]:
    chunk = weights[idx:idx + count]
    if len(chunk) != count:
        raise ValueError(f"checkpoint ended early: requested {count}, got {len(chunk)}")
    return chunk, _align(idx + count, precision_bytes)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mingru_g(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x + 0.5, _sigmoid(x))


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round FP32 values to BF16 while retaining an FP32 NumPy container."""
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def _cuda_fast_tanh(x: np.ndarray) -> np.ndarray:
    """Match the polynomial used by the native CUDA MinGRU candidate path."""
    value = np.clip(np.asarray(x, dtype=np.float32), np.float32(-9.0), np.float32(9.0))
    squared = value * value
    numerator = squared * np.float32(-2.76076847742355e-16) + np.float32(2.00018790482477e-13)
    numerator = squared * numerator + np.float32(-8.60467152213735e-11)
    numerator = squared * numerator + np.float32(5.12229709037114e-08)
    numerator = squared * numerator + np.float32(1.48572235717979e-05)
    numerator = squared * numerator + np.float32(6.37261928875436e-04)
    numerator = squared * numerator + np.float32(4.89352455891786e-03)
    numerator = value * numerator
    denominator = squared * np.float32(1.19825839466702e-06) + np.float32(1.18534705686654e-04)
    denominator = squared * denominator + np.float32(2.26843463243900e-03)
    denominator = squared * denominator + np.float32(4.89352518554385e-03)
    return numerator / denominator


def _cuda_fast_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(
        np.float32(0.5) * (_cuda_fast_tanh(np.float32(0.5) * x) + np.float32(1.0)),
        np.float32(0.0),
        np.float32(1.0),
    )


def _cuda_lerp(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    difference = b - a
    return np.where(
        np.abs(weight) < np.float32(0.5),
        a + weight * difference,
        b - difference * (np.float32(1.0) - weight),
    )


@dataclass
class CheckpointPolicy:
    input_dim: int
    hidden_dim: int
    num_layers: int
    num_actions: int
    encoder: np.ndarray
    decoder: np.ndarray
    log_std: np.ndarray
    mingru_proj: list[np.ndarray]
    state: np.ndarray
    native_bf16: bool = False
    layout_precision_bytes: int = 2

    @classmethod
    def load(
        cls,
        path: str,
        *,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_actions: int = 4,
        native_bf16: bool = False,
        layout_precision_bytes: int = 2,
    ) -> "CheckpointPolicy":
        if layout_precision_bytes not in (2, 4):
            raise ValueError(
                "layout_precision_bytes must be 2 (BF16) or 4 (FP32)"
            )
        serialized = np.fromfile(path, dtype=np.float32)
        raw_count = (
            hidden_dim * input_dim
            + (num_actions + 1) * hidden_dim
            + num_actions
            + num_layers * 3 * hidden_dim * hidden_dim
        )
        if len(serialized) < raw_count:
            raise ValueError(
                f"checkpoint ended early: expected at least {raw_count} floats, "
                f"got {len(serialized)}"
            )

        # CUDA parameter tensors start on 16-byte boundaries, but the FP32
        # master checkpoint is saved with the unpadded logical element count.
        # Tensor starts therefore depend on whether the native arena used
        # two-byte BF16 or four-byte FP32 elements even though both checkpoint
        # files serialize FP32 master values.
        # Consequently the file contains any interior alignment slots and
        # truncates the same number of values from its tail. Pad the tail back
        # with zeros and parse only the native aligned view. Falling back to a
        # packed view silently changes every MinGRU weight after log_std.
        aligned_count = 0
        for count in (
            hidden_dim * input_dim,
            (num_actions + 1) * hidden_dim,
            num_actions,
            *([3 * hidden_dim * hidden_dim] * num_layers),
        ):
            aligned_count = _align(
                aligned_count + count, layout_precision_bytes
            )
        weights = np.pad(
            serialized,
            (0, max(0, aligned_count - len(serialized))),
            mode="constant",
        )

        idx = 0
        encoder, idx = _take_aligned(
            weights,
            idx,
            hidden_dim * input_dim,
            precision_bytes=layout_precision_bytes,
        )
        encoder = encoder.reshape(hidden_dim, input_dim)

        decoder_rows = num_actions + 1
        decoder, idx = _take_aligned(
            weights,
            idx,
            decoder_rows * hidden_dim,
            precision_bytes=layout_precision_bytes,
        )
        decoder = decoder.reshape(decoder_rows, hidden_dim)

        log_std, idx = _take_aligned(
            weights, idx, num_actions, precision_bytes=layout_precision_bytes
        )

        mingru_proj: list[np.ndarray] = []
        for _layer in range(num_layers):
            proj, idx = _take_aligned(
                weights,
                idx,
                3 * hidden_dim * hidden_dim,
                precision_bytes=layout_precision_bytes,
            )
            mingru_proj.append(proj.reshape(3 * hidden_dim, hidden_dim))

        encoder = encoder.astype(np.float32)
        decoder = decoder.astype(np.float32)
        log_std = log_std.astype(np.float32)
        mingru_proj = [proj.astype(np.float32) for proj in mingru_proj]
        if native_bf16:
            encoder = _round_bf16(encoder)
            decoder = _round_bf16(decoder)
            mingru_proj = [_round_bf16(proj) for proj in mingru_proj]

        state = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            encoder=encoder,
            decoder=decoder,
            log_std=log_std,
            mingru_proj=mingru_proj,
            state=state,
            native_bf16=native_bf16,
            layout_precision_bytes=layout_precision_bytes,
        )

    def reset_state(self) -> None:
        self.state.fill(0.0)

    def hidden_features(self, observation: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != self.input_dim:
            raise ValueError(f"expected observation shape ({self.input_dim},), got {obs.shape}")

        if self.native_bf16:
            obs = _round_bf16(obs)
            h = _round_bf16(self.encoder @ obs)
        else:
            h = self.encoder @ obs
        for layer_idx in range(self.num_layers):
            proj = self.mingru_proj[layer_idx] @ h
            if self.native_bf16:
                proj = _round_bf16(proj)
            hidden = proj[0:self.hidden_dim]
            gate = proj[self.hidden_dim : 2 * self.hidden_dim]
            highway_proj = proj[2 * self.hidden_dim : 3 * self.hidden_dim]

            if self.native_bf16:
                candidate = np.where(
                    hidden >= np.float32(0.0),
                    hidden + np.float32(0.5),
                    _cuda_fast_sigmoid(hidden),
                )
                out = _cuda_lerp(self.state[layer_idx], candidate, _sigmoid(gate))
                highway = _sigmoid(highway_proj)
                h = _round_bf16(
                    highway * out + (np.float32(1.0) - highway) * h
                )
                self.state[layer_idx] = _round_bf16(out)
            else:
                out = self.state[layer_idx] + _sigmoid(gate) * (
                    _mingru_g(hidden) - self.state[layer_idx]
                )
                highway = _sigmoid(highway_proj)
                h = highway * out + (1.0 - highway) * h
                self.state[layer_idx] = out

        return h.copy()

    def infer(self, observation: Sequence[float]) -> list[float]:
        h = self.hidden_features(observation)
        decoded = self.decoder @ h
        if self.native_bf16:
            decoded = _round_bf16(decoded)
        actions = decoded[: self.num_actions]
        return [float(max(-1.0, min(1.0, value))) for value in actions]


_MODEL: CheckpointPolicy | None = None
_MODEL_KEY: tuple[str, int, int, int, int, int, bool] | None = None


@dataclass(frozen=True)
class GateActionBias:
    gate_index: int = -1
    progress_observation_index: int = 23
    progress_denominator: int = 6
    pitch: float = 0.0
    roll: float = 0.0
    thrust: float = 0.0
    yaw: float = 0.0
    visibility_observation_index: int = 10
    forward_observation_index: int = 11
    alignment_observation_index: int = 17
    table: tuple[
        tuple[int, tuple[float, float, float, float], bool, float, float], ...
    ] = ()


_ACTION_BIAS: GateActionBias | None = None
_ACTION_BIAS_KEY: tuple[
    str, str, str, str, str, str, str, str, str, str, str
] | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _resolve_action_bias() -> GateActionBias:
    global _ACTION_BIAS
    global _ACTION_BIAS_KEY

    key = (
        os.getenv("PUFFER_POLICY_ACTION_BIAS_GATE_INDEX", "-1").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX", "23").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_DENOMINATOR", "6").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_PITCH", "0").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_ROLL", "0").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_THRUST", "0").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_YAW", "0").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_TABLE_JSON", "").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_VISIBILITY_INDEX", "10").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_ALIGNMENT_INDEX", "17").strip(),
        os.getenv("PUFFER_POLICY_ACTION_BIAS_FORWARD_INDEX", "11").strip(),
    )
    if _ACTION_BIAS is None or _ACTION_BIAS_KEY != key:
        raw_table = json.loads(key[7]) if key[7] else {}
        if not isinstance(raw_table, dict):
            raise ValueError("PUFFER_POLICY_ACTION_BIAS_TABLE_JSON must be an object")
        table: list[
            tuple[int, tuple[float, float, float, float], bool, float, float]
        ] = []
        for raw_gate_index, raw_biases in raw_table.items():
            gate_index = int(raw_gate_index)
            if gate_index < 0:
                raise ValueError("action-bias table gate indices must be nonnegative")
            visible_only = False
            min_alignment = 0.0
            min_forward_m = 0.0
            if isinstance(raw_biases, dict):
                visible_only = raw_biases.get("visible_only", False)
                if not isinstance(visible_only, bool):
                    raise ValueError("visible_only must be a JSON boolean")
                min_alignment = float(raw_biases.get("min_alignment", 0.0))
                if not 0.0 <= min_alignment <= 1.0:
                    raise ValueError("min_alignment must be in [0, 1]")
                min_forward_m = float(raw_biases.get("min_forward_m", 0.0))
                if not np.isfinite(min_forward_m) or min_forward_m < 0.0:
                    raise ValueError("min_forward_m must be finite and nonnegative")
                raw_biases = raw_biases.get("bias")
            if not isinstance(raw_biases, list) or len(raw_biases) != 4:
                raise ValueError(
                    "each action-bias table value must be "
                    "[pitch, roll, thrust, yaw] or an object with that bias"
                )
            biases = tuple(float(value) for value in raw_biases)
            if not all(np.isfinite(value) for value in biases):
                raise ValueError("action-bias table values must be finite")
            table.append(
                (gate_index, biases, visible_only, min_alignment, min_forward_m)
            )
        config = GateActionBias(
            gate_index=int(key[0] or "-1"),
            progress_observation_index=int(key[1] or "23"),
            progress_denominator=int(key[2] or "6"),
            pitch=float(key[3] or "0"),
            roll=float(key[4] or "0"),
            thrust=float(key[5] or "0"),
            yaw=float(key[6] or "0"),
            visibility_observation_index=int(key[8] or "10"),
            alignment_observation_index=int(key[9] or "17"),
            forward_observation_index=int(key[10] or "11"),
            table=tuple(sorted(table)),
        )
        if config.gate_index < -1:
            raise ValueError("PUFFER_POLICY_ACTION_BIAS_GATE_INDEX must be -1 or nonnegative")
        if config.progress_observation_index < 0:
            raise ValueError("PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX must be nonnegative")
        if config.progress_denominator <= 0:
            raise ValueError("PUFFER_POLICY_ACTION_BIAS_DENOMINATOR must be positive")
        if config.visibility_observation_index < 0:
            raise ValueError(
                "PUFFER_POLICY_ACTION_BIAS_VISIBILITY_INDEX must be nonnegative"
            )
        if config.alignment_observation_index < 0:
            raise ValueError(
                "PUFFER_POLICY_ACTION_BIAS_ALIGNMENT_INDEX must be nonnegative"
            )
        if config.forward_observation_index < 0:
            raise ValueError(
                "PUFFER_POLICY_ACTION_BIAS_FORWARD_INDEX must be nonnegative"
            )
        _ACTION_BIAS = config
        _ACTION_BIAS_KEY = key
    return _ACTION_BIAS


def _apply_action_bias(
    observation: Sequence[float], actions: Sequence[float], config: GateActionBias
) -> list[float]:
    output = [float(value) for value in actions]
    if config.gate_index < 0 and not config.table:
        return output
    if config.progress_observation_index >= len(observation):
        raise ValueError(
            "gate-indexed action bias requires normalized progress observation "
            f"at index {config.progress_observation_index}, but input has "
            f"length {len(observation)}"
        )
    if len(output) < 4:
        raise ValueError("gate-indexed action bias requires four policy actions")
    scaled_progress = (
        float(observation[config.progress_observation_index])
        * config.progress_denominator
    )
    active_gate_index = int(round(scaled_progress))
    if abs(scaled_progress - active_gate_index) > 1e-3:
        raise ValueError(
            "gate-indexed action bias received a non-quantized progress value: "
            f"{scaled_progress}/{config.progress_denominator}"
        )
    if active_gate_index == config.gate_index:
        for action_index, bias in enumerate(
            (config.pitch, config.roll, config.thrust, config.yaw)
        ):
            output[action_index] = float(
                max(-1.0, min(1.0, output[action_index] + bias))
            )
    for (
        gate_index,
        biases,
        visible_only,
        min_alignment,
        min_forward_m,
    ) in config.table:
        if active_gate_index != gate_index:
            continue
        if visible_only:
            if config.visibility_observation_index >= len(observation):
                raise ValueError(
                    "visible-only action bias requires visibility observation "
                    f"at index {config.visibility_observation_index}"
                )
            if float(observation[config.visibility_observation_index]) <= 0.5:
                continue
        if min_alignment > 0.0:
            if config.alignment_observation_index >= len(observation):
                raise ValueError(
                    "alignment-gated action bias requires alignment observation "
                    f"at index {config.alignment_observation_index}"
                )
            if float(observation[config.alignment_observation_index]) < min_alignment:
                continue
        if min_forward_m > 0.0:
            if config.forward_observation_index >= len(observation):
                raise ValueError(
                    "range-gated action bias requires forward observation "
                    f"at index {config.forward_observation_index}"
                )
            minimum_forward_norm = float(np.tanh(np.float32(min_forward_m * 0.1)))
            if float(observation[config.forward_observation_index]) < minimum_forward_norm:
                continue
        for action_index, bias in enumerate(biases):
            output[action_index] = float(
                max(-1.0, min(1.0, output[action_index] + bias))
            )
    return output


def _resolve_model() -> CheckpointPolicy:
    global _MODEL
    global _MODEL_KEY

    ckpt_path = os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "").strip()
    if not ckpt_path:
        raise RuntimeError(
            "PUFFER_POLICY_CHECKPOINT_PATH must be set when using policy_callable_checkpoint.py"
        )
    key = (
        os.path.abspath(ckpt_path),
        _env_int("PUFFER_POLICY_INPUT_DIM", 23),
        _env_int("PUFFER_POLICY_HIDDEN_DIM", 128),
        _env_int("PUFFER_POLICY_NUM_LAYERS", 3),
        _env_int("PUFFER_POLICY_NUM_ACTIONS", 4),
        _env_int("PUFFER_POLICY_LAYOUT_PRECISION_BYTES", 2),
        _env_bool("PUFFER_POLICY_NATIVE_BF16", False),
    )
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = CheckpointPolicy.load(
            key[0],
            input_dim=key[1],
            hidden_dim=key[2],
            num_layers=key[3],
            num_actions=key[4],
            layout_precision_bytes=key[5],
            native_bf16=key[6],
        )
        _MODEL_KEY = key
    return _MODEL


def reset() -> None:
    model = _resolve_model()
    model.reset_state()


def infer(observation: Sequence[float]) -> list[float]:
    model = _resolve_model()
    actions = model.infer(observation)
    return _apply_action_bias(observation, actions, _resolve_action_bias())
