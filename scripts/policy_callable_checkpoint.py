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
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _align8(idx: int) -> int:
    return (idx + 7) & ~7


def _take_aligned(weights: np.ndarray, idx: int, count: int) -> tuple[np.ndarray, int]:
    chunk = weights[idx:idx + count]
    if len(chunk) != count:
        raise ValueError(f"checkpoint ended early: requested {count}, got {len(chunk)}")
    return chunk, _align8(idx + count)


def _take_raw(weights: np.ndarray, idx: int, count: int) -> tuple[np.ndarray, int]:
    chunk = weights[idx:idx + count]
    if len(chunk) != count:
        raise ValueError(f"checkpoint ended early: requested {count}, got {len(chunk)}")
    return chunk, idx + count


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mingru_g(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, x + 0.5, _sigmoid(x))


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

    @classmethod
    def load(
        cls,
        path: str,
        *,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_actions: int = 4,
    ) -> "CheckpointPolicy":
        weights = np.fromfile(path, dtype=np.float32)
        parse_error = None
        parsed = None
        for align in (True, False):
            idx = 0
            take_fn = _take_aligned if align else _take_raw
            try:
                encoder, idx = take_fn(weights, idx, hidden_dim * input_dim)
                encoder = encoder.reshape(hidden_dim, input_dim)

                decoder_rows = num_actions + 1
                decoder, idx = take_fn(weights, idx, decoder_rows * hidden_dim)
                decoder = decoder.reshape(decoder_rows, hidden_dim)

                log_std, idx = take_fn(weights, idx, num_actions)

                mingru_proj: list[np.ndarray] = []
                for _layer in range(num_layers):
                    proj, idx = take_fn(weights, idx, 3 * hidden_dim * hidden_dim)
                    mingru_proj.append(proj.reshape(3 * hidden_dim, hidden_dim))
                parsed = (encoder, decoder, log_std, mingru_proj)
                break
            except ValueError as exc:
                parse_error = exc
                continue
        if parsed is None:
            raise ValueError(f"failed to parse checkpoint layout for {path}: {parse_error}")
        encoder, decoder, log_std, mingru_proj = parsed

        state = np.zeros((num_layers, hidden_dim), dtype=np.float32)
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            encoder=encoder.astype(np.float32),
            decoder=decoder.astype(np.float32),
            log_std=log_std.astype(np.float32),
            mingru_proj=[proj.astype(np.float32) for proj in mingru_proj],
            state=state,
        )

    def reset_state(self) -> None:
        self.state.fill(0.0)

    def infer(self, observation: Sequence[float]) -> list[float]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != self.input_dim:
            raise ValueError(f"expected observation shape ({self.input_dim},), got {obs.shape}")

        h = self.encoder @ obs
        for layer_idx in range(self.num_layers):
            proj = self.mingru_proj[layer_idx] @ h
            hidden = proj[0:self.hidden_dim]
            gate = proj[self.hidden_dim : 2 * self.hidden_dim]
            highway_proj = proj[2 * self.hidden_dim : 3 * self.hidden_dim]

            out = self.state[layer_idx] + _sigmoid(gate) * (_mingru_g(hidden) - self.state[layer_idx])
            h = _sigmoid(highway_proj) * out + (1.0 - _sigmoid(highway_proj)) * h
            self.state[layer_idx] = out

        decoded = self.decoder @ h
        actions = decoded[: self.num_actions]
        return [float(max(-1.0, min(1.0, value))) for value in actions]


_MODEL: CheckpointPolicy | None = None
_MODEL_KEY: tuple[str, int, int, int, int] | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    return int(value)


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
    )
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = CheckpointPolicy.load(
            key[0],
            input_dim=key[1],
            hidden_dim=key[2],
            num_layers=key[3],
            num_actions=key[4],
        )
        _MODEL_KEY = key
    return _MODEL


def reset() -> None:
    model = _resolve_model()
    model.reset_state()


def infer(observation: Sequence[float]) -> list[float]:
    model = _resolve_model()
    return model.infer(observation)
