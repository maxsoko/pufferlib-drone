import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "policy_callable_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("policy_callable_checkpoint", MODULE_PATH)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


def _append_aligned(storage: list[float], values: list[float]) -> None:
    storage.extend(values)
    while len(storage) % 8 != 0:
        storage.append(0.0)


def _build_test_checkpoint(path: Path) -> None:
    # dims: input=3, hidden=2, layers=1, actions=2
    values: list[float] = []

    # encoder [hidden, input]
    _append_aligned(values, [
        0.5, 0.1, -0.2,
        -0.3, 0.2, 0.4,
    ])
    # decoder [actions+1, hidden]
    _append_aligned(values, [
        1.0, -0.5,
        -0.3, 0.8,
        0.1, 0.2,
    ])
    # log_std [actions]
    _append_aligned(values, [0.0, 0.0])
    # mingru proj [3*hidden, hidden]
    _append_aligned(values, [
        0.2, -0.1,
        0.1, 0.3,
        -0.2, 0.2,
        0.25, -0.15,
        0.12, 0.07,
        -0.05, 0.09,
    ])

    np.array(values, dtype=np.float32).tofile(path)


def test_checkpoint_policy_infer_is_deterministic_after_reset(tmp_path, monkeypatch):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "3")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "2")

    policy._MODEL = None
    policy._MODEL_KEY = None
    obs = [0.3, -0.2, 0.1]
    first = policy.infer(obs)
    second = policy.infer(obs)
    assert first != second
    policy.reset()
    third = policy.infer(obs)
    assert third == pytest.approx(first)


def test_checkpoint_policy_validates_observation_shape(tmp_path, monkeypatch):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "3")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "2")

    model = policy._resolve_model()
    with pytest.raises(ValueError):
        model.infer([0.1, 0.2])


def test_policy_requires_checkpoint_env(monkeypatch):
    monkeypatch.delenv("PUFFER_POLICY_CHECKPOINT_PATH", raising=False)
    policy._MODEL = None
    policy._MODEL_KEY = None
    with pytest.raises(RuntimeError):
        policy._resolve_model()
