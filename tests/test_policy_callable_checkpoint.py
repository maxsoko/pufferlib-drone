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
    assert policy._resolve_model().native_bf16 is False
    assert policy._resolve_model().layout_precision_bytes == 2


def test_callable_can_explicitly_select_bf16_arithmetic(tmp_path, monkeypatch):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "3")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "2")
    monkeypatch.setenv("PUFFER_POLICY_NATIVE_BF16", "1")

    policy._MODEL = None
    policy._MODEL_KEY = None
    assert policy._resolve_model().native_bf16 is True


def test_bf16_mode_rounds_weights_and_recurrent_state(tmp_path):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    model = policy.CheckpointPolicy.load(
        str(ckpt),
        input_dim=3,
        hidden_dim=2,
        num_layers=1,
        num_actions=2,
        native_bf16=True,
    )

    model.infer([0.3, -0.2, 0.1])
    assert np.array_equal(model.encoder, policy._round_bf16(model.encoder))
    assert np.array_equal(model.state, policy._round_bf16(model.state))


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


def test_native_unpadded_checkpoint_uses_aligned_cuda_view(tmp_path):
    # All tensors except log_std are naturally 8-float aligned. CUDA leaves six
    # interior slots after log_std, while save_weights truncates six values from
    # the final MinGRU tensor because it writes only the logical element count.
    input_dim = 1
    hidden_dim = 8
    num_layers = 1
    num_actions = 2
    logical_count = (
        hidden_dim * input_dim
        + (num_actions + 1) * hidden_dim
        + num_actions
        + 3 * hidden_dim * hidden_dim
    )
    aligned: list[float] = []
    _append_aligned(aligned, [1.0] * (hidden_dim * input_dim))
    _append_aligned(aligned, [2.0] * ((num_actions + 1) * hidden_dim))
    _append_aligned(aligned, [3.0] * num_actions)
    _append_aligned(aligned, [4.0] * (3 * hidden_dim * hidden_dim))
    checkpoint = tmp_path / "native-unpadded.bin"
    np.asarray(aligned[:logical_count], dtype=np.float32).tofile(checkpoint)

    model = policy.CheckpointPolicy.load(
        str(checkpoint),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )

    flat_proj = model.mingru_proj[0].reshape(-1)
    assert flat_proj[:-6] == pytest.approx(4.0)
    assert flat_proj[-6:] == pytest.approx(0.0)


def test_fp32_layout_uses_four_float_tensor_alignment(tmp_path):
    input_dim = 1
    hidden_dim = 8
    num_layers = 1
    num_actions = 2
    counts = [8, 24, 2, 192]
    logical_count = sum(counts)
    aligned = np.zeros(228, dtype=np.float32)
    index = 0
    for value, count in enumerate(counts, start=1):
        aligned[index:index + count] = value
        index = (index + count + 3) & ~3
    checkpoint = tmp_path / "fp32-layout.bin"
    aligned[:logical_count].tofile(checkpoint)

    model = policy.CheckpointPolicy.load(
        str(checkpoint),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=4,
    )

    flat_proj = model.mingru_proj[0].reshape(-1)
    assert flat_proj[:-2] == pytest.approx(4.0)
    assert flat_proj[-2:] == pytest.approx(0.0)


def test_callable_selects_fp32_checkpoint_layout(tmp_path, monkeypatch):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "3")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "2")
    monkeypatch.setenv("PUFFER_POLICY_LAYOUT_PRECISION_BYTES", "4")

    policy._MODEL = None
    policy._MODEL_KEY = None
    assert policy._resolve_model().layout_precision_bytes == 4


def test_gate_indexed_action_bias_uses_normalized_progress_only(
    tmp_path, monkeypatch
):
    ckpt = tmp_path / "tiny.bin"
    _build_test_checkpoint(ckpt)
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", str(ckpt))
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "3")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "2")

    config = policy.GateActionBias(
        gate_index=2,
        progress_observation_index=2,
        progress_denominator=6,
        roll=-0.0025,
        thrust=0.00125,
    )
    actions = [0.1, 0.2, 0.3, 0.4]
    active = policy._apply_action_bias([0.0, 0.0, 2.0 / 6.0], actions, config)
    inactive = policy._apply_action_bias([0.0, 0.0, 1.0 / 6.0], actions, config)

    assert active == pytest.approx([0.1, 0.1975, 0.30125, 0.4])
    assert inactive == pytest.approx(actions)


def test_gate_indexed_action_bias_rejects_wrong_observation_contract():
    config = policy.GateActionBias(gate_index=2)

    with pytest.raises(ValueError, match="normalized progress observation"):
        policy._apply_action_bias([0.0] * 23, [0.0] * 4, config)
    with pytest.raises(ValueError, match="non-quantized progress"):
        policy._apply_action_bias([0.0] * 23 + [0.2], [0.0] * 4, config)


def test_gate_indexed_action_bias_table_preserves_multiple_gate_corrections():
    config = policy.GateActionBias(
        table=(
            (1, (0.0, 0.003, -0.002, 0.0), False, 0.0, 0.0),
            (2, (0.0, -0.0125, 0.00125, 0.0), False, 0.0, 0.0),
        )
    )
    actions = [0.1, 0.2, 0.3, 0.4]

    gate_one = policy._apply_action_bias(
        [0.0] * 23 + [1.0 / 6.0], actions, config
    )
    gate_two = policy._apply_action_bias(
        [0.0] * 23 + [2.0 / 6.0], actions, config
    )
    gate_three = policy._apply_action_bias(
        [0.0] * 23 + [3.0 / 6.0], actions, config
    )

    assert gate_one == pytest.approx([0.1, 0.203, 0.298, 0.4])
    assert gate_two == pytest.approx([0.1, 0.1875, 0.30125, 0.4])
    assert gate_three == pytest.approx(actions)


def test_gate_indexed_action_bias_can_release_on_camera_dropout():
    config = policy.GateActionBias(
        table=((1, (0.0, 0.0025, -0.00125, 0.0), True, 0.0, 0.0),)
    )
    actions = [0.1, 0.2, 0.3, 0.4]
    visible_observation = [0.0] * 24
    visible_observation[10] = 1.0
    visible_observation[23] = 1.0 / 6.0
    hidden_observation = visible_observation.copy()
    hidden_observation[10] = 0.0

    visible = policy._apply_action_bias(visible_observation, actions, config)
    hidden = policy._apply_action_bias(hidden_observation, actions, config)

    assert visible == pytest.approx([0.1, 0.2025, 0.29875, 0.4])
    assert hidden == pytest.approx(actions)


def test_gate_indexed_action_bias_can_require_safe_camera_alignment():
    config = policy.GateActionBias(
        table=((1, (0.0, 0.0025, -0.00125, 0.0), False, 0.8, 0.0),)
    )
    actions = [0.1, 0.2, 0.3, 0.4]
    centered = [0.0] * 24
    centered[17] = 0.9
    centered[23] = 1.0 / 6.0
    off_center = centered.copy()
    off_center[17] = 0.7

    enabled = policy._apply_action_bias(centered, actions, config)
    disabled = policy._apply_action_bias(off_center, actions, config)

    assert enabled == pytest.approx([0.1, 0.2025, 0.29875, 0.4])
    assert disabled == pytest.approx(actions)


def test_gate_indexed_action_bias_can_release_before_gate_crossing():
    config = policy.GateActionBias(
        table=((1, (0.0, 0.0025, -0.00125, 0.0), False, 0.0, 12.0),)
    )
    actions = [0.1, 0.2, 0.3, 0.4]
    far = [0.0] * 24
    far[11] = float(np.tanh(1.3))
    far[23] = 1.0 / 6.0
    close = far.copy()
    close[11] = float(np.tanh(0.8))

    enabled = policy._apply_action_bias(far, actions, config)
    disabled = policy._apply_action_bias(close, actions, config)

    assert enabled == pytest.approx([0.1, 0.2025, 0.29875, 0.4])
    assert disabled == pytest.approx(actions)
