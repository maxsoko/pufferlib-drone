from pathlib import Path

import numpy as np

from scripts.policy_callable_checkpoint import _align, _align8
from scripts.set_policy_log_std import set_log_std


def _checkpoint(
    path: Path, *, input_dim: int = 32, hidden_dim: int = 8, num_actions: int = 4
) -> np.ndarray:
    counts = (
        hidden_dim * input_dim,
        (num_actions + 1) * hidden_dim,
        num_actions,
        *([3 * hidden_dim * hidden_dim] * 3),
    )
    total = 0
    for count in counts:
        total = _align8(total + count)
    values = np.arange(total, dtype=np.float32) / np.float32(100.0)
    values.tofile(path)
    return values


def test_set_log_std_uses_declared_full_policy_input_width(tmp_path):
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    original = _checkpoint(source)

    before, after = set_log_std(
        source,
        output,
        log_std=-3.0,
        input_dim=32,
        hidden_dim=8,
        num_actions=4,
    )

    updated = np.fromfile(output, dtype=np.float32)
    log_std_start = _align8(8 * 32 + 5 * 8)
    changed = np.flatnonzero(updated != original)
    np.testing.assert_array_equal(
        changed, np.arange(log_std_start, log_std_start + 4)
    )
    np.testing.assert_array_equal(before, original[log_std_start : log_std_start + 4])
    np.testing.assert_array_equal(after, np.full(4, -3.0, dtype=np.float32))
    np.testing.assert_array_equal(updated[:log_std_start], original[:log_std_start])
    np.testing.assert_array_equal(updated[log_std_start + 4 :], original[log_std_start + 4 :])


def test_set_log_std_supports_fp32_aligned_checkpoint(tmp_path):
    source = tmp_path / "source_fp32.bin"
    output = tmp_path / "output_fp32.bin"
    input_dim = 31
    hidden_dim = 7
    num_actions = 3
    counts = (
        hidden_dim * input_dim,
        (num_actions + 1) * hidden_dim,
        num_actions,
        *([3 * hidden_dim * hidden_dim] * 3),
    )
    total = 0
    for count in counts:
        total = _align(total + count, 4)
    original = np.arange(total, dtype=np.float32) / np.float32(100.0)
    original.tofile(source)

    before, after = set_log_std(
        source,
        output,
        log_std=-6.0,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        layout_precision_bytes=4,
    )

    updated = np.fromfile(output, dtype=np.float32)
    log_std_start = _align(
        _align(hidden_dim * input_dim, 4) + (num_actions + 1) * hidden_dim,
        4,
    )
    changed = np.flatnonzero(updated != original)
    np.testing.assert_array_equal(
        changed, np.arange(log_std_start, log_std_start + num_actions)
    )
    np.testing.assert_array_equal(before, original[log_std_start:log_std_start + num_actions])
    np.testing.assert_array_equal(after, np.full(num_actions, -6.0, dtype=np.float32))
