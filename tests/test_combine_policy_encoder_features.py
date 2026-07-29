from pathlib import Path

import numpy as np
import pytest

from scripts.combine_policy_encoder_features import combine


def _checkpoint(path: Path, values: np.ndarray) -> Path:
    values.astype(np.float32).tofile(path)
    return path


def test_combines_only_declared_encoder_features(tmp_path: Path, monkeypatch):
    input_dim = 3
    hidden_dim = 2
    # The combiner validates with CheckpointPolicy after writing. Keep this unit
    # fixture focused on byte-level residual isolation rather than checkpoint ABI.
    monkeypatch.setattr(
        "scripts.combine_policy_encoder_features.CheckpointPolicy.load",
        lambda *_args, **_kwargs: None,
    )
    base_values = np.arange(30, dtype=np.float32)
    base = _checkpoint(tmp_path / "base.bin", base_values)
    feature0_values = base_values.copy()
    feature0_values[: input_dim * hidden_dim].reshape(hidden_dim, input_dim)[:, 0] += 2
    feature2_values = base_values.copy()
    feature2_values[: input_dim * hidden_dim].reshape(hidden_dim, input_dim)[:, 2] -= 3
    feature0 = _checkpoint(tmp_path / "feature0.bin", feature0_values)
    feature2 = _checkpoint(tmp_path / "feature2.bin", feature2_values)
    output = tmp_path / "output.bin"

    combine(
        base,
        output,
        [(0, feature0, -2.0), (2, feature2, 0.5)],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=1,
    )
    actual = np.fromfile(output, dtype=np.float32)
    expected = base_values.copy()
    encoder = expected[: input_dim * hidden_dim].reshape(hidden_dim, input_dim)
    encoder[:, 0] -= 4
    encoder[:, 2] -= 1.5
    np.testing.assert_array_equal(actual, expected)


def test_rejects_candidate_changes_outside_declared_feature(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "scripts.combine_policy_encoder_features.CheckpointPolicy.load",
        lambda *_args, **_kwargs: None,
    )
    base = _checkpoint(tmp_path / "base.bin", np.arange(20, dtype=np.float32))
    changed = np.arange(20, dtype=np.float32)
    changed[-1] += 1
    candidate = _checkpoint(tmp_path / "candidate.bin", changed)
    with pytest.raises(ValueError, match="outside the encoder"):
        combine(
            base,
            tmp_path / "out.bin",
            [(1, candidate, 1.0)],
            input_dim=3,
            hidden_dim=2,
            num_actions=1,
        )
