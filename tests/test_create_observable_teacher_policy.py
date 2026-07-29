from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.policy_callable_checkpoint import CheckpointPolicy, _align


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create_observable_teacher_policy.py"


def _checkpoint(
    path: Path,
    *,
    input_dim: int = 32,
    hidden_dim: int = 128,
    precision_bytes: int = 2,
) -> None:
    counts = (
        hidden_dim * input_dim,
        5 * hidden_dim,
        4,
        *([3 * hidden_dim * hidden_dim] * 3),
    )
    total = 0
    for count in counts:
        total = _align(total + count, precision_bytes)
    np.zeros(total, dtype=np.float32).tofile(path)


def test_initializer_writes_the_declared_phase32_layout(tmp_path):
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    _checkpoint(source)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--input-dim",
            "32",
            "--forward-speed",
            "6.0",
            "--forward-kp",
            "0.1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    policy = CheckpointPolicy.load(str(output), input_dim=32)
    assert policy.encoder.shape == (128, 32)
    assert policy.encoder[32, 31] == 1.0
    np.testing.assert_allclose(policy.decoder[0, 0], -0.144, atol=1e-6)
    assert policy.decoder[0, 1] < -0.17


def test_initializer_rejects_nonpositive_forward_speed(tmp_path):
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    _checkpoint(source)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--input-dim",
            "32",
            "--forward-speed",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--forward-speed must be positive" in result.stderr


def test_initializer_supports_fp32_native_alignment(tmp_path):
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    _checkpoint(source, precision_bytes=4)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--input-dim",
            "32",
            "--checkpoint-layout-precision-bytes",
            "4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    policy = CheckpointPolicy.load(
        str(output), input_dim=32, layout_precision_bytes=4)
    assert policy.encoder[32, 31] == 1.0
    assert policy.mingru_proj[0][256, 0] == -10.0


def test_initializer_supports_linear_gate_feature_scaling(tmp_path):
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    _checkpoint(source, precision_bytes=4)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            "--input-dim",
            "32",
            "--checkpoint-layout-precision-bytes",
            "4",
            "--forward-speed",
            "1.2",
            "--forward-kp",
            "0.35",
            "--pitch-action-scale",
            "0.24",
            "--linear-gate-features",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    policy = CheckpointPolicy.load(
        str(output), input_dim=32, layout_precision_bytes=4)
    assert policy.decoder[0, 0] == pytest.approx(-0.1008)
    assert policy.decoder[0, 1] == pytest.approx(-0.42)
