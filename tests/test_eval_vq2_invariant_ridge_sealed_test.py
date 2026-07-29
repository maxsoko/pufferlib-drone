import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.eval_vq2_invariant_ridge_sealed_test import _load_npz_once


ROOT = Path(__file__).resolve().parents[1]


def test_npz_loader_hashes_and_materializes_one_payload(tmp_path: Path) -> None:
    path = tmp_path / "partition.npz"
    np.savez(path, vector_step=np.asarray([1, 2]), value=np.asarray([[3.0], [4.0]]))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    arrays, digest, size = _load_npz_once(path)
    assert digest == expected
    assert size == path.stat().st_size
    assert arrays["vector_step"].tolist() == [1, 2]
    assert arrays["value"].tolist() == [[3.0], [4.0]]


def test_direct_cli_is_mutation_free_and_explicitly_test_only() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_vq2_invariant_ridge_sealed_test.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--test-dataset" in result.stdout
    assert "--n711-report" in result.stdout
    assert "train-dataset" not in result.stdout
    assert "validation-dataset" not in result.stdout
    assert "output-checkpoint" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "actor-steps" not in result.stdout
    assert "probe-steps" not in result.stdout
