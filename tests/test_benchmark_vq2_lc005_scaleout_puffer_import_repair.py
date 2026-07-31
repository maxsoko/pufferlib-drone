from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_lc005_executes_from_repo_root_and_exposes_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_vq2_lc005_scaleout_puffer_import_repair.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "LC004 scale" in result.stdout
