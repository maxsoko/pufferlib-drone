import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_prior_raw_continuation_surface import LEARNING_RATES


ROOT = Path(__file__).resolve().parents[1]


def test_rate_surface_is_fixed_and_increasing() -> None:
    assert LEARNING_RATES == tuple(sorted(LEARNING_RATES))
    assert LEARNING_RATES[-1] == 0.013


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_prior_raw_continuation_surface.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n732-report" in completed.stdout
    assert "--progress-every" in completed.stdout
