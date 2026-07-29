import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_raw_common_descent_prior_validation.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n731-report" in completed.stdout
    assert "--validation-dataset" in completed.stdout
