import subprocess
import sys
from pathlib import Path

from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _sampling_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def test_resume_uses_distinct_second_batch() -> None:
    schedule = _sampling_schedule(272, FIXED_CROP_STARTS, seed=719, batch_size=16, updates=2)
    first = {tuple(row) for row in schedule[0]}
    second = {tuple(row) for row in schedule[1]}
    assert len(first) == len(second) == 16
    assert first.isdisjoint(second)


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/continue_vq2_prior_progress_resume.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n721-report" in completed.stdout
