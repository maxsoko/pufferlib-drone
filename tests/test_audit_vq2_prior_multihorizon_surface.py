import subprocess
import sys
from pathlib import Path

import torch

from scripts.audit_vq2_prior_multihorizon_surface import (
    HORIZONS,
    _anchors,
    _eligible,
)


ROOT = Path(__file__).resolve().parents[1]


def test_anchors_keep_endpoint_inside_sequence() -> None:
    for horizon in HORIZONS:
        anchors = _anchors(128, horizon, burn_in=32, stride=8)
        assert int(anchors[0]) == 31
        assert int(anchors[-1] + horizon) <= 127


def test_eligibility_requires_every_horizon_to_improve() -> None:
    before = {
        str(h): {"progress_mae": 0.2, "kl": 0.001, "action_rmse": 0.0, "action_max_abs": 0.0}
        for h in HORIZONS
    }
    after = {
        str(h): {"progress_mae": 0.1, "kl": 0.001, "action_rmse": 0.0001, "action_max_abs": 0.001}
        for h in HORIZONS
    }
    assert _eligible(before, after)
    after["16"]["progress_mae"] = 0.21
    assert not _eligible(before, after)


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_prior_multihorizon_surface.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n723-report" in completed.stdout
