import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.continue_vq2_invariant_prior_offline import (
    FIXED_CROP_STARTS,
    _balanced_pairs,
    _sampling_schedule,
    _sampling_summary,
    _project_prior_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_pairs_cover_every_event_crop_once() -> None:
    pairs = _balanced_pairs(272, FIXED_CROP_STARTS, 715)
    assert pairs.shape == (1904, 2)
    assert len(np.unique(pairs, axis=0)) == 1904


def test_three_epoch_schedule_is_exactly_balanced() -> None:
    schedule = _sampling_schedule(
        272, FIXED_CROP_STARTS, seed=715, batch_size=16, updates=357
    )
    summary = _sampling_summary(schedule, 272)
    assert schedule.shape == (357, 16, 2)
    assert summary["draws"] == 5712
    assert summary["unique_pairs"] == 1904
    assert summary["unseen_pairs"] == 0
    assert summary["pair_count_minimum"] == 3
    assert summary["pair_count_maximum"] == 3
    assert summary["event_count_minimum"] == 21
    assert summary["event_count_maximum"] == 21


def test_one_update_smoke_draws_sixteen_unique_pairs() -> None:
    schedule = _sampling_schedule(
        272, FIXED_CROP_STARTS, seed=715, batch_size=16, updates=1
    )
    assert len(np.unique(schedule.reshape(-1, 2), axis=0)) == 16


def test_prior_projection_is_elementwise_and_bounded() -> None:
    parameter = torch.nn.Parameter(torch.tensor([-0.2, 0.02, 0.4]))
    projected = _project_prior_parameters(
        [("rssm.prior.test", parameter)],
        {"rssm.prior.test": torch.zeros(3)},
        0.05,
    )
    assert projected == 2
    assert torch.equal(parameter.detach(), torch.tensor([-0.05, 0.02, 0.05]))


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/continue_vq2_invariant_prior_offline.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n714-report" in completed.stdout
