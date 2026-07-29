import numpy as np
import pytest

from scripts.constrained_cma import FullCovarianceCMA


def test_ask_is_antithetic_around_mean():
    cma = FullCovarianceCMA(4, population_size=6, parent_count=3, seed=9)
    candidates = cma.ask()
    assert candidates.shape == (6, 4)
    assert candidates[:3] + candidates[3:] == pytest.approx(0.0)


def test_update_keeps_covariance_positive_and_moves_mean():
    cma = FullCovarianceCMA(3, population_size=6, parent_count=3, seed=4)
    candidates = cma.ask()
    ranking = np.argsort(np.square(candidates - 0.5).sum(axis=1)).tolist()
    cma.tell(ranking)
    assert np.linalg.norm(cma.mean) > 0.0
    assert np.linalg.eigvalsh(cma.covariance).min() > 0.0
    assert cma.generation == 1


def test_saved_state_resumes_exact_next_population(tmp_path):
    original = FullCovarianceCMA(5, population_size=8, parent_count=4, seed=3385)
    first = original.ask()
    original.tell(np.argsort(np.square(first).sum(axis=1)).tolist())
    state_path = tmp_path / "state.npz"
    original.save(state_path)
    resumed = FullCovarianceCMA.load(state_path)
    assert resumed.ask() == pytest.approx(original.ask())
