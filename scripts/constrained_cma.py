#!/usr/bin/env python3
"""Small resumable full-covariance CMA-ES with antithetic sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CMAState:
    dimension: int
    population_size: int
    parent_count: int
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    evolution_path_covariance: np.ndarray
    evolution_path_sigma: np.ndarray
    generation: int
    rng_state: dict[str, Any]


class FullCovarianceCMA:
    def __init__(
        self,
        dimension: int,
        *,
        population_size: int = 24,
        parent_count: int = 12,
        sigma: float = 1.0,
        seed: int = 3385,
        state: CMAState | None = None,
    ) -> None:
        if dimension <= 1:
            raise ValueError("CMA dimension must be greater than one")
        if population_size <= 2 or population_size % 2:
            raise ValueError("antithetic population size must be even and greater than two")
        if not 1 <= parent_count <= population_size:
            raise ValueError("parent count must be within the population")
        self.dimension = int(dimension)
        self.population_size = int(population_size)
        self.parent_count = int(parent_count)
        self.rng = np.random.default_rng(seed)
        self._last_y: np.ndarray | None = None
        self._last_candidates: np.ndarray | None = None

        raw_weights = np.log(parent_count + 0.5) - np.log(
            np.arange(1, parent_count + 1)
        )
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = float(1.0 / np.square(self.weights).sum())
        n = self.dimension
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(
            0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0
        ) + self.c_sigma
        self.c_cov_path = (4.0 + self.mu_eff / n) / (
            n + 4.0 + 2.0 * self.mu_eff / n
        )
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) ** 2 + self.mu_eff),
        )
        self.chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        if state is None:
            self.mean = np.zeros(n, dtype=np.float64)
            self.sigma = float(sigma)
            self.covariance = np.eye(n, dtype=np.float64)
            self.path_covariance = np.zeros(n, dtype=np.float64)
            self.path_sigma = np.zeros(n, dtype=np.float64)
            self.generation = 0
        else:
            self._restore(state)

    def _restore(self, state: CMAState) -> None:
        if (
            state.dimension != self.dimension
            or state.population_size != self.population_size
            or state.parent_count != self.parent_count
        ):
            raise ValueError("saved CMA dimensions do not match")
        self.mean = np.asarray(state.mean, dtype=np.float64).copy()
        self.sigma = float(state.sigma)
        self.covariance = np.asarray(state.covariance, dtype=np.float64).copy()
        self.path_covariance = np.asarray(
            state.evolution_path_covariance, dtype=np.float64
        ).copy()
        self.path_sigma = np.asarray(state.evolution_path_sigma, dtype=np.float64).copy()
        self.generation = int(state.generation)
        self.rng.bit_generator.state = state.rng_state

    def ask(self) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (self.covariance + self.covariance.T)
        )
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        half = self.population_size // 2
        standard = self.rng.standard_normal((half, self.dimension))
        standard = np.concatenate([standard, -standard], axis=0)
        y = standard @ transform.T
        candidates = self.mean + self.sigma * y
        self._last_y = y
        self._last_candidates = candidates
        return candidates.copy()

    def tell(self, ranking: list[int] | np.ndarray) -> None:
        if self._last_y is None or self._last_candidates is None:
            raise RuntimeError("ask must precede tell")
        order = np.asarray(ranking, dtype=np.int64)
        if sorted(order.tolist()) != list(range(self.population_size)):
            raise ValueError("ranking must be a permutation of the population")

        selected_y = self._last_y[order[: self.parent_count]]
        weighted_y = self.weights @ selected_y
        old_covariance = self.covariance.copy()
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (old_covariance + old_covariance.T)
        )
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        self.mean = self.mean + self.sigma * weighted_y
        self.path_sigma = (
            (1.0 - self.c_sigma) * self.path_sigma
            + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff)
            * (inverse_sqrt @ weighted_y)
        )
        path_norm = float(np.linalg.norm(self.path_sigma))
        normalization = np.sqrt(
            max(1e-20, 1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
        )
        h_sigma = float(
            path_norm / normalization / self.chi_n
            < 1.4 + 2.0 / (self.dimension + 1.0)
        )
        self.path_covariance = (
            (1.0 - self.c_cov_path) * self.path_covariance
            + h_sigma
            * np.sqrt(self.c_cov_path * (2.0 - self.c_cov_path) * self.mu_eff)
            * weighted_y
        )
        rank_mu = np.zeros_like(old_covariance)
        for weight, direction in zip(self.weights, selected_y, strict=True):
            rank_mu += weight * np.outer(direction, direction)
        self.covariance = (
            (1.0 - self.c1 - self.c_mu) * old_covariance
            + self.c1
            * (
                np.outer(self.path_covariance, self.path_covariance)
                + (1.0 - h_sigma)
                * self.c_cov_path
                * (2.0 - self.c_cov_path)
                * old_covariance
            )
            + self.c_mu * rank_mu
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.sigma *= float(
            np.exp(self.c_sigma / self.d_sigma * (path_norm / self.chi_n - 1.0))
        )
        self.generation += 1
        self._last_y = None
        self._last_candidates = None

    def state(self) -> CMAState:
        return CMAState(
            dimension=self.dimension,
            population_size=self.population_size,
            parent_count=self.parent_count,
            mean=self.mean.copy(),
            sigma=self.sigma,
            covariance=self.covariance.copy(),
            evolution_path_covariance=self.path_covariance.copy(),
            evolution_path_sigma=self.path_sigma.copy(),
            generation=self.generation,
            rng_state=self.rng.bit_generator.state,
        )

    def diagnostics(self) -> dict[str, float | int]:
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return {
            "generation": self.generation,
            "sigma": self.sigma,
            "mean_l2": float(np.linalg.norm(self.mean)),
            "covariance_min_eigenvalue": float(eigenvalues.min()),
            "covariance_max_eigenvalue": float(eigenvalues.max()),
            "covariance_condition_number": float(
                eigenvalues.max() / max(eigenvalues.min(), 1e-300)
            ),
        }

    def save(self, path: Path) -> None:
        state = self.state()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dimension=np.asarray(state.dimension),
            population_size=np.asarray(state.population_size),
            parent_count=np.asarray(state.parent_count),
            mean=state.mean,
            sigma=np.asarray(state.sigma),
            covariance=state.covariance,
            evolution_path_covariance=state.evolution_path_covariance,
            evolution_path_sigma=state.evolution_path_sigma,
            generation=np.asarray(state.generation),
            rng_state=np.asarray(json.dumps(state.rng_state, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "FullCovarianceCMA":
        data = np.load(path, allow_pickle=False)
        state = CMAState(
            dimension=int(data["dimension"]),
            population_size=int(data["population_size"]),
            parent_count=int(data["parent_count"]),
            mean=np.asarray(data["mean"]),
            sigma=float(data["sigma"]),
            covariance=np.asarray(data["covariance"]),
            evolution_path_covariance=np.asarray(data["evolution_path_covariance"]),
            evolution_path_sigma=np.asarray(data["evolution_path_sigma"]),
            generation=int(data["generation"]),
            rng_state=json.loads(str(data["rng_state"])),
        )
        return cls(
            state.dimension,
            population_size=state.population_size,
            parent_count=state.parent_count,
            state=state,
        )
