"""Trust-region CMA-ES utilities for behavior-guided policy search.

The LLM proposes a complete policy vector.  The proposal is never injected into
the CMA evolution paths directly: it is projected into a Mahalanobis trust
region and used only as the sampling center for the current generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GuidedCenter:
    """Result of projecting an external proposal into the CMA trust region."""

    center: np.ndarray
    mahalanobis_distance: float
    applied_fraction: float


class TrustRegionCMA:
    """Small self-contained CMA-ES implementation with external mean guidance.

    Full covariance is used for modest policy dimensions.  For larger policies,
    diagonal covariance keeps memory and eigendecomposition costs practical.
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float = 0.5,
        population_size: Optional[int] = None,
        covariance_mode: str = "auto",
        full_covariance_max_dim: int = 128,
        lower_bound: float = -6.0,
        upper_bound: float = 6.0,
        seed: Optional[int] = None,
    ):
        self.mean = np.asarray(mean, dtype=float).reshape(-1).copy()
        self.dimension = self.mean.size
        if self.dimension == 0:
            raise ValueError("CMA mean must contain at least one parameter")
        if sigma <= 0:
            raise ValueError("CMA sigma must be positive")
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")

        self.sigma = float(sigma)
        self.initial_sigma = float(sigma)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.rng = np.random.default_rng(seed)
        self.generation = 0

        default_population = 4 + int(3 * np.log(self.dimension))
        self.population_size = int(population_size or default_population)
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")

        if covariance_mode not in {"auto", "full", "diagonal"}:
            raise ValueError("covariance_mode must be auto, full, or diagonal")
        if covariance_mode == "auto":
            covariance_mode = (
                "full" if self.dimension <= full_covariance_max_dim else "diagonal"
            )
        self.covariance_mode = covariance_mode

        self.mu = self.population_size // 2
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        n = self.dimension
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) ** 2 + self.mu_eff),
        )
        self.damps = (
            1.0
            + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        self.p_c = np.zeros(n, dtype=float)
        self.p_sigma = np.zeros(n, dtype=float)
        if self.covariance_mode == "full":
            self.covariance = np.eye(n, dtype=float)
        else:
            self.covariance = np.ones(n, dtype=float)

    def _eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Return covariance square root and inverse square root."""
        if self.covariance_mode == "diagonal":
            diag = np.maximum(self.covariance, 1e-20)
            return np.sqrt(diag), 1.0 / np.sqrt(diag)

        covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        sqrt_cov = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        inv_sqrt_cov = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        return sqrt_cov, inv_sqrt_cov

    @staticmethod
    def _transform(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if matrix.ndim == 1:
            return matrix * vector
        return matrix @ vector

    def repair(self, parameters: np.ndarray) -> np.ndarray:
        """Reflect parameters into the configured box without hard clipping."""
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if values.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} parameters, received {values.size}"
            )
        span = self.upper_bound - self.lower_bound
        shifted = np.mod(values - self.lower_bound, 2.0 * span)
        reflected = self.lower_bound + span - np.abs(shifted - span)
        return reflected

    def project_proposal(
        self,
        proposal: np.ndarray,
        trust_region_radius: float,
        guidance_alpha: float = 1.0,
    ) -> GuidedCenter:
        """Project a complete LLM policy toward the CMA mean in whitened space."""
        if trust_region_radius <= 0:
            raise ValueError("trust_region_radius must be positive")
        if not 0.0 <= guidance_alpha <= 1.0:
            raise ValueError("guidance_alpha must be in [0, 1]")

        proposal = self.repair(proposal)
        delta = proposal - self.mean
        _, inv_sqrt_cov = self._eigendecomposition()
        whitened = self._transform(inv_sqrt_cov, delta) / self.sigma
        distance = float(np.linalg.norm(whitened))
        trust_fraction = min(1.0, trust_region_radius / max(distance, 1e-12))
        applied_fraction = guidance_alpha * trust_fraction
        center = self.repair(self.mean + applied_fraction * delta)
        return GuidedCenter(center, distance, applied_fraction)

    def ask(self, center: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample one CMA population around ``center`` or the numerical mean."""
        sampling_center = self.mean if center is None else np.asarray(center, dtype=float)
        sampling_center = sampling_center.reshape(-1)
        if sampling_center.size != self.dimension:
            raise ValueError("Sampling center has the wrong dimension")

        sqrt_cov, _ = self._eigendecomposition()
        standard_normal = self.rng.standard_normal(
            (self.population_size, self.dimension)
        )
        if sqrt_cov.ndim == 1:
            steps = standard_normal * sqrt_cov
        else:
            steps = standard_normal @ sqrt_cov.T
        return np.vstack(
            [self.repair(sampling_center + self.sigma * step) for step in steps]
        )

    def tell(
        self,
        candidates: np.ndarray,
        rewards: np.ndarray,
        sampling_center: np.ndarray,
    ) -> None:
        """Update CMA from locally sampled and reward-evaluated candidates.

        ``sampling_center`` may be externally guided.  All evolution-path and
        covariance steps are measured from that center, so the unvalidated jump
        from the old numerical mean is not interpreted as a CMA search step.
        """
        candidates = np.asarray(candidates, dtype=float)
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        sampling_center = np.asarray(sampling_center, dtype=float).reshape(-1)
        expected_shape = (self.population_size, self.dimension)
        if candidates.shape != expected_shape:
            raise ValueError(f"Expected candidate shape {expected_shape}, got {candidates.shape}")
        if rewards.size != self.population_size:
            raise ValueError("One reward is required for each candidate")

        order = np.argsort(rewards)[::-1]
        elite = candidates[order[: self.mu]]
        new_mean = np.sum(self.weights[:, None] * elite, axis=0)
        elite_steps = (elite - sampling_center) / self.sigma
        weighted_step = np.sum(self.weights[:, None] * elite_steps, axis=0)

        _, inv_sqrt_cov = self._eigendecomposition()
        whitened_step = self._transform(inv_sqrt_cov, weighted_step)
        self.p_sigma = (
            (1.0 - self.c_sigma) * self.p_sigma
            + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff)
            * whitened_step
        )

        path_norm = np.linalg.norm(self.p_sigma)
        normalization = np.sqrt(
            max(1e-20, 1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
        )
        h_sigma = float(
            path_norm / normalization
            < (1.4 + 2.0 / (self.dimension + 1.0)) * self.chi_n
        )
        self.p_c = (
            (1.0 - self.c_c) * self.p_c
            + h_sigma
            * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
            * weighted_step
        )

        old_factor = (
            1.0
            - self.c_1
            - self.c_mu
            + self.c_1 * (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        )
        if self.covariance_mode == "full":
            rank_one = np.outer(self.p_c, self.p_c)
            rank_mu = sum(
                weight * np.outer(step, step)
                for weight, step in zip(self.weights, elite_steps)
            )
            self.covariance = (
                old_factor * self.covariance
                + self.c_1 * rank_one
                + self.c_mu * rank_mu
            )
            self.covariance = 0.5 * (self.covariance + self.covariance.T)
        else:
            rank_one = self.p_c**2
            rank_mu = np.sum(self.weights[:, None] * elite_steps**2, axis=0)
            self.covariance = (
                old_factor * self.covariance
                + self.c_1 * rank_one
                + self.c_mu * rank_mu
            )
            self.covariance = np.maximum(self.covariance, 1e-20)

        self.sigma *= float(
            np.exp((self.c_sigma / self.damps) * (path_norm / self.chi_n - 1.0))
        )
        self.sigma = float(np.clip(self.sigma, 1e-6, self.upper_bound - self.lower_bound))
        self.mean = self.repair(new_mean)
        self.generation += 1

    def restart(
        self,
        mean: np.ndarray,
        sigma: Optional[float] = None,
        keep_diagonal_scale: bool = False,
    ) -> None:
        """Restart CMA around a promising remote LLM proposal."""
        self.mean = self.repair(mean)
        self.sigma = float(sigma if sigma is not None else self.initial_sigma)
        self.p_c.fill(0.0)
        self.p_sigma.fill(0.0)
        self.generation = 0
        if self.covariance_mode == "full":
            if keep_diagonal_scale:
                diagonal = np.maximum(np.diag(self.covariance), 1e-20)
                self.covariance = np.diag(diagonal)
            else:
                self.covariance = np.eye(self.dimension, dtype=float)
        elif not keep_diagonal_scale:
            self.covariance = np.ones(self.dimension, dtype=float)

    def relocate(self, mean: np.ndarray, reset_paths: bool = True) -> None:
        """Accept a reward-validated external mean without relearning covariance."""
        self.mean = self.repair(mean)
        if reset_paths:
            self.p_c.fill(0.0)
            self.p_sigma.fill(0.0)

    def covariance_summary(self) -> dict:
        """Return lightweight diagnostics without exposing a large matrix."""
        diagonal = (
            np.diag(self.covariance)
            if self.covariance_mode == "full"
            else self.covariance
        )
        return {
            "mode": self.covariance_mode,
            "sigma": self.sigma,
            "diag_min": float(np.min(diagonal)),
            "diag_max": float(np.max(diagonal)),
            "generation": self.generation,
        }
