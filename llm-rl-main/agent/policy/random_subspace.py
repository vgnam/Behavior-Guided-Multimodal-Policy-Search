"""Low-dimensional random-subspace search for high-dimensional policies.

Let theta in R^D be the full policy vector and z in R^d be a latent update,
where d << D.  A Gaussian matrix is orthonormalized to obtain A in R^(Dxd):

    G_ij ~ Normal(0, 1),       A = qr(G)

The full policy update around an anchor theta_a is

    theta(z; theta_a) = clip(theta_a + scale * A @ z, lower, upper).

Because A has orthonormal columns, a historical full-space displacement can
be represented in the current subspace by the least-squares coordinates

    z_hat = A.T @ (theta_history - theta_reference) / scale.
"""

import numpy as np


class RandomSubspace:
    """Map compact latent updates to a full policy parameter vector."""

    def __init__(
        self,
        full_dim: int,
        latent_dim: int,
        seed: int = 0,
        scale: float = 1.0,
        lower_bound: float = -6.0,
        upper_bound: float = 6.0,
    ):
        if full_dim <= 0:
            raise ValueError("full_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")

        self.full_dim = full_dim
        self.latent_dim = min(latent_dim, full_dim)
        self.seed = seed
        self.scale = scale
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.refresh_count = 0
        self.matrix = self._make_matrix(seed)

    def _make_matrix(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        gaussian = rng.normal(size=(self.full_dim, self.latent_dim))
        orthonormal, _ = np.linalg.qr(gaussian, mode="reduced")
        return orthonormal

    def refresh(self) -> None:
        """Generate a deterministic new subspace from the next seed."""
        self.refresh_count += 1
        self.matrix = self._make_matrix(self.seed + self.refresh_count)

    def apply(self, anchor: np.ndarray, latent_update: np.ndarray) -> np.ndarray:
        """Decode z into theta = clip(anchor + scale * A @ z)."""
        anchor = np.asarray(anchor, dtype=float).reshape(-1)
        latent_update = np.asarray(latent_update, dtype=float).reshape(-1)
        if anchor.size != self.full_dim:
            raise ValueError(
                f"Expected anchor dimension {self.full_dim}, got {anchor.size}"
            )
        if latent_update.size != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, got {latent_update.size}"
            )
        decoded = anchor + self.scale * (self.matrix @ latent_update)
        return np.clip(decoded, self.lower_bound, self.upper_bound)

    def project_delta(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Project target-reference into least-squares latent coordinates."""
        reference = np.asarray(reference, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if reference.size != self.full_dim or target.size != self.full_dim:
            raise ValueError(f"Expected full dimension {self.full_dim}")
        return self.matrix.T @ (target - reference) / self.scale
