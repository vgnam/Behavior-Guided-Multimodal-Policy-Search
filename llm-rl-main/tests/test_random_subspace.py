import unittest

import numpy as np

from agent.policy.random_subspace import RandomSubspace


class RandomSubspaceTest(unittest.TestCase):
    def test_matrix_has_orthonormal_columns(self):
        mapper = RandomSubspace(full_dim=18, latent_dim=8, seed=42)

        np.testing.assert_allclose(
            mapper.matrix.T @ mapper.matrix,
            np.eye(8),
            atol=1e-12,
        )

    def test_decode_and_project_round_trip(self):
        mapper = RandomSubspace(full_dim=18, latent_dim=8, seed=42, scale=0.5)
        anchor = np.zeros(18)
        latent_update = np.array([0.1, -0.2, 0.3, 0.0, 0.4, -0.1, 0.2, -0.3])

        decoded = mapper.apply(anchor, latent_update)
        projected = mapper.project_delta(anchor, decoded)

        np.testing.assert_allclose(projected, latent_update, atol=1e-12)

    def test_latent_dimension_is_capped_at_full_dimension(self):
        mapper = RandomSubspace(full_dim=4, latent_dim=10, seed=42)

        self.assertEqual(mapper.latent_dim, 4)
        self.assertEqual(mapper.matrix.shape, (4, 4))

    def test_refresh_changes_matrix_deterministically(self):
        first = RandomSubspace(full_dim=18, latent_dim=8, seed=42)
        second = RandomSubspace(full_dim=18, latent_dim=8, seed=42)

        first.refresh()
        second.refresh()

        np.testing.assert_allclose(first.matrix, second.matrix)


if __name__ == "__main__":
    unittest.main()
