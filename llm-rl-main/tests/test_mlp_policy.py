import unittest

import numpy as np

from agent.policy.mlp_policy import MLPPolicy


class MLPPolicyTest(unittest.TestCase):
    def test_swimmer_32_by_32_parameter_count(self):
        policy = MLPPolicy(8, 2, hidden_sizes=[32, 32], bias=True)

        self.assertEqual(policy.layer_sizes, (8, 32, 32, 2))
        self.assertEqual(policy.parameter_count, 1410)
        self.assertEqual(policy.get_parameters().shape, (1410,))

    def test_flat_parameter_round_trip(self):
        policy = MLPPolicy(8, 2, hidden_sizes=[16, 16], bias=True)
        parameters = np.linspace(-0.5, 0.5, policy.parameter_count)

        policy.update_policy(parameters)

        np.testing.assert_allclose(policy.get_parameters(), parameters)

    def test_tanh_output_has_action_shape_and_bounds(self):
        policy = MLPPolicy(
            8,
            2,
            hidden_sizes=[32, 32],
            hidden_activation="tanh",
            output_activation="tanh",
        )

        action = policy.get_action(np.zeros((8, 1)))

        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_bias_can_be_disabled(self):
        policy = MLPPolicy(8, 2, hidden_sizes=[32, 32], bias=False)

        self.assertEqual(policy.parameter_count, 1344)
        self.assertEqual(policy.biases, [])


if __name__ == "__main__":
    unittest.main()
