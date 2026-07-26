import unittest

import numpy as np
from jinja2 import Environment, FileSystemLoader

from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent


class MLPVisionAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jinja = Environment(loader=FileSystemLoader("agent/policy/templates"))
        cls.template = jinja.get_template("num_optim_vision.j2")

    def make_discrete_agent(self):
        return LLMNumOptimVisionAgent(
            logdir="logs/test",
            dim_action=4,
            dim_state=9,
            max_traj_count=2,
            max_traj_length=10,
            llm_si_template=self.template,
            llm_output_conversion_template=self.template,
            llm_model_name="test/model",
            num_evaluation_episodes=1,
            bias=True,
            optimum=1.0,
            search_step_size=0.1,
            enable_vision=False,
            policy_type="mlp",
            hidden_sizes=[32, 32],
            optimization_mode="latent",
            latent_dim=32,
            projection_seed=42,
            projection_scale=0.25,
            state_encoding="one_hot",
        )

    def test_discrete_state_is_one_hot_encoded(self):
        agent = self.make_discrete_agent()

        encoded = agent._prepare_policy_state(5)

        self.assertEqual(encoded.shape, (1, 9))
        self.assertEqual(encoded[0, 5], 1.0)
        self.assertEqual(np.sum(encoded), 1.0)

    def test_discrete_mlp_dimensions_and_action_logits(self):
        agent = self.make_discrete_agent()
        encoded = agent._prepare_policy_state(5)

        action_logits = agent.policy.get_action(encoded.T)

        self.assertEqual(agent.parameter_dim, 1508)
        self.assertEqual(agent.rank, 32)
        self.assertEqual(action_logits.shape, (1, 4))
        self.assertIn(int(np.argmax(action_logits)), range(4))

    def test_out_of_range_discrete_state_is_rejected(self):
        agent = self.make_discrete_agent()

        with self.assertRaises(ValueError):
            agent._prepare_policy_state(9)


if __name__ == "__main__":
    unittest.main()
