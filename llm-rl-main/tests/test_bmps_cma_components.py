import io
import unittest

import numpy as np

from agent.policy.pairwise_preference import (
    BradleyTerryAggregator,
    PairwisePreference,
    parse_pairwise_response,
)
from agent.policy.temporal_frame_stacker import TemporalFrameStacker
from agent.policy.trust_region_cma import TrustRegionCMA
from agent.llm_num_optim_linear_policy_bmps_cma import LLMNumOptimBMPSCMAAgent


class TrustRegionCMATest(unittest.TestCase):
    def test_remote_proposal_is_projected_in_whitened_space(self):
        cma = TrustRegionCMA(
            mean=np.zeros(3),
            sigma=0.5,
            population_size=6,
            covariance_mode="full",
            seed=4,
        )
        projection = cma.project_proposal(
            np.array([5.0, 5.0, 5.0]),
            trust_region_radius=2.0,
            guidance_alpha=1.0,
        )

        whitened_distance = np.linalg.norm(projection.center - cma.mean) / cma.sigma
        self.assertLessEqual(whitened_distance, 2.0 + 1e-10)
        self.assertGreater(projection.mahalanobis_distance, 2.0)
        self.assertLess(projection.applied_fraction, 1.0)

    def test_reward_ranked_update_moves_mean_and_keeps_covariance_positive(self):
        cma = TrustRegionCMA(
            mean=np.zeros(2),
            sigma=0.4,
            population_size=6,
            covariance_mode="full",
            seed=9,
        )
        center = np.array([0.2, -0.1])
        candidates = cma.ask(center)
        rewards = candidates[:, 0] - 0.1 * np.abs(candidates[:, 1])
        previous_mean = cma.mean.copy()
        cma.tell(candidates, rewards, center)

        self.assertFalse(np.allclose(previous_mean, cma.mean))
        self.assertTrue(np.all(np.linalg.eigvalsh(cma.covariance) > 0.0))
        self.assertGreater(cma.sigma, 0.0)

    def test_large_policy_uses_diagonal_covariance_in_auto_mode(self):
        cma = TrustRegionCMA(
            mean=np.zeros(20),
            population_size=8,
            covariance_mode="auto",
            full_covariance_max_dim=10,
        )
        self.assertEqual(cma.covariance_mode, "diagonal")
        self.assertEqual(cma.covariance.shape, (20,))


class TemporalFrameStackerTest(unittest.TestCase):
    def test_every_frame_contributes_to_one_output_image(self):
        stacker = TemporalFrameStacker(
            motion_threshold=5.0,
            background_learning_rate=0.0,
            tint_strength=0.5,
            occupancy_scale=1.0,
            max_side=None,
        )
        frames = []
        for x_position in (1, 4, 7, 10):
            frame = np.zeros((12, 14, 3), dtype=np.uint8)
            frame[4:8, x_position : x_position + 2] = np.array([220, 220, 220])
            frames.append(frame)
            stacker.add(frame)

        composite = stacker.finalize()
        self.assertEqual(stacker.frame_count, len(frames))
        self.assertEqual(composite.shape, frames[0].shape)
        self.assertEqual(composite.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(composite), 0)
        self.assertFalse(np.array_equal(composite, frames[-1]))

    def test_bmps_cma_samples_periodically_before_stacking(self):
        class FakePolicy:
            def get_parameters(self):
                return np.zeros(2)

            def get_action(self, state):
                return np.zeros((1, 1))

        class FakeEnvironment:
            def __init__(self, world):
                self.world = world

            def render(self):
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                frame[:, self.world.steps % 8] = 255
                return frame

        class FakeWorld:
            discretize = False

            def __init__(self):
                self.steps = 0
                self.accumulated_reward = 0.0
                self.env = FakeEnvironment(self)

            def reset(self):
                self.steps = 0
                self.accumulated_reward = 0.0
                return np.zeros(1)

            def step(self, action):
                self.steps += 1
                self.accumulated_reward += 1.0
                return np.zeros(1), 1.0, self.steps >= 10

            def get_accu_reward(self):
                return self.accumulated_reward

        agent = object.__new__(LLMNumOptimBMPSCMAAgent)
        agent.policy = FakePolicy()
        agent.dim_action = 1
        agent.max_traj_length = 10
        agent.stack_frame_period = 3
        agent.stack_kwargs = {
            "motion_threshold": 5.0,
            "background_learning_rate": 0.0,
            "tint_strength": 0.5,
            "occupancy_scale": 1.0,
        }
        agent.total_steps = 0
        agent.total_episodes = 0

        reward, image, frame_count, terminated_early = agent._rollout_with_superposition(
            FakeWorld(), io.StringIO(), capture_visual=True
        )

        # Initial frame + steps 3, 6, 9 + terminal step 10.
        self.assertEqual(frame_count, 5)
        self.assertEqual(reward, 10.0)
        self.assertFalse(terminated_early)
        self.assertEqual(image.shape, (8, 8, 3))


class PairwisePreferenceTest(unittest.TestCase):
    def test_parser_and_internal_bt_ranking(self):
        parsed = parse_pairwise_response(
            "Preferred: B\nConfidence: high\nEvidence: B stays upright longer.",
            "current",
            "candidate",
        )
        self.assertEqual(parsed.winner, "candidate")
        self.assertEqual(parsed.confidence, "high")

        comparisons = [
            parsed,
            PairwisePreference(
                left_id="candidate",
                right_id="worst",
                winner="candidate",
                confidence="high",
                evidence="candidate avoids failure",
            ),
            PairwisePreference(
                left_id="current",
                right_id="worst",
                winner="current",
                confidence="medium",
                evidence="current lasts longer",
            ),
        ]
        ranking = BradleyTerryAggregator().rank(
            ["current", "candidate", "worst"], comparisons
        )
        self.assertEqual(ranking[0], "candidate")
        self.assertEqual(ranking[-1], "worst")


if __name__ == "__main__":
    unittest.main()
