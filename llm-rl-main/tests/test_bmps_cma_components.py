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
from agent.policy.vlm_analyzer import VLMAnalyzer
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

    def test_policy_proposal_keeps_precision_and_is_repaired_to_bounds(self):
        cma = TrustRegionCMA(
            mean=np.zeros(2),
            sigma=0.4,
            population_size=4,
            covariance_mode="diagonal",
        )
        repaired = cma.repair(np.array([0.123456789, 8.25]))
        self.assertAlmostEqual(repaired[0], 0.123456789)
        self.assertGreaterEqual(repaired[1], -6.0)
        self.assertLessEqual(repaired[1], 6.0)

    def test_llm_proposal_matches_bmps_one_decimal_box_constraint(self):
        agent = object.__new__(LLMNumOptimBMPSCMAAgent)
        agent.cma_lower_bound = -6.0
        agent.cma_upper_bound = 6.0
        proposal = agent._sanitize_llm_proposal(np.array([1.234, 8.25, -6.06]))
        np.testing.assert_array_equal(proposal, np.array([1.2, 6.0, -6.0]))


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
        agent.stack_trajectory_frames = False
        agent.stack_kwargs = {
            "motion_threshold": 5.0,
            "background_learning_rate": 0.0,
            "tint_strength": 0.5,
            "occupancy_scale": 1.0,
        }
        agent.total_steps = 0
        agent.total_episodes = 0

        reward, images, frame_count, terminated_early = agent._rollout_with_visuals(
            FakeWorld(), io.StringIO(), capture_visual=True
        )

        # Initial frame + steps 3, 6, 9 + terminal step 10.
        self.assertEqual(frame_count, 5)
        self.assertEqual(len(images), 5)
        self.assertEqual(reward, 10.0)
        self.assertFalse(terminated_early)
        self.assertTrue(all(image.shape == (8, 8, 3) for image in images))

        agent.stack_trajectory_frames = True
        _, stacked_images, stacked_source_count, _ = agent._rollout_with_visuals(
            FakeWorld(), io.StringIO(), capture_visual=True
        )
        self.assertEqual(stacked_source_count, 5)
        self.assertEqual(len(stacked_images), 1)
        self.assertEqual(stacked_images[0].shape, (8, 8, 3))


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

    def test_separate_mode_sends_every_sampled_frame_to_vlm(self):
        analyzer = VLMAnalyzer(template_dir="agent/policy/templates", max_retries=1)
        captured = {}

        def fake_call(messages, temperature=0.1):
            captured["messages"] = messages
            return "Preferred: A\nConfidence: high\nEvidence: A is steadier.", 0.0

        analyzer._call_vlm_api = fake_call
        frames_a = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        frames_b = [np.ones((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        response, _ = analyzer.analyze_frame_sequences_pair(
            frames_a,
            frames_b,
            env_description="test environment",
        )

        content = captured["messages"][0]["content"]
        image_count = sum(item["type"] == "image_url" for item in content)
        self.assertEqual(image_count, 5)
        self.assertIn("Preferred: A", response)


if __name__ == "__main__":
    unittest.main()
