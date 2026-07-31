import importlib.util
import unittest

import gymnasium as gym
import numpy as np

from envs import air_hockey_env  # noqa: F401


class AirHockeyEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name
            for name in ("Box2D", "cv2")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise unittest.SkipTest(
                f"air-hockey integration requires: {', '.join(missing)}"
            )

    def test_registered_environment_reset_step_and_render(self):
        env = gym.make(
            "AirHockeyJuggle-v0",
            render_mode="rgb_array",
            config_path=(
                "air-hockey-rl/configs/new_juggle/"
                "sysid_best_params_hist2.yaml"
            ),
            seed=0,
        )
        try:
            observation, info = env.reset(seed=7)
            self.assertEqual(observation.shape, (30,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertIsInstance(info, dict)

            next_observation, reward, terminated, truncated, step_info = env.step(
                np.array([4.0, -4.0], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (30,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
            self.assertTrue(
                np.all(np.abs(env.unwrapped._env.last_action) <= 1.0)
            )

            frame = env.render()
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
