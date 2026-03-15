"""
Atari world wrapper for the ProPS framework.

Preprocessing pipeline (pure numpy — no external vision libraries required):
  RGB (H, W, 3)
    → grayscale      (H, W)
    → nearest-neighbour resize to (resize_h, resize_w)
    → flatten        (resize_h * resize_w,)
    → normalize      / 255.0  → [0.0, 1.0]

The resulting 1-D observation is the input to the MLP policy.
`obs_dim` = resize_h * resize_w  (accessible after construction).

Usage:
    world = AtariWorld("ALE/Pong-v5", render_mode=None, max_traj_length=1000)
    obs = world.reset()          # shape: (obs_dim,)
    obs, reward, done = world.step(action)
"""

import numpy as np
import gymnasium as gym
from world.base_world import BaseWorld

# Luma coefficients for RGB → grayscale (ITU-R BT.601)
_LUMA = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


class AtariWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name: str,
        render_mode,
        max_traj_length: int = 1000,
        resize_h: int = 32,
        resize_w: int = 32,
    ):
        super().__init__(gym_env_name)
        self.env = gym.make(gym_env_name, render_mode=render_mode)
        self.max_traj_length = max_traj_length
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.steps = 0
        self.accu_reward = 0.0

        # Pre-compute index arrays for nearest-neighbour resize
        raw_h, raw_w = self.env.observation_space.shape[:2]  # (H, W, C)
        self._row_idx = (np.arange(resize_h) * raw_h / resize_h).astype(int)
        self._col_idx = (np.arange(resize_w) * raw_w / resize_w).astype(int)

        # Public: policy knows how many inputs to expect
        self.obs_dim = resize_h * resize_w

    # ── Preprocessing ────────────────────────────────────────────────────

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        gray = obs[..., :3].astype(np.float32) @ _LUMA       # (H, W)
        resized = gray[np.ix_(self._row_idx, self._col_idx)]  # (rh, rw)
        return (resized / 255.0).reshape(-1)                   # (rh*rw,)

    # ── BaseWorld interface ──────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0.0
        return self._preprocess(obs)

    def step(self, action):
        self.steps += 1
        obs, reward, done, truncated, _ = self.env.step(int(action))
        self.accu_reward += reward
        if self.steps >= self.max_traj_length or truncated:
            done = True
        return self._preprocess(obs), reward, done

    def get_accu_reward(self) -> float:
        return self.accu_reward

    def close(self):
        self.env.close()
