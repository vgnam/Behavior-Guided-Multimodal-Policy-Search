"""Gymnasium adapter for the local air-hockey-rl Box2D benchmark."""

from copy import deepcopy
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_ROOT = _PROJECT_ROOT / "air-hockey-rl"
_DEFAULT_CONFIG_PATH = Path("configs/new_juggle/sysid_best_params_hist2.yaml")


def _deep_update(target, updates):
    """Recursively apply configuration overrides without mutating the caller."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


class AirHockeyBMPSWrapper(gym.Env):
    """Expose air-hockey-rl through the rendering contract BMPS expects."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        render_mode=None,
        config_path=None,
        source_root=None,
        config_overrides=None,
        seed=None,
        renderer_orientation="vertical",
        show_target_position=True,
    ):
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}; expected None, "
                "'human', or 'rgb_array'."
            )

        self.render_mode = render_mode
        self.source_root = self._resolve_source_root(source_root)
        self.config_path = self._resolve_config_path(config_path)
        self.renderer_orientation = str(renderer_orientation)
        self.show_target_position = bool(show_target_position)
        self._renderer = None
        self._env = None

        source_root_string = str(self.source_root)
        if source_root_string not in sys.path:
            sys.path.insert(0, source_root_string)

        try:
            from airhockey import AirHockeyEnv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Air-hockey benchmark requested, but its dependencies are not "
                "available. Install air-hockey-rl and its Box2D dependencies."
            ) from exc

        with self.config_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        if not isinstance(loaded, dict) or not isinstance(
            loaded.get("air_hockey"), dict
        ):
            raise ValueError(
                f"{self.config_path} must contain an 'air_hockey' mapping."
            )

        benchmark_config = deepcopy(loaded["air_hockey"])
        if config_overrides:
            _deep_update(benchmark_config, config_overrides)
        if seed is not None:
            benchmark_config["seed"] = int(seed)

        if benchmark_config.get("simulator") != "box2d":
            raise ValueError(
                "The BMPS adapter currently supports the air-hockey Box2D "
                "simulator only. Real-robot execution requires the benchmark's "
                "hardware safety and reset pipeline."
            )

        self.benchmark_config = benchmark_config
        self._env = AirHockeyEnv(benchmark_config)

        observation_shape = tuple(self._env.observation_space.shape)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(self._env.action_space.low, dtype=np.float32),
            high=np.asarray(self._env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_range = getattr(self._env, "reward_range", (-np.inf, np.inf))

    @staticmethod
    def _resolve_source_root(source_root):
        path = Path(source_root) if source_root is not None else _DEFAULT_SOURCE_ROOT
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        path = path.resolve()
        if not (path / "airhockey").is_dir():
            raise FileNotFoundError(
                f"Could not find the air-hockey package under {path}. "
                "Set env_kwargs.source_root to the air-hockey-rl checkout."
            )
        return path

    def _resolve_config_path(self, config_path):
        path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [_PROJECT_ROOT / path, self.source_root / path]

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find the air-hockey environment config. Searched: {searched}"
        )

    def _get_renderer(self):
        if self._renderer is None:
            try:
                from airhockey.renderers import AirHockeyRenderer
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Air-hockey rendering requires OpenCV. Install the "
                    "air-hockey-rl training dependencies."
                ) from exc
            self._renderer = AirHockeyRenderer(
                self._env,
                orientation=self.renderer_orientation,
                show_target_position=self.show_target_position,
                show_acceleration_arrow=False,
            )
        return self._renderer

    @staticmethod
    def _observation(values):
        return np.asarray(values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del options
        observation, info = self._env.reset(seed=seed)
        if self._renderer is not None:
            self._renderer.prev_paddle_velocities.clear()
            self._renderer.set_action(None)
        return self._observation(observation), dict(info)

    def step(self, action):
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.action_space.shape),
            self.action_space.low,
            self.action_space.high,
        )
        if self._renderer is not None:
            self._renderer.set_action(clipped_action)

        observation, reward, terminated, truncated, info = self._env.step(
            clipped_action
        )
        # The benchmark currently reports a time-limit transition as both
        # terminated and truncated. Normalize it to Gymnasium's convention.
        truncated = bool(truncated)
        terminated = bool(terminated) and not truncated
        return (
            self._observation(observation),
            float(reward),
            terminated,
            truncated,
            dict(info),
        )

    def render(self):
        if self.render_mode is None:
            return None
        renderer = self._get_renderer()
        if self.render_mode == "human":
            renderer.render()
            return None

        # AirHockeyRenderer produces OpenCV BGR; VLMAnalyzer expects RGB.
        bgr_frame = renderer.get_frame()
        return np.ascontiguousarray(bgr_frame[:, :, ::-1])

    def close(self):
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None
        if self.render_mode == "human" and self._renderer is not None:
            try:
                import cv2

                cv2.destroyWindow("Air Hockey 2D")
            except Exception:
                pass
        self._renderer = None


if "AirHockeyJuggle-v0" not in gym.registry:
    gym.register(
        id="AirHockeyJuggle-v0",
        entry_point="envs.air_hockey_env:AirHockeyBMPSWrapper",
        disable_env_checker=False,
    )
