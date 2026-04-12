import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RoboSuiteLiftDiscreteEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        render_mode=None,
        env_name="Lift",
        robot="Panda",
        camera_name="agentview",
        image_height=256,
        image_width=256,
        horizon=200,
        control_freq=20,
        reward_shaping=True,
        state_keys=None,
        state_dim=64,
        translation_delta=1.0,
        gripper_open_value=-1.0,
        gripper_close_value=1.0,
        env_kwargs=None,
    ):
        super().__init__()
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}; expected one of None, 'human', 'rgb_array'."
            )

        self.render_mode = render_mode
        self.env_name = env_name
        self.robot = robot
        self.camera_name = camera_name
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.horizon = int(horizon)
        self.control_freq = int(control_freq)
        self.reward_shaping = bool(reward_shaping)
        self.state_keys = tuple(state_keys or ["robot0_proprio-state", "object-state"])
        self.state_dim = int(state_dim)
        self.translation_delta = float(translation_delta)
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_close_value = float(gripper_close_value)
        self.extra_env_kwargs = dict(env_kwargs or {})

        self._env = None
        self._last_obs = None
        self._last_image = None
        self._last_reward = 0.0
        self._step_count = 0
        self._action_low = None
        self._action_high = None
        self._action_dim = None

        self._build_env()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

    def _load_controller_config(self):
        try:
            from robosuite.controllers import load_composite_controller_config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "robosuite benchmark requested, but the 'robosuite' package is not installed."
            ) from exc

        try:
            return load_composite_controller_config(controller="BASIC", robot=self.robot)
        except TypeError:
            return load_composite_controller_config(controller="BASIC")

    def _build_env(self):
        try:
            import robosuite as suite
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "robosuite benchmark requested, but the 'robosuite' package is not installed."
            ) from exc

        controller_config = self._load_controller_config()
        use_camera_obs = self.render_mode == "rgb_array"

        kwargs = dict(self.extra_env_kwargs)
        kwargs.setdefault("env_name", self.env_name)
        kwargs.setdefault("robots", self.robot)
        kwargs.setdefault("controller_configs", controller_config)
        kwargs.setdefault("has_renderer", self.render_mode == "human")
        kwargs.setdefault("has_offscreen_renderer", use_camera_obs)
        kwargs.setdefault("use_camera_obs", use_camera_obs)
        kwargs.setdefault("use_object_obs", True)
        kwargs.setdefault("reward_shaping", self.reward_shaping)
        kwargs.setdefault("control_freq", self.control_freq)
        kwargs.setdefault("horizon", self.horizon)
        kwargs.setdefault("camera_names", self.camera_name)
        kwargs.setdefault("camera_heights", self.image_height)
        kwargs.setdefault("camera_widths", self.image_width)

        self._env = suite.make(**kwargs)

        low, high = self._env.action_spec
        self._action_low = np.asarray(low, dtype=np.float32).reshape(-1)
        self._action_high = np.asarray(high, dtype=np.float32).reshape(-1)
        self._action_dim = int(self._action_low.shape[0])

    def _extract_state(self, obs_dict):
        parts = []
        for key in self.state_keys:
            if key not in obs_dict:
                raise KeyError(
                    f"Observation key {key!r} not found in robosuite observations. "
                    f"Available keys: {sorted(obs_dict.keys())}"
                )
            parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))

        flat = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)

        if flat.shape[0] >= self.state_dim:
            return flat[: self.state_dim]

        padded = np.zeros(self.state_dim, dtype=np.float32)
        padded[: flat.shape[0]] = flat
        return padded

    def _extract_image(self, obs_dict):
        image_key = f"{self.camera_name}_image"
        image = obs_dict.get(image_key)
        if image is None:
            return None
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3:
            return None
        # robosuite documents images in MuJoCo's default flipped convention.
        return np.flipud(image)

    def _discrete_to_continuous_action(self, action_idx):
        action = np.zeros(self._action_dim, dtype=np.float32)

        if action_idx == 1:
            action[0] = self.translation_delta
        elif action_idx == 2:
            action[0] = -self.translation_delta
        elif action_idx == 3 and self._action_dim >= 2:
            action[1] = self.translation_delta
        elif action_idx == 4 and self._action_dim >= 2:
            action[1] = -self.translation_delta
        elif action_idx == 5 and self._action_dim >= 3:
            action[2] = self.translation_delta
        elif action_idx == 6 and self._action_dim >= 3:
            action[2] = -self.translation_delta
        elif action_idx == 7:
            action[-1] = self.gripper_close_value
        elif action_idx == 8:
            action[-1] = self.gripper_open_value

        return np.clip(action, self._action_low, self._action_high)

    def _action_label(self, action_idx):
        labels = {
            0: "do nothing",
            1: "move +x",
            2: "move -x",
            3: "move +y",
            4: "move -y",
            5: "move +z",
            6: "move -z",
            7: "close gripper",
            8: "open gripper",
        }
        return labels[int(action_idx)]

    def reset(self, seed=None, options=None):
        del seed, options
        obs = self._env.reset()
        self._last_obs = obs
        self._last_image = self._extract_image(obs)
        self._last_reward = 0.0
        self._step_count = 0
        return self._extract_state(obs), {}

    def step(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= self.action_space.n:
            raise ValueError(
                f"Invalid action index {action_idx}; expected 0 <= action < {self.action_space.n}."
            )

        raw_action = self._discrete_to_continuous_action(action_idx)
        obs, reward, done, info = self._env.step(raw_action)
        self._last_obs = obs
        self._last_image = self._extract_image(obs)
        self._last_reward = float(reward)
        self._step_count += 1

        info = dict(info)
        info["benchmark_action"] = self._action_label(action_idx)
        return self._extract_state(obs), float(reward), bool(done), False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._last_image
        if self.render_mode == "human":
            if hasattr(self._env, "render"):
                return self._env.render()
            return self._last_image
        return None

    def close(self):
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
            self._env = None


if "RoboSuiteLiftDiscrete-v0" not in gym.registry:
    gym.register(
        id="RoboSuiteLiftDiscrete-v0",
        entry_point=RoboSuiteLiftDiscreteEnv,
    )
