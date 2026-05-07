import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont


class Grid2OpCase14Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1,
    }

    _SUPPORTED_FEATURES = {
        "rho",
        "line_status",
        "timestep_overflow",
        "time_before_cooldown_line",
        "topo_vect",
        "gen_p",
        "load_p",
        "actual_dispatch",
    }

    def __init__(
        self,
        render_mode=None,
        grid2op_env_name="l2rpn_case14_sandbox",
        reward_class_name="L2RPNReward",
        observation_features=None,
        selected_line_ids=None,
        grid2op_make_kwargs=None,
    ):
        super().__init__()
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}; expected one of None, 'human', 'rgb_array'."
            )

        self.render_mode = render_mode
        self.grid2op_env_name = grid2op_env_name
        self.reward_class_name = reward_class_name
        self.observation_features = tuple(observation_features or ["rho", "line_status"])
        self.selected_line_ids = selected_line_ids
        self.grid2op_make_kwargs = dict(grid2op_make_kwargs or {})

        invalid_features = sorted(
            set(self.observation_features) - self._SUPPORTED_FEATURES
        )
        if invalid_features:
            raise ValueError(
                f"Unsupported observation features: {invalid_features}. "
                f"Supported values are {sorted(self._SUPPORTED_FEATURES)}."
            )

        self._grid_env = None
        self._last_grid_obs = None
        self._last_reward = 0.0
        self._step_count = 0
        self._load_grid_env()
        self._build_action_space()

        initial_obs, _ = self.reset()
        obs_dim = int(initial_obs.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _load_grid_env(self):
        try:
            import grid2op
            from grid2op.Reward import L2RPNReward, RedispReward
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Grid2Op benchmark requested, but the 'grid2op' package is not installed. "
                "Install the project requirements first."
            ) from exc

        reward_classes = {
            "L2RPNReward": L2RPNReward,
            "RedispReward": RedispReward,
        }
        if self.reward_class_name not in reward_classes:
            raise ValueError(
                f"Unsupported reward_class_name={self.reward_class_name!r}. "
                f"Supported values are {sorted(reward_classes)}."
            )

        make_kwargs = dict(self.grid2op_make_kwargs)
        make_kwargs.setdefault("reward_class", reward_classes[self.reward_class_name])

        try:
            self._grid_env = grid2op.make(self.grid2op_env_name, **make_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Grid2Op environment '{self.grid2op_env_name}'. "
                "If this is the first time you use the dataset, Grid2Op may need to download it."
            ) from exc

        if self.selected_line_ids is None:
            self.selected_line_ids = tuple(range(int(self._grid_env.n_line)))
        else:
            self.selected_line_ids = tuple(int(line_id) for line_id in self.selected_line_ids)

        for line_id in self.selected_line_ids:
            if line_id < 0 or line_id >= int(self._grid_env.n_line):
                raise ValueError(
                    f"Invalid line id {line_id}; expected values between 0 and {int(self._grid_env.n_line) - 1}."
                )

    def _build_action_space(self):
        self._line_ids = tuple(self.selected_line_ids)
        self._actions = [self._grid_env.action_space()]
        self._action_labels = ["do nothing"]

        for line_id in self._line_ids:
            self._actions.append(
                self._grid_env.action_space({"set_line_status": [(line_id, -1)]})
            )
            self._action_labels.append(f"disconnect line {line_id}")

        for line_id in self._line_ids:
            self._actions.append(
                self._grid_env.action_space({"set_line_status": [(line_id, +1)]})
            )
            self._action_labels.append(f"reconnect line {line_id}")

        self.action_space = spaces.Discrete(len(self._actions))

    def _raw_reset(self, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options

        try:
            result = self._grid_env.reset(**kwargs)
        except TypeError:
            try:
                if seed is not None:
                    result = self._grid_env.reset(seed=seed)
                else:
                    result = self._grid_env.reset()
            except TypeError:
                result = self._grid_env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def _extract_feature(self, obs, feature_name):
        if feature_name == "rho":
            return np.asarray(obs.rho, dtype=np.float32)[list(self._line_ids)]
        if feature_name == "line_status":
            return np.asarray(obs.line_status, dtype=np.float32)[list(self._line_ids)]
        if feature_name == "timestep_overflow":
            return np.asarray(obs.timestep_overflow, dtype=np.float32)[list(self._line_ids)]
        if feature_name == "time_before_cooldown_line":
            return np.asarray(obs.time_before_cooldown_line, dtype=np.float32)[list(self._line_ids)]
        if feature_name == "topo_vect":
            return np.asarray(obs.topo_vect, dtype=np.float32).reshape(-1)
        if feature_name == "gen_p":
            return np.asarray(obs.gen_p, dtype=np.float32).reshape(-1)
        if feature_name == "load_p":
            return np.asarray(obs.load_p, dtype=np.float32).reshape(-1)
        if feature_name == "actual_dispatch":
            return np.asarray(obs.actual_dispatch, dtype=np.float32).reshape(-1)
        raise ValueError(f"Unsupported feature {feature_name!r}")

    def _obs_to_vector(self, obs):
        parts = [self._extract_feature(obs, feature) for feature in self.observation_features]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self._raw_reset(seed=seed, options=options)
        self._last_grid_obs = obs
        self._last_reward = 0.0
        self._step_count = 0
        return self._obs_to_vector(obs), info

    def step(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self._actions):
            raise ValueError(
                f"Invalid action index {action_idx}; expected 0 <= action < {len(self._actions)}."
            )

        result = self._grid_env.step(self._actions[action_idx])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False

        self._last_grid_obs = obs
        self._last_reward = float(reward)
        self._step_count += 1

        info = dict(info)
        info["benchmark_action"] = self._action_labels[action_idx]
        return self._obs_to_vector(obs), float(reward), terminated, truncated, info

    def _render_dashboard(self):
        if self._last_grid_obs is None:
            return None

        width, height = 1200, 720
        bg = (247, 247, 243)
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        rho = np.asarray(self._last_grid_obs.rho, dtype=np.float32)[list(self._line_ids)]
        line_status = np.asarray(self._last_grid_obs.line_status, dtype=np.float32)[list(self._line_ids)]

        overloaded = int(np.sum(rho > 1.0))
        disconnected = int(np.sum(line_status < 0.5))
        max_rho = float(np.max(rho)) if rho.size else 0.0

        draw.text((24, 18), f"Grid2Op benchmark: {self.grid2op_env_name}", fill=(20, 20, 20), font=font)
        draw.text(
            (24, 42),
            "Observation dashboard for BMPS. Each tile is one transmission line.",
            fill=(20, 20, 20),
            font=font,
        )
        draw.text(
            (24, 66),
            "Bar height and color show rho. Gray tiles are disconnected lines. Red bars indicate rho > 1.",
            fill=(20, 20, 20),
            font=font,
        )
        draw.text(
            (24, 98),
            f"step={self._step_count}  reward={self._last_reward:.3f}  max_rho={max_rho:.2f}  "
            f"overloaded={overloaded}/{len(self._line_ids)}  disconnected={disconnected}/{len(self._line_ids)}",
            fill=(20, 20, 20),
            font=font,
        )

        cols = 10
        tile_w = 100
        tile_h = 220
        x0 = 24
        y0 = 150
        bar_bottom = 185

        for idx, line_id in enumerate(self._line_ids):
            row = idx // cols
            col = idx % cols
            left = x0 + col * (tile_w + 12)
            top = y0 + row * (tile_h + 18)
            right = left + tile_w
            bottom = top + tile_h

            is_connected = bool(line_status[idx] >= 0.5)
            tile_fill = (255, 255, 255) if is_connected else (225, 225, 225)
            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=8,
                fill=tile_fill,
                outline=(70, 70, 70),
                width=2,
            )

            rho_value = float(max(0.0, rho[idx]))
            clamped_rho = min(rho_value, 2.0)
            bar_height = int((clamped_rho / 2.0) * 120.0)
            bar_left = left + 34
            bar_right = right - 34
            bar_top = top + bar_bottom - bar_height
            bar_bottom_y = top + bar_bottom

            if rho_value >= 1.0:
                bar_color = (210, 44, 44)
            elif rho_value >= 0.8:
                bar_color = (232, 145, 31)
            else:
                bar_color = (54, 140, 84)

            draw.rectangle(
                (bar_left, top + 64, bar_right, bar_bottom_y),
                outline=(160, 160, 160),
                width=1,
            )
            draw.rectangle(
                (bar_left, bar_top, bar_right, bar_bottom_y),
                fill=bar_color,
                outline=bar_color,
            )

            draw.text((left + 10, top + 12), f"line {line_id}", fill=(20, 20, 20), font=font)
            draw.text((left + 10, top + 32), f"rho={rho_value:.2f}", fill=(20, 20, 20), font=font)
            draw.text(
                (left + 10, top + 196),
                "connected" if is_connected else "disconnected",
                fill=(20, 20, 20),
                font=font,
            )

        ranking_left = 1080
        draw.text((ranking_left, 150), "Highest rho lines", fill=(20, 20, 20), font=font)
        sorted_pairs = sorted(
            [(line_id, float(rho_val), bool(line_status[idx] >= 0.5)) for idx, (line_id, rho_val) in enumerate(zip(self._line_ids, rho))],
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        for rank, (line_id, rho_val, is_connected) in enumerate(sorted_pairs, start=1):
            status = "up" if is_connected else "down"
            draw.text(
                (ranking_left, 150 + 24 * rank),
                f"{rank}. line {line_id}: rho={rho_val:.2f} ({status})",
                fill=(20, 20, 20),
                font=font,
            )

        return np.asarray(image, dtype=np.uint8)

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return None

        frame = self._render_dashboard()
        if self.render_mode == "human":
            return frame
        return frame

    def close(self):
        if self._grid_env is not None:
            close = getattr(self._grid_env, "close", None)
            if callable(close):
                close()
            self._grid_env = None


if "Grid2OpCase14-v0" not in gym.registry:
    gym.register(
        id="Grid2OpCase14-v0",
        entry_point=Grid2OpCase14Env,
    )
