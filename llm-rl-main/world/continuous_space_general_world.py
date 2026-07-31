from world.base_world import BaseWorld
import gymnasium as gym


class ContinualSpaceGeneralWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=1000,
        env_kwargs=None,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None]
        self.gym_env_name = gym_env_name
        self.render_mode = render_mode
        self.env_kwargs = dict(env_kwargs or {})
        self._new_reward_enabled = False
        self.env = self._make_env()
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.discretize = True
        else:
            self.discretize = False

    def _make_env(self, new_reward=False):
        env_kwargs = dict(self.env_kwargs)
        flatten_observation = bool(env_kwargs.pop("flatten_observation", False))
        pass_render_mode = bool(env_kwargs.pop("pass_render_mode", True))

        make_kwargs = dict(env_kwargs)
        if pass_render_mode:
            make_kwargs["render_mode"] = self.render_mode

        if self.gym_env_name == "gym_navigation:NavigationTrack-v0":
            make_kwargs.setdefault("track_id", 1)
            env = gym.make(
                self.gym_env_name,
                **make_kwargs,
            )
            if flatten_observation:
                env = gym.wrappers.FlattenObservation(env)
            return env

        if self.gym_env_name == "maze-sample-3x3-v0":
            make_kwargs.pop("render_mode", None)
            env = gym.make(
                self.gym_env_name,
                enable_render=self.render_mode,
                **make_kwargs,
            )
            if flatten_observation:
                env = gym.wrappers.FlattenObservation(env)
            return env

        if new_reward:
            make_kwargs.setdefault("healthy_reward", 0)

        env = gym.make(
            self.gym_env_name,
            **make_kwargs,
        )
        if flatten_observation:
            env = gym.wrappers.FlattenObservation(env)
        return env

    def reset(self, new_reward=False):
        new_reward = bool(new_reward)
        should_recreate = (
            not hasattr(self, "env")
            or self.env is None
            or new_reward != self._new_reward_enabled
        )
        if should_recreate and hasattr(self, "env") and self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        if should_recreate:
            self.env = self._make_env(new_reward=new_reward)
            self._new_reward_enabled = new_reward

        state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0
        return state

    def step(self, action):
        self.steps += 1
        action = action[0]
        state, reward, terminated, truncated, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length or terminated or truncated:
            done = True
        else:
            done = False

        return state, reward, done

    def get_accu_reward(self):
        return self.accu_reward
