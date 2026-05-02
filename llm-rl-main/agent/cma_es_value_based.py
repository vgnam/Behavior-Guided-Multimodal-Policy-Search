import pickle
import time

import cma
import numpy as np

from world.base_world import BaseWorld
from agent.policy.value_based_policy import ValueBasedPolicy
from agent.policy.reward_summary import print_reward_summary


class CMAESValueBasedAgent:
    """CMA-ES for tabular value-based policies (discrete state/action)."""

    def __init__(
        self,
        logdir,
        n_states,
        n_actions,
        max_traj_length,
        num_evaluation_episodes,
        population_size=32,
        sigma=0.1,
        elite_count=None,
        candidate_evaluation_episodes=1,
        covariance_type="auto",
        full_covariance_max_dim=256,
        decomposition_frequency=None,
        min_sigma=1e-12,
        max_sigma=None,
        seed=None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0.0
        self.total_steps = 0
        self.total_episodes = 0
        self.training_episodes = 0

        self.logdir = logdir
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.max_traj_length = int(max_traj_length)
        self.num_evaluation_episodes = max(1, int(num_evaluation_episodes))
        self.population_size = max(2, int(population_size))
        self.initial_sigma = float(sigma)
        self.elite_count = None if elite_count is None else max(1, int(elite_count))
        self.candidate_evaluation_episodes = max(1, int(candidate_evaluation_episodes))
        self.covariance_type = str(covariance_type)
        self.full_covariance_max_dim = max(1, int(full_covariance_max_dim))
        self.decomposition_frequency = (
            None if decomposition_frequency is None else max(1, int(decomposition_frequency))
        )
        self.min_sigma = float(min_sigma)
        self.max_sigma = None if max_sigma is None else float(max_sigma)
        self.seed = seed

        self.policy = ValueBasedPolicy(n_states=self.n_states, n_actions=self.n_actions)
        self.rank = self.n_states * self.n_actions

        self.policy.initialize_policy()
        self.theta = self.policy.get_parameters().reshape(-1).astype(np.float64)
        self.resolved_covariance_type = self._resolve_covariance_type()
        self.es = self._build_strategy(self.theta)

    @staticmethod
    def _format_flat_params(params_flat):
        params_flat = np.asarray(params_flat, dtype=np.float64).reshape(-1)
        return np.array2string(
            params_flat,
            precision=6,
            separator=", ",
            threshold=np.inf,
            max_line_width=10_000,
        )

    def _resolve_covariance_type(self):
        covariance_type = self.covariance_type.lower()
        if covariance_type in {"full", "diagonal"}:
            return covariance_type
        if covariance_type != "auto":
            raise ValueError(
                f"Unsupported covariance_type: {self.covariance_type}. Use auto, full, or diagonal."
            )
        return "full" if self.rank <= self.full_covariance_max_dim else "diagonal"

    def _build_strategy(self, initial_mean):
        options = {
            "popsize": self.population_size,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "verb_time": 0,
        }
        if self.seed is not None:
            options["seed"] = int(self.seed)
        if self.elite_count is not None:
            options["CMA_mu"] = int(self.elite_count)
        options["CMA_diagonal"] = True if self.resolved_covariance_type == "diagonal" else 0
        return cma.CMAEvolutionStrategy(
            np.asarray(initial_mean, dtype=np.float64).tolist(),
            self.initial_sigma,
            options,
        )

    def _current_mean(self):
        mean = getattr(self.es, "mean", None)
        if mean is None:
            mean = getattr(getattr(self.es, "result", None), "xfavorite", None)
        if mean is None:
            raise RuntimeError("Unable to recover the current CMA-ES mean.")
        return np.asarray(mean, dtype=np.float64).reshape(-1)

    def _set_policy_from_flat(self, params_flat):
        self.policy.update_policy(np.array(params_flat, dtype=np.float32).reshape(-1))

    def rollout_episode(self, world: BaseWorld, logging_file=None):
        state = world.reset()

        if logging_file is not None:
            logging_file.write("state | action | reward\n")

        done = False
        while not done:
            action = int(self.policy.get_action(state))
            next_state, reward, done = world.step(action)

            if logging_file is not None:
                logging_file.write(f"{state} | {action} | {reward}\n")

            state = next_state
            self.total_steps += 1

        if logging_file is not None:
            logging_file.write(f"Total reward: {world.get_accu_reward()}\n")

        self.total_episodes += 1
        return float(world.get_accu_reward())

    def _evaluate_params(self, world: BaseWorld, params_flat, n_episodes):
        self._set_policy_from_flat(params_flat)
        rewards = [self.rollout_episode(world, logging_file=None) for _ in range(n_episodes)]
        return float(np.mean(rewards))

    def train_policy(self, world: BaseWorld, logdir):
        previous_theta = self.theta.copy()
        candidates = self.es.ask()
        rewards = np.zeros(len(candidates), dtype=np.float64)

        for idx, candidate in enumerate(candidates):
            rewards[idx] = self._evaluate_params(
                world,
                candidate,
                self.candidate_evaluation_episodes,
            )

        costs = (-rewards).tolist()
        self.es.tell(candidates, costs)

        if self.max_sigma is not None:
            self.es.sigma = min(float(self.es.sigma), self.max_sigma)
        if self.min_sigma is not None:
            self.es.sigma = max(float(self.es.sigma), self.min_sigma)

        self.theta = self._current_mean()
        self._set_policy_from_flat(self.theta)

        elite_count = int(getattr(getattr(self.es, "sp", None), "mu", self.elite_count or 0))
        stop_conditions = dict(self.es.stop())
        flat_params_str = self._format_flat_params(self.theta)

        print(f"[CMA-ES value-based params] {flat_params_str}")

        with open(f"{logdir}/parameters.txt", "w", encoding="utf-8") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_flat.txt", "w", encoding="utf-8") as f:
            f.write(flat_params_str)
            f.write("\n")

        with open(f"{logdir}/cma_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("backend: pycma\n")
            f.write("policy_type: value_based\n")
            f.write(f"population_size: {int(getattr(self.es, 'popsize', len(candidates)))}\n")
            f.write(f"elite_count: {elite_count}\n")
            f.write(f"candidate_evaluation_episodes: {self.candidate_evaluation_episodes}\n")
            f.write(f"covariance_type: {self.resolved_covariance_type}\n")
            f.write(f"full_covariance_max_dim: {self.full_covariance_max_dim}\n")
            f.write(f"generation: {int(getattr(self.es, 'countiter', 0))}\n")
            f.write(f"sigma: {float(self.es.sigma):.10f}\n")
            f.write(f"best_reward: {float(np.max(rewards)):.6f}\n")
            f.write(f"mean_reward: {float(np.mean(rewards)):.6f}\n")
            f.write(f"median_reward: {float(np.median(rewards)):.6f}\n")
            f.write(f"worst_reward: {float(np.min(rewards)):.6f}\n")
            f.write(f"step_norm: {float(np.linalg.norm(self.theta - previous_theta)):.6f}\n")
            f.write(f"stop_conditions: {stop_conditions}\n")
            f.write(f"parameters_flat: {flat_params_str}\n")

        logging_filename = f"{logdir}/training_rollout.txt"
        with open(logging_filename, "w", encoding="utf-8") as logging_file:
            results = [
                self.rollout_episode(world, logging_file)
                for _ in range(self.num_evaluation_episodes)
            ]

        mean_reward, _ = print_reward_summary(results)
        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = mean_reward
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            with open(logging_filename, "w", encoding="utf-8") as logging_file:
                result = self.rollout_episode(world, logging_file)
            results.append(result)
        return results

    def save_state(self, path):
        state = {
            "theta": self.theta,
            "strategy": self.es,
            "np_random_state": np.random.get_state(),
            "resolved_covariance_type": self.resolved_covariance_type,
            "policy_params": self.policy.get_parameters().copy(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.es = state["strategy"]
        self.resolved_covariance_type = state.get(
            "resolved_covariance_type",
            self._resolve_covariance_type(),
        )
        np_random_state = state.get("np_random_state")
        if np_random_state is not None:
            np.random.set_state(np_random_state)

        saved_params = state.get("policy_params")
        if saved_params is not None:
            self.policy.update_policy(saved_params)

        self.theta = np.asarray(state.get("theta", self._current_mean()), dtype=np.float64)
        self._set_policy_from_flat(self.theta)
