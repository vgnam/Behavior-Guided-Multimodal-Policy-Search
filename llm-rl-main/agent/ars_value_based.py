"""
Augmented Random Search (ARS) baseline for value-based (tabular Q) policies.
"""

import time
import numpy as np

from world.base_world import BaseWorld
from agent.policy.value_based_policy import ValueBasedPolicy
from agent.policy.openai_es_utils import batched_weighted_sum
from agent.policy.reward_summary import print_reward_summary


class ARSValueBasedAgent:
    def __init__(
        self,
        logdir,
        n_states,
        n_actions,
        max_traj_length,
        num_evaluation_episodes,
        n_directions=16,
        deltas_used=16,
        step_size=0.02,
        delta_std=0.03,
        shift=0.0,
        candidate_evaluation_episodes=1,
        reward_normalization_epsilon=1e-8,
        grad_batch_size=500,
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

        self.n_directions = max(1, int(n_directions))
        self.deltas_used = max(1, int(deltas_used))
        self.step_size = float(step_size)
        self.delta_std = float(delta_std)
        self.shift = float(shift)
        self.candidate_evaluation_episodes = max(1, int(candidate_evaluation_episodes))
        self.reward_normalization_epsilon = float(reward_normalization_epsilon)
        self.grad_batch_size = max(1, int(grad_batch_size))

        self.rng = np.random.default_rng(seed)

        self.policy = ValueBasedPolicy(n_states=self.n_states, n_actions=self.n_actions)
        self.rank = self.n_states * self.n_actions

        self.policy.initialize_policy()
        self.theta = self.policy.get_parameters().reshape(-1).astype(np.float32)

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

    def _set_policy_from_flat(self, params_flat):
        self.policy.update_policy(np.array(params_flat, dtype=np.float32).reshape(-1))

    def rollout_episode(self, world: BaseWorld, reward_shift=0.0, logging_file=None):
        state = world.reset()

        if logging_file is not None:
            logging_file.write(
                f"{', '.join(str(x) for x in self.policy.get_parameters().reshape(-1))}\n"
            )
            logging_file.write("parameter ends\n\n")
            logging_file.write("state | action | reward\n")

        done = False
        total_reward = 0.0
        while not done:
            action = int(self.policy.get_action(state))
            next_state, reward, done = world.step(action)
            shifted_reward = float(reward) - float(reward_shift)
            total_reward += shifted_reward

            if logging_file is not None:
                logging_file.write(f"{state} | {action} | {reward}\n")

            state = next_state
            self.total_steps += 1

        if logging_file is not None:
            logging_file.write(f"Total reward: {total_reward}\n")

        self.total_episodes += 1
        return float(total_reward)

    def _evaluate_params(self, world: BaseWorld, params_flat, n_episodes, reward_shift):
        self._set_policy_from_flat(params_flat)
        rewards = [
            self.rollout_episode(world, reward_shift=reward_shift, logging_file=None)
            for _ in range(n_episodes)
        ]
        return float(np.mean(rewards))

    def _estimate_gradient(self, selected_rewards, selected_directions):
        reward_std = float(np.std(selected_rewards))
        if not np.isfinite(reward_std) or reward_std < self.reward_normalization_epsilon:
            return np.zeros(self.rank, dtype=np.float32), reward_std

        weights = selected_rewards[:, 0] - selected_rewards[:, 1]
        grad, count = batched_weighted_sum(
            weights,
            (direction for direction in selected_directions),
            batch_size=self.grad_batch_size,
        )
        grad = np.asarray(grad, dtype=np.float32)

        expected_count = selected_directions.shape[0]
        if count != expected_count:
            raise RuntimeError("Invalid direction count during ARS gradient estimation")

        grad /= (float(expected_count) * reward_std)
        return grad, reward_std

    def train_policy(self, world: BaseWorld, logdir):
        if self.delta_std <= 0:
            raise ValueError("delta_std must be > 0 for ARS")

        num_directions = self.n_directions
        top_k = min(self.deltas_used, num_directions)

        base_theta = self.theta.copy()
        directions = self.rng.normal(0.0, 1.0, size=(num_directions, self.rank)).astype(np.float32)

        rewards = np.zeros((num_directions, 2), dtype=np.float32)
        for idx in range(num_directions):
            delta = self.delta_std * directions[idx]
            reward_pos = self._evaluate_params(
                world,
                base_theta + delta,
                self.candidate_evaluation_episodes,
                reward_shift=self.shift,
            )
            reward_neg = self._evaluate_params(
                world,
                base_theta - delta,
                self.candidate_evaluation_episodes,
                reward_shift=self.shift,
            )
            rewards[idx, 0] = reward_pos
            rewards[idx, 1] = reward_neg

        max_rewards = np.max(rewards, axis=1)
        selected_idx = np.argsort(max_rewards)[-top_k:]
        selected_rewards = rewards[selected_idx]
        selected_directions = directions[selected_idx]

        grad, reward_std = self._estimate_gradient(selected_rewards, selected_directions)
        step = self.step_size * grad

        self.theta = base_theta + step
        self._set_policy_from_flat(self.theta)

        flat_params_str = self._format_flat_params(self.theta)
        print(f"[ARS value-based params] {flat_params_str}")

        with open(f"{logdir}/parameters.txt", "w", encoding="utf-8") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_flat.txt", "w", encoding="utf-8") as f:
            f.write(flat_params_str)
            f.write("\n")

        with open(f"{logdir}/ars_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(f"n_directions: {num_directions}\n")
            f.write(f"deltas_used: {top_k}\n")
            f.write(f"delta_std: {self.delta_std}\n")
            f.write(f"step_size: {self.step_size}\n")
            f.write(f"shift: {self.shift}\n")
            f.write(
                f"candidate_evaluation_episodes: {self.candidate_evaluation_episodes}\n"
            )
            f.write(f"reward_std_selected: {reward_std:.10f}\n")
            f.write(f"max_reward_pos: {float(np.max(rewards[:, 0])):.6f}\n")
            f.write(f"max_reward_neg: {float(np.max(rewards[:, 1])):.6f}\n")
            f.write(f"mean_reward_pos: {float(np.mean(rewards[:, 0])):.6f}\n")
            f.write(f"mean_reward_neg: {float(np.mean(rewards[:, 1])):.6f}\n")
            f.write(f"grad_norm: {float(np.linalg.norm(grad)):.6f}\n")
            f.write(f"step_norm: {float(np.linalg.norm(step)):.6f}\n")
            f.write(f"selected_idx: {selected_idx.tolist()}\n")
            f.write(f"parameters_flat: {flat_params_str}\n")

        logging_filename = f"{logdir}/training_rollout.txt"
        with open(logging_filename, "w", encoding="utf-8") as logging_file:
            results = [
                self.rollout_episode(world, reward_shift=0.0, logging_file=logging_file)
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
                result = self.rollout_episode(world, reward_shift=0.0, logging_file=logging_file)
            results.append(result)
        return results
