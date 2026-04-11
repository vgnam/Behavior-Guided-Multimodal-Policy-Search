"""
OpenAI-ES baseline for linear policies.

This agent performs parameter-space search with antithetic Gaussian
perturbations and updates the policy via an ES gradient estimate.

The update path follows the same high-level flow used in
openai/evolution-strategies-starter:
    - paired positive/negative perturbations
    - return processing (centered rank / sign variants)
    - batched weighted sum for gradient estimate
    - optimizer step (SGD or Adam)
"""

import time
import numpy as np

from world.base_world import BaseWorld
from agent.policy.linear_policy import LinearPolicy
from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.openai_es_utils import (
        AdamFlat,
        SGDFlat,
        batched_weighted_sum,
        compute_centered_ranks,
)
from agent.policy.reward_summary import print_reward_summary


class OpenAIESLinearPolicyAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_length,
        num_evaluation_episodes,
        bias=True,
        population_size=32,
        sigma=0.1,
        noise_stdev=None,
        learning_rate=0.03,
        candidate_evaluation_episodes=1,
        return_proc_mode="centered_rank",
        optimizer_type="adam",
        use_centered_ranks=True,
        use_adam=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        l2coeff=None,
        grad_batch_size=500,
        sgd_momentum=0.9,
        seed=None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0.0
        self.total_steps = 0
        self.total_episodes = 0
        self.training_episodes = 0

        self.logdir = logdir
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.max_traj_length = max_traj_length
        self.num_evaluation_episodes = max(1, int(num_evaluation_episodes))
        self.bias = bias

        self.population_size = max(2, int(population_size))
        self.sigma = float(sigma)
        self.noise_stdev = float(noise_stdev) if noise_stdev is not None else self.sigma
        self.learning_rate = float(learning_rate)
        self.candidate_evaluation_episodes = max(1, int(candidate_evaluation_episodes))
        self.return_proc_mode = str(return_proc_mode)
        self.use_centered_ranks = bool(use_centered_ranks)
        self.use_adam = bool(use_adam)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)
        self.l2coeff = float(weight_decay if l2coeff is None else l2coeff)
        self.grad_batch_size = max(1, int(grad_batch_size))
        self.sgd_momentum = float(sgd_momentum)

        if self.return_proc_mode == "centered_rank" and not self.use_centered_ranks:
            self.return_proc_mode = "raw"

        self.rng = np.random.default_rng(seed)

        if not self.bias:
            self.policy = LinearPolicyNoBias(dim_actions=dim_action, dim_states=dim_state)
            self.rank = dim_action * dim_state
        else:
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
            self.rank = dim_action * dim_state + dim_action

        self.policy.initialize_policy()
        self.theta = self.policy.get_parameters().reshape(-1).astype(np.float32)

        # Keep backward compatibility with old bool flag.
        if optimizer_type is None:
            optimizer_type = "adam" if self.use_adam else "sgd"
        if use_adam is not None:
            optimizer_type = "adam" if bool(use_adam) else "sgd"
        self.optimizer_type = str(optimizer_type).lower()

        if self.optimizer_type == "adam":
            self.optimizer = AdamFlat(
                self.rank,
                stepsize=self.learning_rate,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
                epsilon=self.adam_epsilon,
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = SGDFlat(
                self.rank,
                stepsize=self.learning_rate,
                momentum=self.sgd_momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    def _set_policy_from_flat(self, params_flat):
        self.policy.update_policy(np.array(params_flat, dtype=np.float32).reshape(-1))

    def _state_to_features(self, raw_state):
        arr = np.asarray(raw_state)

        # Discrete encoded states are converted to one-hot features.
        if np.isscalar(raw_state) or arr.ndim == 0:
            idx = int(arr.item())
            idx = min(max(idx, 0), self.dim_state - 1)
            feat = np.zeros(self.dim_state, dtype=np.float32)
            feat[idx] = 1.0
            return feat

        vec = arr.astype(np.float32).reshape(-1)

        if vec.size == self.dim_state:
            return vec

        if vec.size == 1 and self.dim_state > 1:
            idx = int(vec[0])
            idx = min(max(idx, 0), self.dim_state - 1)
            feat = np.zeros(self.dim_state, dtype=np.float32)
            feat[idx] = 1.0
            return feat

        if vec.size < self.dim_state:
            pad = np.zeros(self.dim_state - vec.size, dtype=np.float32)
            return np.concatenate([vec, pad], axis=0)

        return vec[: self.dim_state]

    def rollout_episode(self, world: BaseWorld, logging_file=None):
        state = self._state_to_features(world.reset())
        state = np.expand_dims(state, axis=0)

        if logging_file is not None:
            logging_file.write(
                f"{', '.join(str(x) for x in self.policy.get_parameters().reshape(-1))}\n"
            )
            logging_file.write("parameter ends\n\n")
            logging_file.write("state | action | reward\n")

        done = False
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])

            next_state, reward, done = world.step(action)

            if logging_file is not None:
                logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")

            state = self._state_to_features(next_state)
            self.total_steps += 1

        if logging_file is not None:
            logging_file.write(f"Total reward: {world.get_accu_reward()}\n")

        self.total_episodes += 1
        return float(world.get_accu_reward())

    def _evaluate_params(self, world: BaseWorld, params_flat, n_episodes):
        self._set_policy_from_flat(params_flat)
        rewards = [self.rollout_episode(world, logging_file=None) for _ in range(n_episodes)]
        return float(np.mean(rewards))

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            random_params = self.rng.normal(0.0, 3.0, size=self.rank)
            self._set_policy_from_flat(random_params)

            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            with open(logging_filename, "w", encoding="utf-8") as logging_file:
                result = self.rollout_episode(world, logging_file)
            print(f"[OpenAI-ES warmup] episode={episode} reward={result:.3f}")

        self.theta = self.policy.get_parameters().reshape(-1).astype(np.float32)

    def train_policy(self, world: BaseWorld, logdir):
        if self.noise_stdev <= 0:
            raise ValueError("noise_stdev (or sigma) must be > 0 for OpenAI-ES")

        base_theta = self.theta.copy()
        noises = self.rng.normal(0.0, 1.0, size=(self.population_size, self.rank)).astype(np.float32)

        returns_n2 = np.zeros((self.population_size, 2), dtype=np.float32)
        signreturns_n2 = np.zeros((self.population_size, 2), dtype=np.float32)

        for i in range(self.population_size):
            delta = self.noise_stdev * noises[i]
            reward_pos = self._evaluate_params(
                world,
                base_theta + delta,
                self.candidate_evaluation_episodes,
            )
            reward_neg = self._evaluate_params(
                world,
                base_theta - delta,
                self.candidate_evaluation_episodes,
            )
            returns_n2[i, 0] = reward_pos
            returns_n2[i, 1] = reward_neg
            signreturns_n2[i, 0] = np.sign(reward_pos)
            signreturns_n2[i, 1] = np.sign(reward_neg)

        if self.return_proc_mode == "centered_rank":
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif self.return_proc_mode == "sign":
            proc_returns_n2 = signreturns_n2
        elif self.return_proc_mode == "centered_sign_rank":
            proc_returns_n2 = compute_centered_ranks(signreturns_n2)
        elif self.return_proc_mode == "raw":
            proc_returns_n2 = returns_n2
        else:
            raise NotImplementedError(self.return_proc_mode)

        weights = proc_returns_n2[:, 0] - proc_returns_n2[:, 1]
        grad, count = batched_weighted_sum(
            weights,
            (noise for noise in noises),
            batch_size=self.grad_batch_size,
        )
        grad = np.asarray(grad, dtype=np.float32)
        grad /= returns_n2.size

        if grad.shape != (self.rank,) or count != self.population_size:
            raise RuntimeError("Invalid gradient shape/count during OpenAI-ES update")

        # Follows OpenAI ES: optimizer.update(-g + l2coeff * theta)
        globalg = -grad + self.l2coeff * base_theta
        self.theta, update_ratio = self.optimizer.update(base_theta, globalg)
        self._set_policy_from_flat(self.theta)

        step = self.theta - base_theta

        with open(f"{logdir}/parameters.txt", "w", encoding="utf-8") as f:
            f.write(str(self.policy))

        with open(f"{logdir}/es_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(f"population_size: {self.population_size}\n")
            f.write(f"noise_stdev: {self.noise_stdev}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")
            f.write(f"return_proc_mode: {self.return_proc_mode}\n")
            f.write(f"optimizer_type: {self.optimizer_type}\n")
            f.write(f"l2coeff: {self.l2coeff}\n")
            f.write(f"reward_pos_mean: {float(np.mean(returns_n2[:, 0])):.6f}\n")
            f.write(f"reward_neg_mean: {float(np.mean(returns_n2[:, 1])):.6f}\n")
            f.write(f"grad_norm: {float(np.linalg.norm(grad)):.6f}\n")
            f.write(f"step_norm: {float(np.linalg.norm(step)):.6f}\n")
            f.write(f"update_ratio: {update_ratio:.8f}\n")

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
