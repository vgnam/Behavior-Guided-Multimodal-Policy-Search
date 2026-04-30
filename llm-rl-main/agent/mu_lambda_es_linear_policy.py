"""
(mu, lambda)-ES baseline for linear policies.

This agent prefers EvoLib when available and falls back to neorl or a built-in
numpy backend otherwise.
"""

import os
import tempfile
import time
import numpy as np
import yaml

from world.base_world import BaseWorld
from agent.policy.linear_policy import LinearPolicy
from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.reward_summary import print_reward_summary

try:
    from neorl import ES as NEORL_ES
except ModuleNotFoundError:
    NEORL_ES = None

try:
    from evolib import Population as EVOLIB_Population
except Exception:
    EVOLIB_Population = None


class MuLambdaESLinearPolicyAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_length,
        num_evaluation_episodes,
        bias=True,
        mu=8,
        lam=32,
        sigma=0.1,
        sigma_decay=1.0,
        min_sigma=1e-12,
        max_sigma=None,
        param_bound=10.0,
        cxmode="blend",
        alpha=0.5,
        cxpb=0.6,
        mutpb=0.3,
        smin=0.01,
        smax=0.5,
        clip=True,
        ncores=1,
        candidate_evaluation_episodes=1,
        es_backend="auto",
        seed=None,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0.0
        self.total_steps = 0
        self.total_episodes = 0
        self.training_episodes = 0

        self.logdir = logdir
        self.dim_action = int(dim_action)
        self.dim_state = int(dim_state)
        self.max_traj_length = int(max_traj_length)
        self.num_evaluation_episodes = max(1, int(num_evaluation_episodes))
        self.bias = bool(bias)

        self.mu = max(1, int(mu))
        self.lam = max(2, int(lam))
        if self.mu > self.lam:
            raise ValueError("mu must be <= lam for (mu, lambda)-ES")

        self.sigma = float(sigma)
        self.sigma_decay = float(sigma_decay)
        self.min_sigma = float(min_sigma)
        self.max_sigma = None if max_sigma is None else float(max_sigma)
        self.param_bound = float(param_bound)

        self.cxmode = str(cxmode)
        self.alpha = float(alpha)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)
        self.smin = float(smin)
        self.smax = float(smax)
        self.clip = bool(clip)
        self.ncores = max(1, int(ncores))

        self.candidate_evaluation_episodes = max(1, int(candidate_evaluation_episodes))
        self.es_backend = str(es_backend).strip().lower()
        if self.es_backend not in {"auto", "evolib", "neorl", "numpy"}:
            raise ValueError("es_backend must be one of: auto, evolib, neorl, numpy")
        self.seed = seed
        self._backend_notice_printed = False

        self.rng = np.random.default_rng(seed)

        if not self.bias:
            self.policy = LinearPolicyNoBias(dim_actions=self.dim_action, dim_states=self.dim_state)
            self.rank = self.dim_action * self.dim_state
        else:
            self.policy = LinearPolicy(dim_actions=self.dim_action, dim_states=self.dim_state)
            self.rank = self.dim_action * self.dim_state + self.dim_action

        if self.param_bound <= 0:
            raise ValueError("param_bound must be > 0 for (mu, lambda)-ES")

        self.policy.initialize_policy()
        self.theta = self.policy.get_parameters().reshape(-1).astype(np.float32)
        self.bounds = {
            f"x{i + 1}": ["float", -self.param_bound, self.param_bound]
            for i in range(self.rank)
        }

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
        params = np.array(params_flat, dtype=np.float32).reshape(-1)
        params = self._clip_params(params)
        self.policy.update_policy(params)

    def _clip_params(self, params_flat):
        return np.clip(np.asarray(params_flat, dtype=np.float32), -self.param_bound, self.param_bound)

    def _build_initial_population(self, base_theta):
        pop = base_theta[None, :] + self.sigma * self.rng.normal(
            0.0,
            1.0,
            size=(self.lam, self.rank),
        ).astype(np.float32)
        pop = self._clip_params(pop)
        pop[0] = self._clip_params(base_theta)
        return pop.tolist()

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

    def _sample_offspring(self, parents):
        parent_count = parents.shape[0]
        if parent_count <= 0:
            raise ValueError("parents cannot be empty")

        parent_a = parents[self.rng.integers(parent_count)]
        parent_b = parents[self.rng.integers(parent_count)]

        if self.rng.random() < self.cxpb:
            if self.cxmode.lower() == "blend":
                # BLX-alpha style interpolation/extrapolation around two parents.
                gamma = self.rng.uniform(
                    low=-self.alpha,
                    high=1.0 + self.alpha,
                    size=self.rank,
                ).astype(np.float32)
                child = gamma * parent_a + (1.0 - gamma) * parent_b
            else:
                mix = self.rng.uniform(0.0, 1.0, size=self.rank).astype(np.float32)
                child = mix * parent_a + (1.0 - mix) * parent_b
        else:
            child = np.array(parent_a, dtype=np.float32, copy=True)

        if self.rng.random() < self.mutpb:
            step_scale = self.rng.uniform(self.smin, self.smax)
            mutation = step_scale * self.rng.normal(0.0, 1.0, size=self.rank).astype(np.float32)
            child = child + mutation

        return self._clip_params(child)

    def _print_backend_notice(self, message):
        if not self._backend_notice_printed:
            print(message)
            self._backend_notice_printed = True

    def _resolve_backend(self):
        if self.es_backend == "evolib":
            if EVOLIB_Population is not None:
                return "evolib"
            self._print_backend_notice(
                "[(mu, lambda)-ES] EvoLib requested but unavailable; using built-in numpy backend."
            )
            return "numpy_fallback"

        if self.es_backend == "neorl":
            if NEORL_ES is not None:
                return "neorl.ES"
            self._print_backend_notice(
                "[(mu, lambda)-ES] neorl requested but unavailable; using built-in numpy backend."
            )
            return "numpy_fallback"

        if self.es_backend == "numpy":
            return "numpy_fallback"

        # auto: prefer EvoLib first, then neorl, then numpy fallback.
        if EVOLIB_Population is not None:
            return "evolib"
        if NEORL_ES is not None:
            return "neorl.ES"
        self._print_backend_notice(
            "[(mu, lambda)-ES] EvoLib/neorl not found; using built-in numpy backend."
        )
        return "numpy_fallback"

    def _build_evolib_config(self, random_seed):
        return {
            "parent_pool_size": int(self.mu),
            "offspring_pool_size": int(self.lam),
            "max_generations": 1,
            "max_indiv_age": 0,
            "num_elites": 0,
            "random_seed": random_seed,
            "evolution": {"strategy": "mu_comma_lambda"},
            "modules": {
                "policy": {
                    "type": "vector",
                    "structure": "flat",
                    "dim": int(self.rank),
                    "initializer": "normal",
                    "bounds": [-self.param_bound, self.param_bound],
                    "init_bounds": [-self.param_bound, self.param_bound],
                    "mutation": {
                        "strategy": "constant",
                        "probability": float(self.mutpb),
                        "strength": float(max(self.sigma, self.min_sigma)),
                    },
                    "crossover": {
                        "strategy": "constant",
                        "probability": float(self.cxpb),
                        "operator": "blx",
                        "alpha": float(self.alpha),
                    },
                }
            },
        }

    def _evolve_with_evolib(self, world: BaseWorld, initial_population):
        if EVOLIB_Population is None:
            raise RuntimeError("EvoLib backend requested but unavailable")

        evolib_seed = None
        if self.seed is not None:
            evolib_seed = int(self.seed) + int(self.training_episodes)

        cfg = self._build_evolib_config(evolib_seed)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_mu_lambda_es_evolib.yaml",
            delete=False,
            encoding="utf-8",
        ) as tmp_cfg:
            yaml.safe_dump(cfg, tmp_cfg, sort_keys=False)
            cfg_path = tmp_cfg.name

        def _fitness(indiv):
            params = np.asarray(indiv.para["policy"].vector, dtype=np.float32)
            reward = self._evaluate_params(
                world,
                params,
                self.candidate_evaluation_episodes,
            )
            # EvoLib minimizes fitness by default, so convert reward maximization
            # into loss minimization.
            indiv.fitness = -float(reward)

        try:
            pop = EVOLIB_Population(config_path=cfg_path, fitness_function=_fitness)
        finally:
            try:
                os.remove(cfg_path)
            except OSError:
                pass

        parent_population = np.asarray(initial_population[: self.mu], dtype=np.float32)
        parent_population = self._clip_params(parent_population)

        for idx, indiv in enumerate(pop.indivs):
            indiv.para["policy"].vector = np.asarray(
                parent_population[idx],
                dtype=np.float32,
            )

        pop.run_one_generation()
        best_indiv = pop.best(sort=True)
        x_best = self._clip_params(np.asarray(best_indiv.para["policy"].vector, dtype=np.float32))
        best_loss = float(best_indiv.fitness)
        y_best = -best_loss

        es_hist = {
            "generation": int(pop.generation_num),
            "best_loss": best_loss,
            "best_reward": y_best,
            "mean_reward": float(-pop.mean_fitness),
            "median_reward": float(-pop.median_fitness),
            "worst_reward": float(-pop.worst_fitness),
        }

        return x_best, y_best, es_hist

    def _evolve_with_numpy(self, world: BaseWorld, initial_population):
        population = np.asarray(initial_population, dtype=np.float32)
        rewards = np.asarray(
            [
                self._evaluate_params(world, candidate, self.candidate_evaluation_episodes)
                for candidate in population
            ],
            dtype=np.float64,
        )

        ranked = np.argsort(rewards)[::-1]
        parents = population[ranked[: self.mu]]
        parent_rewards = rewards[ranked[: self.mu]]

        offspring = np.asarray(
            [self._sample_offspring(parents) for _ in range(self.lam)],
            dtype=np.float32,
        )
        offspring_rewards = np.asarray(
            [
                self._evaluate_params(world, candidate, self.candidate_evaluation_episodes)
                for candidate in offspring
            ],
            dtype=np.float64,
        )

        best_idx = int(np.argmax(offspring_rewards))
        x_best = np.asarray(offspring[best_idx], dtype=np.float32)
        y_best = float(offspring_rewards[best_idx])
        hist = {
            "parent_best_reward": float(np.max(parent_rewards)),
            "parent_mean_reward": float(np.mean(parent_rewards)),
            "offspring_best_reward": y_best,
            "offspring_mean_reward": float(np.mean(offspring_rewards)),
        }
        return x_best, y_best, hist

    def _update_sigma(self):
        self.sigma *= self.sigma_decay
        if self.max_sigma is not None:
            self.sigma = min(self.sigma, self.max_sigma)
        self.sigma = max(self.sigma, self.min_sigma)

    def train_policy(self, world: BaseWorld, logdir):
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0 for (mu, lambda)-ES")
        if self.sigma_decay <= 0:
            raise ValueError("sigma_decay must be > 0 for (mu, lambda)-ES")
        if self.smin <= 0 or self.smax < self.smin:
            raise ValueError("Require 0 < smin <= smax for (mu, lambda)-ES")
        if not (0.0 <= self.cxpb <= 1.0) or not (0.0 <= self.mutpb <= 1.0):
            raise ValueError("cxpb and mutpb must be in [0, 1] for (mu, lambda)-ES")

        base_theta = self.theta.copy()
        initial_population = self._build_initial_population(base_theta)

        backend_name = self._resolve_backend()
        if backend_name == "evolib":
            x_best, y_best, es_hist = self._evolve_with_evolib(world, initial_population)
        elif backend_name == "neorl.ES":
            def _fit(individual):
                params = self._clip_params(individual)
                return self._evaluate_params(
                    world,
                    params,
                    self.candidate_evaluation_episodes,
                )

            es_seed = None if self.seed is None else int(self.seed) + int(self.training_episodes)
            es = NEORL_ES(
                mode="max",
                bounds=self.bounds,
                fit=_fit,
                lambda_=self.lam,
                mu=self.mu,
                cxmode=self.cxmode,
                alpha=self.alpha,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                smin=self.smin,
                smax=self.smax,
                clip=self.clip,
                ncores=self.ncores,
                seed=es_seed,
            )
            x_best, y_best, es_hist = es.evolute(ngen=1, x0=initial_population, verbose=False)
        else:
            x_best, y_best, es_hist = self._evolve_with_numpy(world, initial_population)

        self.theta = self._clip_params(np.asarray(x_best, dtype=np.float32))
        step = self.theta - base_theta
        self._set_policy_from_flat(self.theta)

        self._update_sigma()

        flat_params_str = self._format_flat_params(self.theta)
        print(f"[(mu, lambda)-ES params] {flat_params_str}")

        with open(f"{logdir}/parameters.txt", "w", encoding="utf-8") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_flat.txt", "w", encoding="utf-8") as f:
            f.write(flat_params_str)
            f.write("\n")

        with open(f"{logdir}/mu_lambda_es_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write(f"backend: {backend_name}\n")
            f.write(f"mu: {self.mu}\n")
            f.write(f"lam: {self.lam}\n")
            f.write(f"candidate_evaluation_episodes: {self.candidate_evaluation_episodes}\n")
            f.write(f"sigma: {self.sigma:.10f}\n")
            f.write(f"sigma_decay: {self.sigma_decay:.10f}\n")
            f.write(f"min_sigma: {self.min_sigma:.10f}\n")
            f.write(f"max_sigma: {self.max_sigma}\n")
            f.write(f"param_bound: {self.param_bound}\n")
            f.write(f"cxmode: {self.cxmode}\n")
            f.write(f"alpha: {self.alpha}\n")
            f.write(f"cxpb: {self.cxpb}\n")
            f.write(f"mutpb: {self.mutpb}\n")
            f.write(f"smin: {self.smin}\n")
            f.write(f"smax: {self.smax}\n")
            f.write(f"clip: {self.clip}\n")
            f.write(f"ncores: {self.ncores}\n")
            f.write(f"best_reward: {float(y_best):.6f}\n")
            f.write(f"step_norm: {float(np.linalg.norm(step)):.6f}\n")
            f.write(f"history_keys: {list(es_hist.keys()) if isinstance(es_hist, dict) else type(es_hist)}\n")
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
