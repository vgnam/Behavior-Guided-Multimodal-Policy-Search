"""
LLM-guided black-box optimization agent for MLP policies.

Supports:
  - Continuous action spaces (MuJoCo, etc.): get_action → raw values
  - Discrete action spaces (Atari, gridworld): get_action → logits, argmax used
  - Optional intrinsic-rank random projection: the LLM proposes `intrinsic_rank`
    real numbers, which are lifted back to the full MLP parameter space via a fixed
    random orthogonal projection matrix (same idea as LLMNumOptimRndmPrjAgent).
  - Optional observation random projection (handled inside MLPPolicy itself).
  - Optional ProPS-V vision mode: periodic VLM frame analysis + neighborhood sampling.
"""

import re
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.policy.mlp_policy import MLPPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias, ReplayBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld


class LLMNumOptimMLPAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        hidden_dims,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        optimum,
        search_step_size=0.1,
        intrinsic_rank=None,
        obs_projection_dim=None,
        is_discrete=False,
        # ProPS-V vision parameters
        enable_vision=False,
        env_desc_file=None,
        vlm_model_name="gpt-4o",
        decay_horizon=100,
        frame_sample_period=50,
        n_neighbors=5,
        poisson_lam=2.0,
        neighbor_step=0.1,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.vlm_api_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.is_discrete = is_discrete
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.max_traj_length = max_traj_length

        # Vision / ProPS-V
        self.enable_vision = enable_vision
        self.env_desc_file = env_desc_file
        self.n_neighbors = n_neighbors
        self.poisson_lam = poisson_lam
        self.neighbor_step = neighbor_step
        self.visual_analysis_history = []

        self.policy = MLPPolicy(
            dim_states=dim_state,
            dim_actions=dim_action,
            hidden_dims=list(hidden_dims),
            obs_projection_dim=obs_projection_dim,
        )
        full_param_count = self.policy.param_count

        # Intrinsic random projection in parameter space (optional)
        if intrinsic_rank and intrinsic_rank < full_param_count:
            G = np.random.randn(full_param_count, full_param_count)
            Q, _ = np.linalg.qr(G)
            self._proj_full_to_low = Q[:, :intrinsic_rank]     # (full, rank)
            self._proj_low_to_full = Q[:, :intrinsic_rank].T   # (rank, full)
            self.rank = intrinsic_rank
            self._uses_projection = True
        else:
            self._proj_full_to_low = None
            self._proj_low_to_full = None
            self.rank = full_param_count
            self._uses_projection = False

        print(
            f"[MLPAgent] param_count={full_param_count}, "
            f"rank={self.rank}, intrinsic_proj={self._uses_projection}, "
            f"obs_proj={obs_projection_dim is not None}, discrete={is_discrete}, "
            f"vision={enable_vision}"
        )

        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )

        if self.enable_vision:
            from agent.policy.frame_sampler import FrameSampler
            from agent.policy.adaptive_visual_guidance import AdaptiveVisualGuidance
            from agent.policy.vlm_analyzer import VLMAnalyzer
            self.frame_sampler = FrameSampler(sample_period=frame_sample_period)
            self.visual_guidance = AdaptiveVisualGuidance(decay_horizon=decay_horizon)
            self.vlm_analyzer = VLMAnalyzer(vlm_model_name=vlm_model_name)

    # ── Projection helpers ───────────────────────────────────────────────

    def _low_to_full(self, low: np.ndarray) -> np.ndarray:
        if self._uses_projection:
            return (self._proj_full_to_low @ low.reshape(-1)).reshape(-1)
        return low.reshape(-1)

    def _full_to_low(self, full: np.ndarray) -> np.ndarray:
        if self._uses_projection:
            return (self._proj_full_to_low.T @ full.reshape(-1)).reshape(-1)
        return full.reshape(-1)

    # ── Rollout ──────────────────────────────────────────────────────────

    def rollout_episode(self, world: BaseWorld, logging_file, record=True,
                        capture_frames=False):
        state = world.reset()
        state = np.expand_dims(state, axis=0)  # (1, obs_dim)

        # Log current (low-dim) parameters
        low_params = self._full_to_low(self.policy.get_parameters())
        logging_file.write(
            ", ".join(f"{v:.5g}" for v in low_params) + "\n"
        )
        logging_file.write("parameter ends\n\n")
        logging_file.write("state | action | reward\n")

        done = False
        step_idx = 0
        trajectory = []
        first_episode_done = False
        first_episode_reward = None
        first_episode_steps = None
        episode_num = 1

        while True:
            raw_action = self.policy.get_action(state.T)  # (1, dim_action)
            if self.is_discrete:
                action = int(np.argmax(raw_action))
            else:
                action = np.reshape(raw_action, (1, self.dim_action))
                if hasattr(world, "discretize") and world.discretize:
                    action = np.argmax(action)
                    action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action} | {reward}\n")

            # Optionally capture frames for VLM analysis
            frame = None
            if capture_frames and hasattr(world, "env"):
                is_sampled = (
                    step_idx == 0
                    or step_idx % self.frame_sampler.sample_period == 0
                    or done
                    or step_idx == self.max_traj_length - 1
                )
                if is_sampled:
                    try:
                        frame = world.env.render()
                        if not isinstance(frame, np.ndarray):
                            frame = None
                    except Exception:
                        frame = None

            if capture_frames:
                trajectory.append({
                    "state": state.T[0].copy(),
                    "action": action if self.is_discrete else action[0].copy(),
                    "reward": reward,
                    "frame": frame,
                    "episode_num": episode_num,
                })

            state = next_state
            step_idx += 1
            self.total_steps += 1

            if done:
                if not first_episode_done:
                    first_episode_done = True
                    first_episode_reward = world.get_accu_reward()
                    first_episode_steps = step_idx
                if capture_frames and step_idx < self.max_traj_length:
                    episode_num += 1
                    state = world.reset()
                    state = np.expand_dims(state, axis=0)
                    continue
                else:
                    break

            if step_idx >= self.max_traj_length:
                break

        total_reward = (
            first_episode_reward
            if first_episode_reward is not None
            else world.get_accu_reward()
        )
        terminated_early = (
            first_episode_steps < self.max_traj_length
            if first_episode_steps is not None
            else step_idx < self.max_traj_length
        )

        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        if record:
            self.replay_buffer.add(low_params, total_reward)

        if capture_frames:
            return total_reward, trajectory, terminated_early
        return total_reward

    # ── Neighborhood helpers (ProPS-V) ───────────────────────────────────

    def _generate_neighbors(self, params_arr, n):
        rng = np.random.default_rng()
        neighbors = []
        for _ in range(n):
            steps = rng.poisson(self.poisson_lam, size=len(params_arr))
            signs = rng.choice([-1, 1], size=len(params_arr))
            delta = steps * signs * self.neighbor_step
            neighbor = np.clip(params_arr + delta, -6.0, 6.0)
            neighbors.append(neighbor)
        return neighbors

    def _rollout_neighbors(self, world, anchor_params, label, logdir):
        all_params = [anchor_params] + self._generate_neighbors(anchor_params, self.n_neighbors)
        rollout_data = []
        for idx, params in enumerate(all_params):
            self.policy.update_policy(self._low_to_full(params))
            with open(f"{logdir}/nb_{label}_{idx}.txt", "w") as f:
                reward, traj, term_early = self.rollout_episode(
                    world, f, record=False, capture_frames=True
                )
            params_str = ", ".join(f"params[{j}]: {v:.5g}" for j, v in enumerate(params))
            rollout_data.append({"params": params, "params_str": params_str,
                                  "reward": reward, "trajectory": traj,
                                  "terminated_early": term_early})

        def _analyze(entry, idx):
            frame_indices = self.frame_sampler.sample_frames(
                entry["trajectory"], entry["terminated_early"], self.max_traj_length
            )
            frames = self.frame_sampler.get_frames(entry["trajectory"], frame_indices)
            if not frames or not any(f.get("frame") is not None for f in frames):
                return idx, None, 0.0
            analysis, vlm_time = self.vlm_analyzer.analyze_frames(
                frames,
                self.env_desc_file if self.env_desc_file else "RL Environment",
                entry["reward"],
                entry["terminated_early"],
                current_params=entry["params_str"],
            )
            return idx, analysis, vlm_time

        analyses = [None] * len(rollout_data)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_analyze, e, i): i for i, e in enumerate(rollout_data)}
            for future in as_completed(futures):
                idx, analysis, vlm_time = future.result()
                analyses[idx] = analysis
                self.vlm_api_time += vlm_time
                self.api_call_time += vlm_time

        results = [
            {"params_str": e["params_str"], "reward": e["reward"], "analysis": analyses[i]}
            for i, e in enumerate(rollout_data)
        ]
        anchor_result = results[0]
        neighbor_results = sorted(results[1:], key=lambda x: x["reward"], reverse=True)
        return anchor_result, neighbor_results

    # ── Warmup ───────────────────────────────────────────────────────────

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        os.makedirs(logdir, exist_ok=True)
        for ep in range(num_episodes):
            self.policy.initialize_policy()
            print(f"Rolling out warmup episode {ep}...")
            with open(f"{logdir}/warmup_rollout_{ep}.txt", "w") as f:
                result = self.rollout_episode(world, f)
            print(f"  Result: {result}")

    # ── Training step ────────────────────────────────────────────────────

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(
                r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
            )
            matches = pattern.findall(s)
            results = [float(m[1]) for m in matches]
            assert len(results) == self.rank, (
                f"Expected {self.rank} params, got {len(results)}"
            )
            return np.array(results, dtype=float)

        def str_nd_examples(n: int) -> str:
            text = ""
            for weights, reward in self.replay_buffer.buffer:
                line = "; ".join(f"params[{i}]: {weights[i]:.4g}" for i in range(n))
                text += f"{line}; f(params): {reward:.3f}\n"
            return text

        neighborhood_analysis = None

        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            use_vision = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                random_state=np.random.RandomState(self.training_episodes)
            )
            print(f"[ProPS-V MLP] iter={self.training_episodes}, λ={lambda_t:.3f}, VLM={use_vision}")

            if use_vision:
                saved_params = self._full_to_low(self.policy.get_parameters()).copy()

                # CURRENT anchor
                print("[Neighborhood] Anchor: CURRENT")
                cur_anchor, cur_neighbors = self._rollout_neighbors(
                    world, saved_params, "current", logdir
                )
                if cur_anchor["analysis"]:
                    self.visual_analysis_history.append({
                        "iteration": self.training_episodes,
                        "lambda": lambda_t,
                        "reward": cur_anchor["reward"],
                        "analysis": cur_anchor["analysis"],
                        "params": cur_anchor["params_str"],
                    })
                    with open(f"{logdir}/vlm_analysis.txt", "w", encoding="utf-8") as vf:
                        vf.write(cur_anchor["analysis"])

                # BEST / WORST anchors
                best_nb_anchor = worst_nb_anchor = None
                best_nb_neighbors = worst_nb_neighbors = []
                if len(self.replay_buffer.buffer) > 0:
                    rb = self.replay_buffer.buffer
                    best_rb_params, best_rb_reward = max(rb, key=lambda x: x[1])
                    worst_rb_params, worst_rb_reward = min(rb, key=lambda x: x[1])
                    print(f"[Neighborhood] Anchor: BEST (reward={best_rb_reward:.2f})")
                    best_nb_anchor, best_nb_neighbors = self._rollout_neighbors(
                        world, np.array(best_rb_params), "best", logdir
                    )
                    if worst_rb_reward != best_rb_reward:
                        print(f"[Neighborhood] Anchor: WORST (reward={worst_rb_reward:.2f})")
                        worst_nb_anchor, worst_nb_neighbors = self._rollout_neighbors(
                            world, np.array(worst_rb_params), "worst", logdir
                        )

                # Restore policy
                self.policy.update_policy(self._low_to_full(saved_params))

                def _anchor_block(label, anchor_res, neighbors):
                    lines = [f"### Anchor [{label}]  reward={anchor_res['reward']:.2f}",
                             f"  Params : {anchor_res['params_str']}",
                             f"  Behavior: {anchor_res['analysis'] or '(no VLM analysis)'}"]
                    if neighbors:
                        lines.append(f"\n  ---- Neighbors of [{label}] ----")
                        for rank_i, nb in enumerate(neighbors, 1):
                            lines += [f"  Neighbor #{rank_i}  reward={nb['reward']:.2f}",
                                      f"    Params : {nb['params_str']}",
                                      f"    Behavior: {nb['analysis'] or '(no VLM analysis)'}"]
                    return "\n".join(lines)

                blocks = ["## Neighborhood Behavioral Landscape\n",
                          _anchor_block("CURRENT", cur_anchor, cur_neighbors)]
                if best_nb_anchor:
                    blocks.append(_anchor_block("BEST (replay buffer)", best_nb_anchor, best_nb_neighbors))
                if worst_nb_anchor:
                    blocks.append(_anchor_block("WORST (replay buffer)", worst_nb_anchor, worst_nb_neighbors))
                neighborhood_analysis = "\n\n".join(blocks)
                with open(f"{logdir}/neighborhood_analysis.txt", "w", encoding="utf-8") as nf:
                    nf.write(neighborhood_analysis)

        # LLM call
        print("Updating MLP policy via LLM...")
        if self.enable_vision:
            new_low_params, reasoning, api_time = (
                self.llm_brain.llm_update_parameters_num_optim_vision(
                    str_nd_examples(self.rank),
                    parse_parameters,
                    self.training_episodes,
                    self.env_desc_file,
                    rank=self.rank,
                    optimum=self.optimum,
                    search_step_size=self.search_step_size,
                    neighborhood_analysis=neighborhood_analysis,
                )
            )
        elif self.env_desc_file is not None:
            # ProPS+ (semantics): env description passed to LLM but no VLM
            new_low_params, reasoning, api_time = (
                self.llm_brain.llm_update_parameters_num_optim_semantics(
                    str_nd_examples(self.rank),
                    parse_parameters,
                    self.training_episodes,
                    self.env_desc_file,
                    rank=self.rank,
                    optimum=self.optimum,
                    search_step_size=self.search_step_size,
                )
            )
        else:
            new_low_params, reasoning, api_time = (
                self.llm_brain.llm_update_parameters_num_optim(
                    str_nd_examples(self.rank),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    self.search_step_size,
                )
            )
        self.api_call_time += api_time

        # Lift from intrinsic → full param space and apply to policy
        new_full_params = self._low_to_full(new_low_params)
        self.policy.update_policy(new_full_params)

        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_reasoning.txt", "w", encoding="utf-8") as f:
            f.write(reasoning)
        print("Policy updated!")

        # Evaluation rollouts
        print(f"Rolling out episode {self.training_episodes}...")
        results = []
        with open(f"{logdir}/training_rollout.txt", "w") as lf:
            for _ in range(self.num_evaluation_episodes):
                results.append(self.rollout_episode(world, lf, record=False))
        print(f"Results: {results}")
        result = float(np.mean(results))

        self.replay_buffer.add(new_low_params, result)
        self.training_episodes += 1

        return (
            time.process_time() - self.start_time,
            self.api_call_time,
            self.total_episodes,
            self.total_steps,
            result,
        )
