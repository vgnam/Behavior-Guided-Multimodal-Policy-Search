"""
BMPS Q-Table Agent: Vision-Guided Q-Learning

This module implements BMPS for discrete state spaces (Q-tables).
Key components:
- Q-table policy for discrete state-action spaces
- Periodic frame sampling
- Adaptive visual guidance annealing
- VLM-based visual analysis
"""

from agent.policy.q_table import QTable
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from agent.policy.frame_sampler import FrameSampler
from agent.policy.adaptive_visual_guidance import AdaptiveVisualGuidance
from agent.policy.vlm_analyzer import VLMAnalyzer
from world.base_world import BaseWorld
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import traceback
import numpy as np
import re
import time


class LLMNumOptimQTableVisionAgent:
    """
    BMPS Agent for Q-table learning with vision-guided optimization.
    
    Extends standard Q-learning with:
    - Periodic frame sampling
    - Adaptive visual guidance schedule
    - VLM analysis for policy improvement
    """
    
    def __init__(
        self,
        logdir,
        actions,
        states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        optimum,
        env_desc_file=None,
        vlm_model_name="openrouter/google/gemma-3-27b-it",
        decay_horizon=100,
        frame_sample_period=50,
        enable_vision=True,
        env_kwargs=None,
        n_neighbors=5,
        poisson_lam=2.0,
        neighbor_step=0.1,
        ablate_anchor=None,
        llm_api_key=None,
        llm_api_base=None,
        vlm_api_key=None,
        vlm_api_base=None,
    ):
        """
        Initialize BMPS Q-Table agent.
        
        Args:
            logdir: Directory for logging
            actions: Action space description
            states: State space description
            max_traj_count: Maximum trajectories in replay buffer
            max_traj_length: Maximum trajectory length
            llm_si_template: Jinja2 template for system prompt
            llm_output_conversion_template: Template for output formatting
            llm_model_name: Name of LLM for policy optimization
            num_evaluation_episodes: Number of episodes for evaluation
            optimum: Expected optimal reward
            env_desc_file: Path to environment description file
            vlm_model_name: Name of VLM model for visual analysis
            decay_horizon: T_decay for visual guidance annealing
            frame_sample_period: P — capture a VLM frame every P timesteps
            enable_vision: Whether to enable vision-guided features            env_kwargs: Additional environment kwargs
            n_neighbors: Number of Poisson-perturbed neighbors per anchor
            poisson_lam: Lambda (mean) of Poisson distribution for step sizes
            neighbor_step: Base step size multiplied by Poisson sample
        """
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.vlm_api_time = 0
        self.total_llm_prompt_tokens = 0
        self.total_llm_completion_tokens = 0
        self.total_vlm_prompt_tokens = 0
        self.total_vlm_completion_tokens = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.actions = actions
        self.states = states
        self.optimum = optimum
        self.env_kwargs = env_kwargs
        self.env_desc_file = env_desc_file
        self.max_traj_length = max_traj_length
        
        # Vision-specific parameters
        self.enable_vision = enable_vision
        self.n_neighbors = n_neighbors
        self.poisson_lam = poisson_lam
        self.neighbor_step = neighbor_step
        self.ablate_anchor = ablate_anchor
        self.visual_analysis_history = []  # Store visual analyses
        
        # Q-table policy
        self.q_table = QTable(actions=actions, states=states)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        
        # LLM brain
        self.llm_brain = LLMBrain(
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
        )
        
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.rank = len(self.q_table.mapping)
        
        # Initialize vision components if enabled
        if self.enable_vision:
            self.frame_sampler = FrameSampler(
                sample_period=frame_sample_period
            )
            self.visual_guidance = AdaptiveVisualGuidance(
                decay_horizon=decay_horizon
            )
            self.vlm_analyzer = VLMAnalyzer(
                vlm_model_name=vlm_model_name,
                vlm_api_key=vlm_api_key,
                vlm_api_base=vlm_api_base,
            )
            print(f"[BMPS Q-Table] Vision features enabled (VLM: {vlm_model_name}, T_decay: {decay_horizon})")
        else:
            print("[BMPS Q-Table] Vision features disabled - using numerical optimization only")

    def rollout_episode_with_frames(
        self,
        world: BaseWorld,
        logging_file,
        record=True,
        capture_frames=False
    ):
        """
        Rollout one episode and optionally capture frames for visual analysis.
        
        Args:
            world: Environment
            logging_file: File for logging
            record: Whether to record in replay buffer
            capture_frames: Whether to capture rendered frames
            
        Returns:
            Tuple of (total_reward, trajectory_with_frames, terminated_early)
        """
        state = world.reset()
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        episode_num = 1
        trajectory = []
        first_episode_done = False
        first_episode_reward = None
        first_episode_steps = None
        
        while True:
            action = self.q_table.get_action(state)
            action = int(np.reshape(action, (1,)))
            
            # Capture frame only at sampled timesteps to avoid rendering every step
            frame = None
            if capture_frames and hasattr(world.env, 'render'):
                is_sampled_step = (
                    step_idx == 0
                    or step_idx % self.frame_sampler.sample_period == 0
                    or step_idx == self.max_traj_length - 1
                )
                if is_sampled_step:
                    try:
                        frame = world.env.render()
                        if not isinstance(frame, np.ndarray):
                            frame = None
                    except:
                        frame = None
            
            next_state, reward, done = world.step(action)
            # Also render on done (last frame of episode) if not already rendered
            if done and capture_frames and frame is None and hasattr(world.env, 'render'):
                try:
                    frame = world.env.render()
                    if not isinstance(frame, np.ndarray):
                        frame = None
                except:
                    frame = None
            logging_file.write(f"{state} | {action} | {reward}\n")
            
            # Store trajectory step with frame
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'frame': frame,
                'episode_num': episode_num,
            })
            
            step_idx += 1
            self.total_steps += 1
            
            if done:
                # Record first natural episode completion for reward/terminated_early
                if not first_episode_done:
                    first_episode_done = True
                    first_episode_reward = world.get_accu_reward()
                    first_episode_steps = step_idx
                
                if capture_frames and step_idx < self.max_traj_length:
                    # Pre-rollout: auto-reset and keep collecting frames for VLM
                    # so we always have a full max_traj_length trajectory
                    episode_num += 1
                    state = world.reset()
                    continue
                else:
                    break
            else:
                state = next_state
            
            if step_idx >= self.max_traj_length:
                break
        
        # Use reward/steps from the first natural episode end when available
        if first_episode_reward is not None:
            total_reward = first_episode_reward
            terminated_early = (first_episode_steps < self.max_traj_length)
        else:
            total_reward = world.get_accu_reward()
            terminated_early = (step_idx < self.max_traj_length)
        
        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        
        if record:
            self.replay_buffer.add(
                np.array([self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]),
                total_reward,
            )
        
        return total_reward, trajectory, terminated_early

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        """Standard rollout without frame capture (for compatibility)."""
        total_reward, _, _ = self.rollout_episode_with_frames(
            world, logging_file, record, capture_frames=False
        )
        return total_reward

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        """Warmup phase with random Q-table initialization."""
        for episode in range(num_episodes):
            self.q_table.initialize_policy()
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    # ------------------------------------------------------------------ #
    #  Neighborhood Behavioral Sampling helpers                           #
    # ------------------------------------------------------------------ #

    def _generate_neighbors(self, params_arr, n):
        """
        Generate n parameter vectors near params_arr using Poisson perturbation.
        Same logic as linear-policy agent but applied to flat Q-table param arrays.
        """
        rng = np.random.default_rng()
        neighbors = []
        for _ in range(n):
            steps = rng.poisson(self.poisson_lam, size=len(params_arr))
            signs = rng.choice([-1, 1], size=len(params_arr))
            delta = steps * signs * self.neighbor_step
            neighbor = np.round(np.clip(params_arr + delta, -6.0, 6.0), 1)
            neighbors.append(neighbor)
        return neighbors

    def _rollout_neighbors(self, world, anchor_params, label, logdir, env_description):
        """
        Roll out the anchor Q-table and n Poisson-perturbed neighbors.
        Rollouts sequential; VLM calls parallel via ThreadPoolExecutor.
        Restores original Q-table after all rollouts.

        Returns:
            anchor_result: dict {params_str, reward, analysis}
            neighbor_results: list of dicts sorted by reward descending
        """
        saved_mapping = dict(self.q_table.mapping)
        all_params = [anchor_params] + self._generate_neighbors(anchor_params, self.n_neighbors)
        rollout_data = []

        # --- Sequential rollouts ----------------------------------------
        for idx, params in enumerate(all_params):
            self.q_table.update_policy(params)
            log_file = f"{logdir}/nb_{label}_{idx}.txt"
            with open(log_file, "w") as f:
                reward, traj, term_early = self.rollout_episode_with_frames(
                    world, f, record=False, capture_frames=True
                )
            params_str = ", ".join(f"params[{j}]: {v:.5g}" for j, v in enumerate(params))
            rollout_data.append({
                "params": params,
                "params_str": params_str,
                "reward": reward,
                "trajectory": traj,
                "terminated_early": term_early,
            })

        # Restore original Q-table immediately after rollouts
        self.q_table.mapping = saved_mapping

        # --- Parallel VLM analysis --------------------------------------
        def _analyze(entry, idx):
            frame_indices = self.frame_sampler.sample_frames(
                entry["trajectory"], entry["terminated_early"], self.max_traj_length
            )
            frames = self.frame_sampler.get_frames(entry["trajectory"], frame_indices)
            if not frames or not any(f.get("frame") is not None for f in frames):
                return idx, None, 0.0, 0, 0
            analysis, vlm_time, vlm_pt, vlm_ct = self.vlm_analyzer.analyze_frames(
                frames,
                env_description,
                entry["reward"],
                entry["terminated_early"],
                current_params=entry["params_str"],
            )
            return idx, analysis, vlm_time, vlm_pt, vlm_ct

        analyses = [None] * len(rollout_data)
        total_vlm_time = 0.0
        total_vlm_pt = 0
        total_vlm_ct = 0
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_analyze, entry, idx): idx
                for idx, entry in enumerate(rollout_data)
            }
            for future in as_completed(futures):
                idx, analysis, vlm_time, vlm_pt, vlm_ct = future.result()
                analyses[idx] = analysis
                total_vlm_time += vlm_time
                total_vlm_pt += vlm_pt
                total_vlm_ct += vlm_ct

        self.vlm_api_time += total_vlm_time
        self.api_call_time += total_vlm_time
        self.total_vlm_prompt_tokens += total_vlm_pt
        self.total_vlm_completion_tokens += total_vlm_ct

        results = []
        for i, entry in enumerate(rollout_data):
            results.append({
                "params_str": entry["params_str"],
                "reward": entry["reward"],
                "analysis": analyses[i],
            })

        anchor_result = results[0]
        neighbor_results = sorted(results[1:], key=lambda x: x["reward"], reverse=True)
        return anchor_result, neighbor_results

    def _rollout_params_with_frames(self, world: BaseWorld, params: np.ndarray):
        """
        Temporarily apply a param vector to the Q-table, rollout with frame capture,
        then restore the original Q-table state.

        Returns:
            Tuple of (frames, episode_reward, terminated_early)
        """
        # Save current mapping
        saved_mapping = dict(self.q_table.mapping)

        # Apply candidate params
        self.q_table.update_policy(params)

        # Rollout with frame capture (write log to throwaway buffer)
        log_buf = io.StringIO()
        episode_reward, trajectory, terminated_early = self.rollout_episode_with_frames(
            world, log_buf, record=False, capture_frames=True
        )

        # Sample frames
        frame_indices = self.frame_sampler.sample_frames(
            trajectory, terminated_early, self.max_traj_length
        )
        frames = self.frame_sampler.get_frames(trajectory, frame_indices)

        # Restore original Q-table
        self.q_table.mapping = saved_mapping

        return frames, episode_reward, terminated_early

    def train_policy(self, world: BaseWorld, logdir):
        """
        Train Q-table policy with optional vision-guided feedback.
        
        Implements BMPS for Q-learning.
        """
        def parse_parameters(input_text):
            # Parse Q-values from LLM response
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)
            
            results = []
            for match in matches:
                results.append(float(match[1]))
            print(results)
            assert len(results) == self.rank
            return np.array(results).reshape((self.rank,))

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):
            """Format Q-table examples for prompt."""
            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text

        # ===== STEP 1: Determine if VLM should be invoked =====
        visual_analysis = None
        neighborhood_analysis = None
        lambda_t = 0.0
        
        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            should_invoke_vlm = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                np.random.RandomState(self.training_episodes)
            )
            
            print(f"\n[BMPS] λ_t = {lambda_t:.3f}, VLM Invocation: {should_invoke_vlm}")
            
            # ===== STEP 2: Neighborhood Behavioral Sampling + VLM =====
            if should_invoke_vlm:
                # Read env description once (shared across all VLM calls)
                if self.env_desc_file:
                    with open(f"agent/policy/templates/{self.env_desc_file}", "r") as f:
                        env_description = f.read()
                else:
                    env_description = "Q-learning environment"

                saved_params = np.array(
                    [self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]
                )
                print(f"[Neighborhood] Running {self.n_neighbors} neighbors per anchor "
                      f"(Poisson λ={self.poisson_lam}, step={self.neighbor_step})...")

                # --- CURRENT anchor ---
                print("[Neighborhood] Anchor: CURRENT")
                cur_anchor, cur_neighbors = self._rollout_neighbors(
                    world, saved_params, "current", logdir, env_description
                )
                visual_analysis = cur_anchor["analysis"]

                if visual_analysis is not None:
                    self.visual_analysis_history.append({
                        'iteration': self.training_episodes,
                        'lambda_t': lambda_t,
                        'num_frames': self.frame_sampler.sample_period,
                        'analysis': visual_analysis,
                        'reward': cur_anchor["reward"],
                    })
                    with open(f"{logdir}/vlm_analysis.txt", "w", encoding="utf-8") as vf:
                        vf.write(visual_analysis)
                    print(f"[VLM] CURRENT anchor analysis done ({len(visual_analysis)} chars)")

                # --- BEST and WORST anchors from replay buffer ---
                best_nb_anchor = None
                best_nb_neighbors = []
                worst_nb_anchor = None
                worst_nb_neighbors = []

                if len(self.replay_buffer.buffer) > 0:
                    rb = self.replay_buffer.buffer
                    best_rb_params, best_rb_reward = max(rb, key=lambda x: x[1])
                    worst_rb_params, worst_rb_reward = min(rb, key=lambda x: x[1])

                    if self.ablate_anchor != "best":
                        print(f"[Neighborhood] Anchor: BEST (reward={best_rb_reward:.2f})")
                        best_nb_anchor, best_nb_neighbors = self._rollout_neighbors(
                            world, np.array(best_rb_params).reshape(-1), "best", logdir, env_description
                        )

                    if self.ablate_anchor != "worst" and worst_rb_reward != best_rb_reward:
                        print(f"[Neighborhood] Anchor: WORST (reward={worst_rb_reward:.2f})")
                        worst_nb_anchor, worst_nb_neighbors = self._rollout_neighbors(
                            world, np.array(worst_rb_params).reshape(-1), "worst", logdir, env_description
                        )

                # Restore original Q-table
                self.q_table.mapping = {i: saved_params[i] for i in range(len(saved_params))}

                # ===== STEP 3: Build neighborhood analysis summary =====
                def _anchor_block(label, anchor_res, neighbors, top_k=2):
                    lines = [f"### Anchor [{label}]  reward={anchor_res['reward']:.2f}"]
                    lines.append(f"  Params : {anchor_res['params_str']}")
                    lines.append(f"  Behavior: {anchor_res['analysis'] if anchor_res['analysis'] else '(no VLM analysis available)'}")
                    if neighbors:
                        top_neighbors = neighbors[:top_k]
                        bottom_neighbors = neighbors[-top_k:] if len(neighbors) > top_k else []
                        lines.append(f"\n  ---- Top-{top_k} highest-reward neighbors of [{label}] ----")
                        for rank_i, nb in enumerate(top_neighbors, start=1):
                            lines.append(f"  Neighbor #{rank_i} (highest)  reward={nb['reward']:.2f}")
                            lines.append(f"    Params : {nb['params_str']}")
                            lines.append(f"    Behavior: {nb['analysis'] if nb['analysis'] else '(no VLM analysis available)'}")
                        if bottom_neighbors:
                            lines.append(f"\n  ---- Bottom-{top_k} lowest-reward neighbors of [{label}] ----")
                            for rank_i, nb in enumerate(bottom_neighbors, start=1):
                                lines.append(f"  Neighbor #{rank_i} (lowest)  reward={nb['reward']:.2f}")
                                lines.append(f"    Params : {nb['params_str']}")
                                lines.append(f"    Behavior: {nb['analysis'] if nb['analysis'] else '(no VLM analysis available)'}")
                    return "\n".join(lines)

                blocks = ["## Neighborhood Behavioral Landscape\n"]
                if self.ablate_anchor != "current":
                    blocks.append(_anchor_block("CURRENT", cur_anchor, cur_neighbors))
                if self.ablate_anchor != "best" and best_nb_anchor is not None:
                    blocks.append(_anchor_block("BEST (replay buffer)", best_nb_anchor, best_nb_neighbors))
                if self.ablate_anchor != "worst" and worst_nb_anchor is not None:
                    blocks.append(_anchor_block("WORST (replay buffer)", worst_nb_anchor, worst_nb_neighbors))
                neighborhood_analysis = "\n\n".join(blocks)

                with open(f"{logdir}/neighborhood_analysis.txt", "w", encoding="utf-8") as nf:
                    nf.write(neighborhood_analysis)
                print(f"[Neighborhood] Landscape summary saved ({len(neighborhood_analysis)} chars)")
        
        # ===== STEP 4: Update Q-table using LLM with vision context =====
        print("\nUpdating Q-table policy with LLM...")
        
        new_parameter_list, reasoning, api_time, llm_prompt_tokens, llm_completion_tokens = self.llm_brain.llm_update_parameters_num_optim_vision(
            str_nd_examples(self.replay_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file if self.env_desc_file else "env_descriptions/default.j2",
            rank=self.rank,
            optimum=self.optimum,
            actions=self.actions,
            neighborhood_analysis=neighborhood_analysis,
        )
        
        self.api_call_time += api_time
        self.total_llm_prompt_tokens += llm_prompt_tokens
        self.total_llm_completion_tokens += llm_completion_tokens

        # Update Q-table
        print(f"Old Q-table size: {len(self.q_table.mapping)}")
        print(f"New params shape: {new_parameter_list.shape}")
        self.q_table.update_policy(new_parameter_list)
        print(f"Updated Q-table size: {len(self.q_table.mapping)}")
        
        # Log Q-table
        logging_q_filename = f"{logdir}/parameters.txt"
        with open(logging_q_filename, "w") as logging_q_file:
            logging_q_file.write(str(self.q_table.mapping))
        
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        with open(q_reasoning_filename, "w", encoding="utf-8") as q_reasoning_file:
            q_reasoning_file.write(reasoning)
        
        print("Q-table policy updated!")

        # ===== STEP 5: Evaluate new Q-table =====
        print(f"Evaluating Q-table (episode {self.training_episodes})...")
        logging_filename = f"{logdir}/training_rollout_final.txt"
        with open(logging_filename, "w") as logging_file:
            results = []
            for idx in range(self.num_evaluation_episodes):
                result = self.rollout_episode(world, logging_file, record=False)
                results.append(result)
            print(f"Results: {results}")
            
            result = np.mean(results)
            self.replay_buffer.add(
                np.array([self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]),
                result,
            )

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        
        iter_vlm_pt = self.total_vlm_prompt_tokens
        iter_vlm_ct = self.total_vlm_completion_tokens
        
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, llm_prompt_tokens, llm_completion_tokens, iter_vlm_pt, iter_vlm_ct
    
    def evaluate_policy(self, world: BaseWorld, logdir):
        """Evaluate current Q-table policy."""
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            with open(logging_filename, "w") as logging_file:
                result = self.rollout_episode(world, logging_file, record=False)
                results.append(result)
        return results
