"""
ProPS-V: Vision-Guided Prompted Policy Search

This module implements the ProPS-V agent that combines:
1. Numerical optimization (from ProPS)
2. Semantic reasoning (from ProPS+)  
3. Vision-guided feedback (new in ProPS-V)

Key components:
- Periodic frame sampling
- Adaptive visual guidance annealing (Eq. 3)
- VLM-based visual analysis
- Integrated update rule (Eq. 4)
"""

from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias, ReplayBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from agent.policy.frame_sampler import FrameSampler
from agent.policy.adaptive_visual_guidance import AdaptiveVisualGuidance
from agent.policy.vlm_analyzer import VLMAnalyzer
from world.base_world import BaseWorld
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import time
import os


class LLMNumOptimVisionAgent:
    """
    ProPS-V Agent: Vision-Guided Prompted Policy Search
    
    Implements the full ProPS-V algorithm with:
    - Periodic frame sampling
    - Adaptive visual guidance schedule  
    - VLM analysis integration
    """
    
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        env_desc_file=None,
        vlm_model_name="gpt-4o",
        decay_horizon=100,
        frame_sample_period=50,
        enable_vision=True,
        n_neighbors=3,
        poisson_lam=2.0,
        neighbor_step=0.1,
    ):
        """
        Initialize ProPS-V agent.
        
        Args:
            logdir: Directory for logging
            dim_action: Dimension of action space
            dim_state: Dimension of state space
            max_traj_count: Maximum number of trajectories to store
            max_traj_length: Maximum trajectory length
            llm_si_template: Jinja2 template for system prompt
            llm_output_conversion_template: Template for output formatting
            llm_model_name: Name of LLM model for policy optimization
            num_evaluation_episodes: Number of episodes for evaluation
            bias: Whether to use bias in linear policy
            optimum: Expected optimal reward
            search_step_size: Step size for exploration
            env_desc_file: Environment description text (semantic info)
            vlm_model_name: Name of VLM model for visual analysis
            decay_horizon: T_decay for visual guidance annealing
            frame_sample_period: P — capture a VLM frame every P timesteps
            enable_vision: Whether to enable vision-guided feedback
            n_neighbors: Number of Poisson-perturbed neighbors per anchor
            poisson_lam: Lambda (mean) of Poisson distribution for step sizes
            neighbor_step: Base step size multiplied by Poisson sample
        """
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.vlm_api_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        
        # Environment and policy parameters
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.env_desc_file = env_desc_file
        self.max_traj_length = max_traj_length
        self.enable_vision = enable_vision
        self.n_neighbors = n_neighbors
        self.poisson_lam = poisson_lam
        self.neighbor_step = neighbor_step
        
        # Compute parameter count
        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count
        
        # Initialize policy
        if not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(
                dim_actions=dim_action, dim_states=dim_state
            )
        
        # Initialize replay buffers
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.visual_analysis_history = []  # Store ψ_1, ..., ψ_N
        
        # Initialize LLM brain
        self.llm_brain = LLMBrain(
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name
        )
        
        # Initialize vision components
        if self.enable_vision:
            self.frame_sampler = FrameSampler(
                sample_period=frame_sample_period
            )
            self.visual_guidance = AdaptiveVisualGuidance(
                decay_horizon=decay_horizon
            )
            self.vlm_analyzer = VLMAnalyzer(
                vlm_model_name=vlm_model_name
            )
        
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        
        # For bias, add extra dimension to state
        if self.bias:
            self.dim_state += 1
    
    # ------------------------------------------------------------------ #
    #  Neighborhood Behavioral Sampling helpers                           #
    # ------------------------------------------------------------------ #

    def _generate_neighbors(self, params_arr, n):
        """
        Generate n parameter vectors near params_arr using Poisson perturbation.

        For each dimension d:
            steps  ~ Poisson(poisson_lam)          # non-negative integer steps
            sign   ~ Uniform({-1, +1})
            delta  = steps * sign * neighbor_step  # keeps values on 1-dp grid
            new[d] = clip(round(params_arr[d] + delta, 1), -6.0, 6.0)

        Args:
            params_arr: 1-D numpy array of current parameter values
            n: Number of neighbors to generate

        Returns:
            List of n numpy arrays, each perturbed from params_arr
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

    def _rollout_neighbors(self, world, anchor_params, label, logdir):
        """
        Roll out the anchor policy and n Poisson-perturbed neighbors.

        Rollouts are performed **sequentially** (env is not thread-safe).
        VLM analysis calls are dispatched **in parallel** via ThreadPoolExecutor.

        Args:
            world: RL environment
            anchor_params: 1-D numpy array for the anchor policy
            label: String label used in log-file names (e.g. "current", "best")
            logdir: Directory for rollout logs

        Returns:
            anchor_result: dict {params_str, reward, analysis} for the anchor itself
            neighbor_results: list of dicts {params_str, reward, analysis} for the
                              n perturbed neighbors, sorted by reward descending
        """
        # Build the full candidate list: anchor first, then n neighbors
        all_params = [anchor_params] + self._generate_neighbors(anchor_params, self.n_neighbors)
        rollout_data = []

        # --- Sequential rollouts ----------------------------------------
        for idx, params in enumerate(all_params):
            self.policy.update_policy(params)
            log_file = f"{logdir}/nb_{label}_{idx}.txt"
            with open(log_file, "w") as f:
                reward, traj, term_early = self.rollout_episode(
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

        # --- Parallel VLM analysis --------------------------------------
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
        total_vlm_time = 0.0
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_analyze, entry, idx): idx
                for idx, entry in enumerate(rollout_data)
            }
            for future in as_completed(futures):
                idx, analysis, vlm_time = future.result()
                analyses[idx] = analysis
                total_vlm_time += vlm_time

        self.vlm_api_time += total_vlm_time
        self.api_call_time += total_vlm_time

        # Assemble results
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

    def rollout_episode(
        self, 
        world: BaseWorld, 
        logging_file, 
        record=True,
        capture_frames=False
    ):
        """
        Rollout one episode and optionally capture frames for visual analysis.
        
        Args:
            world: Environment to interact with
            logging_file: File handle for logging
            record: Whether to record in replay buffer
            capture_frames: Whether to capture rendered frames
            
        Returns:
            Total episodic reward
        """
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        
        # Log parameters
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        
        done = False
        step_idx = 0
        episode_num = 1
        trajectory = []
        first_episode_done = False
        first_episode_reward = None
        first_episode_steps = None
        
        if record:
            self.traj_buffer.start_new_trajectory()
        
        while True:
            # Get action from policy
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            
            # Step environment
            next_state, reward, done = world.step(action)
            
            # Log
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            
            # Capture frame only at sampled timesteps to avoid rendering every step
            frame = None
            if capture_frames and hasattr(world.env, 'render'):
                is_sampled_step = (
                    step_idx == 0
                    or step_idx % self.frame_sampler.sample_period == 0
                    or done
                    or step_idx == self.max_traj_length - 1
                )
                if is_sampled_step:
                    try:
                        frame = world.env.render()
                        if not isinstance(frame, np.ndarray):
                            frame = None
                    except:
                        frame = None
            
            # Store trajectory step
            if record or capture_frames:
                trajectory.append({
                    'state': state.T[0].copy(),
                    'action': action[0].copy(),
                    'reward': reward,
                    'frame': frame,
                    'episode_num': episode_num,
                })
            
            # Add to replay buffer
            if record:
                self.traj_buffer.add_step(state, action, reward)
            
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
                    state = np.expand_dims(state, axis=0)
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
            terminated_early = first_episode_steps < self.max_traj_length
        else:
            total_reward = world.get_accu_reward()
            terminated_early = step_idx < self.max_traj_length
        
        # Log total reward
        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        
        # Return trajectory info if capturing frames
        if capture_frames:
            return total_reward, trajectory, terminated_early
        else:
            return total_reward
    
    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        """
        Perform random warmup episodes to initialize replay buffer.
        
        Args:
            world: Environment
            logdir: Directory for warmup logs
            num_episodes: Number of warmup episodes
        """
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            print(f"Rolling out warmup episode {episode}...")
            
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            
            result = self.rollout_episode(world, logging_file, record=True, capture_frames=False)
            
            self.replay_buffer.add(
                np.array(self.policy.get_parameters()).reshape(-1),
                world.get_accu_reward()
            )
            
            logging_file.close()
            print(f"Result: {result}")
    
    def train_policy(self, world: BaseWorld, logdir):
        """
        Train policy for one iteration using ProPS-V.
        
        Implements:
        - Periodic frame sampling
        - Adaptive visual guidance (Eq. 3)
        - Vision-guided parameter update (Eq. 4)
        
        Args:
            world: Environment
            logdir: Directory for episode logs
            
        Returns:
            Tuple of (cpu_time, api_time, total_episodes, total_steps, total_reward)
        """
        
        def parse_parameters(input_text):
            """Parse parameters from LLM output."""
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)
            
            results = []
            for match in matches:
                results.append(float(match[1]))
            print(results)
            assert len(results) == self.rank, f"Expected {self.rank} params, got {len(results)}"
            return np.array(results).reshape(-1)
        
        def str_nd_examples(replay_buffer, traj_buffer, n):
            """Format numerical examples for prompt."""
            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))
            
            text = ""
            for idx, (parameters, reward) in enumerate(all_parameters):
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text

        # ===== STEP 1: Determine lambda and whether VLM will run =====
        visual_analysis = None
        lambda_t = 0.0
        use_vision_this_iter = False

        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            use_vision_this_iter = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                random_state=np.random.RandomState(self.training_episodes)
            )
            print(f"ProPS-V Iteration {self.training_episodes}")
            print(f"λ_t = {lambda_t:.3f}")
            print(f"VLM invocation: {use_vision_this_iter}")

        # Get best visual analysis from history BEFORE running VLM this iter
        best_visual_analysis = None
        best_visual_entry = None
        if self.visual_analysis_history:
            best_visual_entry = max(self.visual_analysis_history, key=lambda x: x['reward'])
            best_visual_analysis = best_visual_entry['analysis']
            print(f"[VLM] Best history: iter {best_visual_entry['iteration']} reward={best_visual_entry['reward']:.2f}")

        # ===== STEP 2: Neighborhood Behavioral Sampling + VLM ======
        # For each anchor (CURRENT, BEST, WORST) generate Poisson-perturbed
        # neighbors, roll them out sequentially, then analyse all frames in
        # parallel via the VLM.  Results feed a local behavioral landscape
        # summary that is appended to the LLM prompt.
        neighborhood_analysis = None
        if use_vision_this_iter:
            saved_params = self.policy.get_parameters().reshape(-1).copy()
            print(f"[Neighborhood] Running {self.n_neighbors} neighbors per anchor "
                  f"(Poisson λ={self.poisson_lam}, step={self.neighbor_step})...")

            # --- CURRENT anchor ---
            print("[Neighborhood] Anchor: CURRENT")
            cur_anchor, cur_neighbors = self._rollout_neighbors(
                world, saved_params, "current", logdir
            )
            visual_analysis = cur_anchor["analysis"]   # keeps existing ProPS-V path

            # Record current anchor in visual_analysis_history (matching original format)
            if visual_analysis is not None:
                self.visual_analysis_history.append({
                    'iteration': self.training_episodes,
                    'lambda': lambda_t,
                    'reward': cur_anchor["reward"],
                    'analysis': visual_analysis,
                    'num_frames': self.frame_sampler.sample_period,
                    'params': cur_anchor["params_str"],
                })
                visual_log_file = f"{logdir}/vlm_analysis.txt"
                with open(visual_log_file, "w", encoding="utf-8") as vf:
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

                print(f"[Neighborhood] Anchor: BEST (reward={best_rb_reward:.2f})")
                best_nb_anchor, best_nb_neighbors = self._rollout_neighbors(
                    world, np.array(best_rb_params).reshape(-1), "best", logdir
                )

                # Only run WORST if it differs meaningfully from BEST
                if worst_rb_reward != best_rb_reward:
                    print(f"[Neighborhood] Anchor: WORST (reward={worst_rb_reward:.2f})")
                    worst_nb_anchor, worst_nb_neighbors = self._rollout_neighbors(
                        world, np.array(worst_rb_params).reshape(-1), "worst", logdir
                    )

            # Restore original policy so STEP 4 sees the right current params
            self.policy.update_policy(saved_params)

            # ===== STEP 3: Build neighborhood analysis summary for LLM =====
            def _anchor_block(label, anchor_res, neighbors):
                """Format a short behavioral-landscape block for one anchor."""
                lines = [f"### Anchor [{label}]  reward={anchor_res['reward']:.2f}"]
                lines.append(f"  Params : {anchor_res['params_str']}")
                if anchor_res["analysis"]:
                    lines.append(f"  Behavior: {anchor_res['analysis']}")
                else:
                    lines.append(f"  Behavior: (no VLM analysis available)")

                if neighbors:
                    lines.append(f"\n  ---- Neighbors of [{label}] ----")
                    for rank_i, nb in enumerate(neighbors, start=1):
                        lines.append(f"  Neighbor #{rank_i}  reward={nb['reward']:.2f}")
                        lines.append(f"    Params : {nb['params_str']}")
                        if nb["analysis"]:
                            lines.append(f"    Behavior: {nb['analysis']}")
                        else:
                            lines.append(f"    Behavior: (no VLM analysis available)")
                return "\n".join(lines)

            blocks = ["## Neighborhood Behavioral Landscape\n"]
            blocks.append(_anchor_block("CURRENT", cur_anchor, cur_neighbors))
            if best_nb_anchor is not None:
                blocks.append(_anchor_block("BEST (replay buffer)", best_nb_anchor, best_nb_neighbors))
            if worst_nb_anchor is not None:
                blocks.append(_anchor_block("WORST (replay buffer)", worst_nb_anchor, worst_nb_neighbors))
            neighborhood_analysis = "\n\n".join(blocks)

            nb_log_file = f"{logdir}/neighborhood_analysis.txt"
            with open(nb_log_file, "w", encoding="utf-8") as nf:
                nf.write(neighborhood_analysis)
            print(f"[Neighborhood] Landscape summary saved ({len(neighborhood_analysis)} chars)")

        # ===== STEP 4: Update policy using LLM with vision context =====
        print("\nUpdating policy with LLM...")
        current_params_for_llm = self.policy.get_parameters().reshape(-1)
        params_str_for_llm = ", ".join(f"params[{i}]: {v:.5g}" for i, v in enumerate(current_params_for_llm))
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_vision(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            neighborhood_analysis=neighborhood_analysis,
        )
        self.api_call_time += api_time

        self.policy.update_policy(new_parameter_list)

        # Log parameters and reasoning (reasoning now contains visual_analysis in system prompt)
        logging_q_filename = f"{logdir}/parameters.txt"
        with open(logging_q_filename, "w") as logging_q_file:
            logging_q_file.write(str(self.policy))
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        with open(q_reasoning_filename, "w", encoding="utf-8") as q_reasoning_file:
            q_reasoning_file.write(reasoning)
        print("Policy updated!")

        # ===== STEP 5: Evaluate new policy =====
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        results = []
        with open(logging_filename, "w") as logging_file:
            for idx in range(self.num_evaluation_episodes):
                result = self.rollout_episode(
                    world, logging_file,
                    record=(idx == 0),
                    capture_frames=False
                )
                results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)

        # ===== STEP 6: Add to replay buffer =====
        self.replay_buffer.add(new_parameter_list, result)
        
        # Increment iteration counter
        self.training_episodes += 1
        if self.enable_vision:
            self.visual_guidance.increment_iteration()
        
        # Return statistics
        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time + self.vlm_api_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward
    
    def evaluate_policy(self, world: BaseWorld, logdir):
        """
        Evaluate the current policy.
        
        Args:
            world: Environment
            logdir: Directory for evaluation logs
            
        Returns:
            List of episode rewards
        """
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False, capture_frames=False)
            results.append(result)
            logging_file.close()
        return results
