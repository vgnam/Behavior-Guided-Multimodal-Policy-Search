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
        trajectory = []
        
        if record:
            self.traj_buffer.start_new_trajectory()
        
        while not done:
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
            
            # Capture frame if requested
            frame = None
            if capture_frames and hasattr(world.env, 'render'):
                try:
                    frame = world.env.render()
                    # Ensure frame is numpy array
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
                    'frame': frame
                })
            
            # Add to replay buffer
            if record:
                self.traj_buffer.add_step(state, action, reward)
            
            state = next_state
            step_idx += 1
            self.total_steps += 1
        
        # Log total reward
        total_reward = world.get_accu_reward()
        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        
        # Check if terminated early (failure)
        terminated_early = step_idx < self.max_traj_length
        
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

        # ===== STEP 2: Rollout current policy with frame capture for VLM ======
        if use_vision_this_iter:
            print("Running episode with frame capture for VLM analysis...")
            pre_log_filename = f"{logdir}/training_rollout_pre.txt"
            with open(pre_log_filename, "w") as pre_log:
                pre_result, trajectory, terminated_early = self.rollout_episode(
                    world, pre_log, record=False, capture_frames=True
                )

            # ===== STEP 3: Sample frames and run VLM =====
            frame_indices = self.frame_sampler.sample_frames(
                trajectory, terminated_early, self.max_traj_length
            )
            frames = self.frame_sampler.get_frames(trajectory, frame_indices)

            if len(frames) > 0 and any(f.get('frame') is not None for f in frames):
                current_params = self.policy.get_parameters().reshape(-1)
                params_str = ", ".join(f"params[{i}]: {v:.5g}" for i, v in enumerate(current_params))
                visual_analysis, vlm_time = self.vlm_analyzer.analyze_frames(
                    frames,
                    self.env_desc_file if self.env_desc_file else "RL Environment",
                    pre_result,
                    terminated_early,
                    current_params=params_str,
                )
                self.vlm_api_time += vlm_time
                self.api_call_time += vlm_time

                self.visual_analysis_history.append({
                    'iteration': self.training_episodes,
                    'lambda': lambda_t,
                    'reward': pre_result,
                    'analysis': visual_analysis,
                    'num_frames': len(frames),
                    'params': params_str,
                })

                visual_log_file = f"{logdir}/vlm_analysis.txt"
                with open(visual_log_file, "w", encoding="utf-8") as vf:
                    vf.write(visual_analysis)

                print(f"[VLM] Analysis complete ({len(visual_analysis)} chars)")
            else:
                print("No frames captured for visual analysis")

        # ===== STEP 4: Update policy using LLM with vision context =====
        print("\nUpdating policy with LLM...")
        current_params_for_llm = self.policy.get_parameters().reshape(-1)
        params_str_for_llm = ", ".join(f"params[{i}]: {v:.5g}" for i, v in enumerate(current_params_for_llm))
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_vision(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            visual_analysis,
            lambda_t,
            self.rank,
            self.optimum,
            self.search_step_size,
            visual_params=params_str_for_llm if visual_analysis else None,
            best_visual_analysis=best_visual_analysis,
            best_visual_entry=best_visual_entry,
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
