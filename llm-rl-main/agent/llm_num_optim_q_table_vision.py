"""
ProPS-V Q-Table Agent: Vision-Guided Q-Learning

This module implements ProPS-V for discrete state spaces (Q-tables).
Key components:
- Q-table policy for discrete state-action spaces
- Critical frame sampling
- Adaptive visual guidance annealing
- VLM-based visual analysis
"""

from agent.policy.q_table import QTable
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from agent.policy.adaptive_visual_guidance import AdaptiveVisualGuidance
from agent.policy.vlm_analyzer import VLMAnalyzer
from world.base_world import BaseWorld
import traceback
import numpy as np
import re
import time


class LLMNumOptimQTableVisionAgent:
    """
    ProPS-V Agent for Q-table learning with vision-guided optimization.
    
    Extends standard Q-learning with:
    - Critical frame sampling
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
        vlm_model_name="gpt-4o",
        decay_horizon=100,
        enable_vision=True,
        vlm_enable_reasoning=False,
        env_kwargs=None,
    ):
        """
        Initialize ProPS-V Q-Table agent.
        
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
            enable_vision: Whether to enable vision-guided features
            vlm_enable_reasoning: If True, prepend chain-of-thought instruction to VLM prompt
            env_kwargs: Additional environment kwargs
        """
        self.start_time = time.process_time()
        self.api_call_time = 0
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
        self.visual_analysis_history = []  # Store visual analyses
        
        # Q-table policy
        self.q_table = QTable(actions=actions, states=states)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        
        # LLM brain
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0
        self.rank = len(self.q_table.mapping)
        
        # Initialize vision components if enabled
        if self.enable_vision:
            self.visual_guidance = AdaptiveVisualGuidance(
                decay_horizon=decay_horizon
            )
            self.vlm_analyzer = VLMAnalyzer(
                vlm_model_name=vlm_model_name,
                timeout=120,  # 2 minutes timeout for VLM API calls
                enable_reasoning=vlm_enable_reasoning,
            )
            print(f"[ProPS-V Q-Table] Vision features enabled (VLM: {vlm_model_name}, T_decay: {decay_horizon})")
        else:
            print("[ProPS-V Q-Table] Vision features disabled - using numerical optimization only")

    def rollout_episode_with_frames(
        self,
        world: BaseWorld,
        logging_file,
        record=True,
        capture_frames=False
    ):
        """
        Rollout one episode, optionally capturing only the last frame for VLM analysis.

        Returns:
            (total_reward, last_frame_data, terminated_early)  when capture_frames=True
            (total_reward, None, terminated_early)              when capture_frames=False
        """
        state = world.reset()
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        last_frame_data = None

        while not done:
            action = self.q_table.get_action(state)
            action = int(np.reshape(action, (1,)))

            next_state, reward, done = world.step(action)
            logging_file.write(f"{state} | {action} | {reward}\n")

            # Overwrite last frame at every step (only keep the final one)
            if capture_frames and hasattr(world.env, 'render'):
                try:
                    frame = world.env.render()
                    if isinstance(frame, np.ndarray):
                        last_frame_data = {
                            'frame': frame,
                            'state': state,
                            'action': action,
                            'reward': reward,
                            'timestep': step_idx,
                        }
                except:
                    pass

            state = next_state
            step_idx += 1
            self.total_steps += 1

            if step_idx >= self.max_traj_length:
                break

        total_reward = world.get_accu_reward()
        terminated_early = (step_idx < self.max_traj_length)

        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1

        if record:
            self.replay_buffer.add(
                np.array([self.q_table.mapping[i] for i in range(len(self.q_table.mapping))]),
                total_reward,
            )

        return total_reward, last_frame_data, terminated_early

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
            logging_file = open(logging_filename, "w", encoding="utf-8")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def train_policy(self, world: BaseWorld, logdir):
        """
        Train Q-table policy with optional vision-guided feedback.
        
        Implements ProPS-V for Q-learning.
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
        lambda_t = 0.0
        
        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            should_invoke_vlm = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                np.random.RandomState(self.training_episodes)
            )
            
            print(f"\n[ProPS-V] lambda_t = {lambda_t:.3f}, VLM Invocation: {should_invoke_vlm}")
            
            # ===== STEP 2: Run episode, capture last frame if VLM needed =====
            if should_invoke_vlm:
                logging_filename = f"{logdir}/training_rollout.txt"
                with open(logging_filename, "w", encoding="utf-8") as logging_file:
                    episode_reward, last_frame_data, terminated_early = self.rollout_episode_with_frames(
                        world, logging_file, record=False, capture_frames=True
                    )
                
                # ===== STEP 3: Analyze single frame with VLM =====
                if last_frame_data is not None and last_frame_data.get('frame') is not None:
                    if self.env_desc_file:
                        with open(f"agent/policy/templates/{self.env_desc_file}", "r", encoding="utf-8") as f:
                            env_description = f.read()
                    else:
                        env_description = "Q-learning environment"
                    
                    visual_analysis, vlm_api_time = self.vlm_analyzer.analyze_single_frame(
                        last_frame_data['frame'],
                        env_description,
                        episode_reward,
                        terminated_early,
                        timestep=last_frame_data.get('timestep', -1),
                        state=last_frame_data.get('state'),
                        action=last_frame_data.get('action'),
                    )
                    self.api_call_time += vlm_api_time
                    
                    self.visual_analysis_history.append({
                        'iteration': self.training_episodes,
                        'lambda_t': lambda_t,
                        'analysis': visual_analysis,
                        'reward': episode_reward
                    })
                    
                    visual_log_file = f"{logdir}/vlm_output.txt"
                    with open(visual_log_file, "a", encoding="utf-8") as vf:
                        vf.write(f"\n{'='*60}\n")
                        vf.write(f"Iteration {self.training_episodes} (lambda={lambda_t:.3f}, reward={episode_reward:.2f})\n")
                        vf.write(f"Timestep: {last_frame_data.get('timestep', -1)}, Terminated early: {terminated_early}\n")
                        vf.write(f"{'='*60}\n")
                        vf.write(visual_analysis + "\n")
                    print(f"VLM analysis saved to {visual_log_file}")
        
        # ===== STEP 4: Update Q-table using LLM with vision context =====
        print("\nUpdating Q-table policy with LLM...")
        
        if self.enable_vision:
            # Use vision-specific method
            new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_q_table_vision(
                str_nd_examples(self.replay_buffer, self.rank),
                parse_parameters,
                self.training_episodes,
                self.env_desc_file if self.env_desc_file else "env_descriptions/default.j2",
                visual_analysis,
                lambda_t,
                self.visual_guidance.get_prompt_instruction(self.training_episodes),
                self.actions,
                self.rank,
                self.optimum
            )
        else:
            # Use standard numerical optimization
            new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
                str_nd_examples(self.replay_buffer, self.rank),
                parse_parameters,
                self.training_episodes,
                self.rank,
                self.optimum,
                actions=self.actions,
            )
        
        self.api_call_time += api_time

        # Update Q-table
        print(f"Old Q-table size: {len(self.q_table.mapping)}")
        print(f"New params shape: {new_parameter_list.shape}")
        self.q_table.update_policy(new_parameter_list)
        print(f"Updated Q-table size: {len(self.q_table.mapping)}")
        
        # Log Q-table
        logging_q_filename = f"{logdir}/parameters.txt"
        with open(logging_q_filename, "w", encoding="utf-8") as logging_q_file:
            logging_q_file.write(str(self.q_table.mapping))
        
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        with open(q_reasoning_filename, "w", encoding="utf-8") as q_reasoning_file:
            q_reasoning_file.write(reasoning)
        
        print("Q-table policy updated!")

        # ===== STEP 5: Evaluate new Q-table =====
        print(f"Evaluating Q-table (episode {self.training_episodes})...")
        logging_filename = f"{logdir}/training_rollout_final.txt"
        with open(logging_filename, "w", encoding="utf-8") as logging_file:
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
        
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward
    
    def evaluate_policy(self, world: BaseWorld, logdir):
        """Evaluate current Q-table policy."""
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            with open(logging_filename, "w", encoding="utf-8") as logging_file:
                result = self.rollout_episode(world, logging_file, record=False)
                results.append(result)
        return results
