"""
Runner for ProPS-V: Vision-Guided Prompted Policy Search

This module provides the training loop for ProPS-V agents,
integrating vision-language model feedback with policy optimization.
"""

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    bias=None,
    rank=None,
    optimum=1000,
    search_step_size=0.1,
    env_kwargs=None,
    env_desc_file=None,
    vlm_model_name="gpt-4o",
    decay_horizon=100,
    reward_change_threshold=0.1,
    enable_vision=True,
):
    """
    Run ProPS-V training loop.
    
    Args:
        task: Task type (currently supports "cont_state_llm_num_optim_vision")
        num_episodes: Number of training episodes
        gym_env_name: Name of the Gym environment
        render_mode: Rendering mode ("rgb_array" for frame capture, None otherwise)
        logdir: Directory for logs
        dim_actions: Dimension of action space
        dim_states: Dimension of state space
        max_traj_count: Maximum trajectories in replay buffer
        max_traj_length: Maximum trajectory length
        template_dir: Directory containing Jinja2 templates
        llm_si_template_name: Template name for system instructions
        llm_output_conversion_template_name: Template for output conversion
        llm_model_name: LLM model for policy optimization
        num_evaluation_episodes: Number of evaluation rollouts per iteration
        warmup_episodes: Number of random warmup episodes
        warmup_dir: Directory to load/save warmup data
        bias: Whether to use bias in linear policy
        rank: Parameter rank (for random projection, unused in vision)
        optimum: Expected optimal reward
        search_step_size: Step size for exploration
        env_kwargs: Additional environment kwargs
        env_desc_file: Path to environment description file
        vlm_model_name: Vision-Language Model for visual analysis
        decay_horizon: T_decay for visual guidance annealing (Eq. 3)
        reward_change_threshold: δ threshold for transition states (Eq. 2)
        enable_vision: Whether to enable vision-guided feedback
    """
    assert task in ["cont_state_llm_num_optim_vision"], \
        f"ProPS-V runner only supports 'cont_state_llm_num_optim_vision', got '{task}'"

    # Load Jinja2 templates
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    
    # Load environment description if provided
    if env_desc_file and os.path.exists(env_desc_file):
        with open(env_desc_file, 'r') as f:
            env_description = f.read()
    else:
        env_description = "RL Environment"
    
    # Initialize world
    # For vision, we need rgb_array rendering to capture frames
    if enable_vision and render_mode is None:
        render_mode = "rgb_array"
        print(f"[ProPS-V] Enabling rgb_array rendering for frame capture")
    
    world = ContinualSpaceGeneralWorld(
        gym_env_name,
        render_mode,
        max_traj_length,
    )
    
    # Initialize ProPS-V agent
    agent = LLMNumOptimVisionAgent(
        logdir,
        dim_actions,
        dim_states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        env_desc_file=env_description,
        vlm_model_name=vlm_model_name,
        decay_horizon=decay_horizon,
        reward_change_threshold=reward_change_threshold,
        enable_vision=enable_vision,
    )
    
    print('[ProPS-V] Initialization done')
    print(f'  LLM: {llm_model_name}')
    print(f'  VLM: {vlm_model_name}')
    print(f'  Vision Enabled: {enable_vision}')
    print(f'  Decay Horizon: {decay_horizon}')
    print(f'  Reward Change Threshold: {reward_change_threshold}')
    
    # Warmup phase
    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        print(f'\n[ProPS-V] Starting warmup with {warmup_episodes} episodes...')
        agent.random_warmup(world, warmup_dir, warmup_episodes)
        print('[ProPS-V] Warmup complete')
    else:
        print(f'[ProPS-V] Loading warmup data from {warmup_dir}')
        agent.replay_buffer.load(warmup_dir)
    
    # Training loop
    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, CPU Time, API Time (LLM+VLM), Total Episodes, Total Steps, Total Reward\n")
    overall_log_file.flush()
    
    # Create vision statistics log
    vision_stats_file = open(f"{logdir}/vision_statistics.txt", "w")
    vision_stats_file.write("Iteration, Lambda, VLM Invoked, Phase, Num Critical Frames\n")
    vision_stats_file.flush()
    
    print('\n' + '='*70)
    print(f"[ProPS-V] Starting training for {num_episodes} episodes")
    print('='*70 + '\n')
    
    for episode in range(num_episodes):
        print('\n' + '='*70)
        print(f"ProPS-V Episode: {episode}/{num_episodes}")
        print('='*70)
        
        # Create episode log directory
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        # Log current guidance statistics
        if enable_vision:
            stats = agent.visual_guidance.get_statistics()
            print(f"\nGuidance Statistics:")
            print(f"  λ_t = {stats['current_lambda']:.3f}")
            print(f"  Phase: {stats['phase']}")
            print(f"  VLM Invocation Prob: {stats['vlm_invocation_probability']:.1%}")
            print(f"  Iterations until pure numerical: {stats['iterations_until_pure_numerical']}")
        
        # Train policy (with retries)
        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = agent.train_policy(
                    world, curr_episode_dir
                )
                
                # Log overall statistics
                overall_log_file.write(
                    f"{episode}, {cpu_time:.2f}, {api_time:.2f}, {total_episodes}, {total_steps}, {total_reward:.2f}\n"
                )
                overall_log_file.flush()
                
                # Log vision statistics
                if enable_vision:
                    stats = agent.visual_guidance.get_statistics()
                    # Check if visual analysis was generated this iteration
                    vlm_invoked = (
                        len(agent.visual_analysis_history) > 0 and 
                        agent.visual_analysis_history[-1]['iteration'] == episode
                    )
                    num_frames = (
                        agent.visual_analysis_history[-1]['num_critical_frames'] 
                        if vlm_invoked else 0
                    )
                    
                    vision_stats_file.write(
                        f"{episode}, {stats['current_lambda']:.3f}, {vlm_invoked}, "
                        f"{stats['phase']}, {num_frames}\n"
                    )
                    vision_stats_file.flush()
                
                print(f"\n[SUCCESS] Trial {trial_idx + 1} succeeded")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  CPU Time: {cpu_time:.2f}s")
                print(f"  API Time: {api_time:.2f}s (LLM: {agent.api_call_time:.2f}s, VLM: {agent.vlm_api_time:.2f}s)")
                break
                
            except Exception as e:
                print(f"\n[ERROR] Trial {trial_idx + 1}/5 failed: {e}")
                traceback.print_exc()
                
                if trial_idx == 4:
                    print(f"\n[FAILURE] Episode {episode} failed after 5 attempts")
                    overall_log_file.close()
                    vision_stats_file.close()
                    return
                else:
                    continue
    
    overall_log_file.close()
    vision_stats_file.close()
    
    print('\n' + '='*70)
    print("[ProPS-V] Training Complete!")
    print('='*70)
    print(f"  Final Episode: {episode}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total API Time: {api_time:.2f}s")
    print(f"    - LLM Time: {agent.api_call_time:.2f}s")
    print(f"    - VLM Time: {agent.vlm_api_time:.2f}s")
    if enable_vision:
        print(f"  Total VLM Calls: {len(agent.visual_analysis_history)}")
    print(f'\nLogs saved to: {logdir}')
