"""
Runner for BMPS: Vision-Guided Prompted Policy Search

This module provides the training loop for BMPS agents,
integrating vision-language model feedback with policy optimization.
"""

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent
from agent.llm_num_optim_q_table_vision import LLMNumOptimQTableVisionAgent
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
    vlm_model_name="openrouter/google/gemma-3-27b-it",
    decay_horizon=100,
    frame_sample_period=50,
    enable_vision=True,
    n_neighbors=5,
    poisson_lam=2.0,
    neighbor_step=0.1,
    ablate_anchor=None,
    parallel_vlm_calls=True,
    vlm_max_workers=None,
    llm_api_key=None,
    llm_api_base=None,
    vlm_api_key=None,
    vlm_api_base=None,
    vlm_frame_mode="overlay",
    optimization_mode="direct",
    latent_dim=32,
    projection_seed=0,
    projection_scale=1.0,
    projection_refresh_interval=0,
    policy_type="linear",
    hidden_sizes=None,
    hidden_activation="tanh",
    output_activation="tanh",
    max_total_tokens=None,
):
    """
    Run BMPS training loop.
    
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
        rank: Legacy parameter rank (unused by the vision runner)
        optimum: Expected optimal reward
        search_step_size: Step size for exploration
        env_kwargs: Additional environment kwargs
        env_desc_file: Path to environment description file
        vlm_model_name: Vision-Language Model for visual analysis
        decay_horizon: T_decay for visual guidance annealing (Eq. 3)
        frame_sample_period: P — capture a frame every P timesteps for VLM
        enable_vision: Whether to enable vision-guided feedback
        parallel_vlm_calls: Call the VLM for anchor and neighbors concurrently
        vlm_max_workers: Maximum number of concurrent VLM calls
        optimization_mode: Full-space "direct" search or random-subspace "latent" search
        latent_dim: Number of latent coordinates exposed to the LLM
        projection_seed: Seed for the orthonormal random projection
        projection_scale: Full-space update scale alpha
        projection_refresh_interval: Refresh the projection every N iterations (0 disables)
        policy_type: Continuous policy architecture, "linear" or "mlp"
        hidden_sizes: Width of each MLP hidden layer
        hidden_activation: MLP hidden activation
        output_activation: MLP output activation
        max_total_tokens: Stop after cumulative LLM and VLM usage reaches this budget.
            The final iteration may overshoot the budget because token usage is
            only known after the API calls complete. ``None`` disables the limit.
    """
    assert task in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision"], \
        f"BMPS runner only supports 'cont_state_llm_num_optim_vision' or 'dist_state_llm_num_optim_vision', got '{task}'"
    if max_total_tokens is not None and max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be a positive integer or None")

    # Load Jinja2 templates
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    
    # env_desc_file is passed as a template name for {% include %} in j2 templates
    env_description = env_desc_file if env_desc_file else None
    
    # Initialize world
    # For vision, we need rgb_array rendering to capture frames
    if enable_vision and render_mode is None:
        render_mode = "rgb_array"
        print(f"[BMPS] Enabling rgb_array rendering for frame capture")
    
    if task == "dist_state_llm_num_optim_vision":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
        if policy_type == "mlp":
            state_values = (
                dim_states[0]
                if len(dim_states) == 1 and isinstance(dim_states[0], list)
                else dim_states
            )
            action_values = (
                dim_actions[0]
                if len(dim_actions) == 1 and isinstance(dim_actions[0], list)
                else dim_actions
            )
            agent = LLMNumOptimVisionAgent(
                logdir,
                len(action_values),
                len(state_values),
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                True if bias is None else bias,
                optimum,
                search_step_size,
                env_desc_file=env_description,
                vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon,
                frame_sample_period=frame_sample_period,
                enable_vision=enable_vision,
                vlm_frame_mode=vlm_frame_mode,
                n_neighbors=n_neighbors,
                poisson_lam=poisson_lam,
                neighbor_step=neighbor_step,
                ablate_anchor=ablate_anchor,
                parallel_vlm_calls=parallel_vlm_calls,
                vlm_max_workers=vlm_max_workers,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                vlm_api_key=vlm_api_key,
                vlm_api_base=vlm_api_base,
                optimization_mode=optimization_mode,
                latent_dim=latent_dim,
                projection_seed=projection_seed,
                projection_scale=projection_scale,
                projection_refresh_interval=projection_refresh_interval,
                policy_type=policy_type,
                hidden_sizes=hidden_sizes,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                state_encoding="one_hot",
            )
        else:
            agent = LLMNumOptimQTableVisionAgent(
                logdir,
                dim_actions,
                dim_states,
                max_traj_count,
                max_traj_length,
                llm_si_template,
                llm_output_conversion_template,
                llm_model_name,
                num_evaluation_episodes,
                optimum,
                env_desc_file=env_description,
                vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon,
                frame_sample_period=frame_sample_period,
                enable_vision=enable_vision,
                vlm_frame_mode=vlm_frame_mode,
                env_kwargs=env_kwargs,
                n_neighbors=n_neighbors,
                poisson_lam=poisson_lam,
                neighbor_step=neighbor_step,
                ablate_anchor=ablate_anchor,
                parallel_vlm_calls=parallel_vlm_calls,
                vlm_max_workers=vlm_max_workers,
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                vlm_api_key=vlm_api_key,
                vlm_api_base=vlm_api_base,
            )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
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
            frame_sample_period=frame_sample_period,
            enable_vision=enable_vision,
            vlm_frame_mode=vlm_frame_mode,
            n_neighbors=n_neighbors,
            poisson_lam=poisson_lam,
            neighbor_step=neighbor_step,
            ablate_anchor=ablate_anchor,
            parallel_vlm_calls=parallel_vlm_calls,
            vlm_max_workers=vlm_max_workers,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            vlm_api_key=vlm_api_key,
            vlm_api_base=vlm_api_base,
            optimization_mode=optimization_mode,
            latent_dim=latent_dim,
            projection_seed=projection_seed,
            projection_scale=projection_scale,
            projection_refresh_interval=projection_refresh_interval,
            policy_type=policy_type,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
    
    print('[BMPS] Initialization done')
    print(f'  LLM: {llm_model_name}')
    print(f'  VLM: {vlm_model_name}')
    print(f'  Vision Enabled: {enable_vision}')
    print(f'  Decay Horizon: {decay_horizon}')
    print(f'  Frame Sample Period: {frame_sample_period}')
    print(f'  VLM Frame Mode: {vlm_frame_mode}')
    print(f'  Parallel VLM Calls: {parallel_vlm_calls}')
    if parallel_vlm_calls:
        print(f'  VLM Max Workers: {vlm_max_workers or "executor default"}')
    if task == "cont_state_llm_num_optim_vision" or policy_type == "mlp":
        print(f'  Optimization Mode: {optimization_mode}')
        print(f'  Policy Type: {policy_type}')
        if policy_type == "mlp":
            print(f'  Hidden Sizes: {hidden_sizes}')
        if optimization_mode == "latent":
            print(f'  Latent Dimension: {agent.rank}/{agent.parameter_dim}')
            print(f'  Projection Scale: {projection_scale}')
    else:
        print('  Policy Type: q_table')
    
    # Warmup phase
    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        print(f'\n[BMPS] Starting warmup with {warmup_episodes} episodes...')
        agent.random_warmup(world, warmup_dir, warmup_episodes)
        print('[BMPS] Warmup complete')
    else:
        print(f'[BMPS] Loading warmup data from {warmup_dir}')
        agent.replay_buffer.load(warmup_dir)

    best_reward = float("-inf")
    best_parameters = None
    best_source = None

    def update_best_result(source):
        """Track and persist the best evaluated policy seen so far."""
        nonlocal best_reward, best_parameters, best_source
        if not agent.replay_buffer.buffer:
            return

        parameters, reward = max(agent.replay_buffer.buffer, key=lambda item: item[1])
        reward = float(reward)
        if reward <= best_reward:
            return

        best_reward = reward
        best_parameters = np.asarray(parameters).reshape(-1).copy()
        best_source = source
        with open(f"{logdir}/best_result.txt", "w", encoding="utf-8") as best_file:
            best_file.write(f"Best reward: {best_reward}\n")
            best_file.write(f"Found at: {best_source}\n")
            best_file.write(
                "Parameters: "
                + np.array2string(best_parameters, separator=", ", threshold=np.inf)
                + "\n"
            )
        print(f"[BEST] New best reward: {best_reward:.6f} ({best_source})")

    update_best_result("warmup")
    
    # Training loop
    overall_log_file = open(f"{logdir}/overall_log.txt", "w", encoding="utf-8")
    overall_log_file.write("Iteration, CPU Time, API Time (LLM+VLM), Total Episodes, Total Steps, Total Reward\n")
    overall_log_file.flush()
    
    # Create vision statistics log
    vision_stats_file = open(f"{logdir}/vision_statistics.txt", "w", encoding="utf-8")
    vision_stats_file.write("Iteration, Lambda, VLM Invoked, Phase, Num Frames\n")
    vision_stats_file.flush()
    
    # Token statistics log
    token_stats_file = open(f"{logdir}/token_statistics.txt", "w", encoding="utf-8")
    token_stats_file.write("Iteration, Iteration LLM Prompt Tokens, Iteration LLM Completion Tokens, "
                           "Iteration VLM Prompt Tokens, Iteration VLM Completion Tokens, "
                           "Cumulative LLM Prompt Tokens, Cumulative LLM Completion Tokens, "
                           "Cumulative VLM Prompt Tokens, Cumulative VLM Completion Tokens, "
                           "Cumulative Total Tokens\n")
    token_stats_file.flush()
    
    # Timing statistics log
    timing_stats_file = open(f"{logdir}/timing_statistics.txt", "w", encoding="utf-8")
    timing_stats_file.write("Iteration, LLM API Time (s), VLM API Time (s), CPU Time (s), Total Time (s)\n")
    timing_stats_file.flush()
    
    import time as _time
    wall_start_time = _time.time()
    
    print('\n' + '='*70)
    print(f"[BMPS] Starting training for {num_episodes} episodes")
    print('='*70 + '\n')
    
    token_budget_reached = False
    for episode in range(num_episodes):
        print('\n' + '='*70)
        print(f"BMPS Episode: {episode}/{num_episodes}")
        print('='*70)
        
        # Create episode log directory
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        

        # Train policy (with retries)
        for trial_idx in range(5):
            try:
                # Save VLM API time before iteration to compute per-iteration VLM time
                vlm_time_before = agent.vlm_api_time
                llm_time_before = agent.api_call_time - agent.vlm_api_time
                
                iter_start = _time.time()
                cpu_time, api_time, total_episodes, total_steps, total_reward, \
                    llm_pt, llm_ct, vlm_pt, vlm_ct = agent.train_policy(
                    world, curr_episode_dir
                )
                iter_time = _time.time() - iter_start
                
                # Compute per-iteration times
                iter_vlm_time = agent.vlm_api_time - vlm_time_before
                iter_llm_time = (agent.api_call_time - agent.vlm_api_time) - llm_time_before
                
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
                        agent.visual_analysis_history[-1]['num_frames'] 
                        if vlm_invoked else 0
                    )
                    
                    vision_stats_file.write(
                        f"{episode}, {stats['current_lambda']:.3f}, {vlm_invoked}, "
                        f"{stats['phase']}, {num_frames}\n"
                    )
                    vision_stats_file.flush()
                
                # Token statistics
                total_tokens = (agent.total_llm_prompt_tokens + agent.total_llm_completion_tokens +
                                agent.total_vlm_prompt_tokens + agent.total_vlm_completion_tokens)
                token_stats_file.write(
                    f"{episode}, {llm_pt}, {llm_ct}, {vlm_pt}, {vlm_ct}, "
                    f"{agent.total_llm_prompt_tokens}, {agent.total_llm_completion_tokens}, "
                    f"{agent.total_vlm_prompt_tokens}, {agent.total_vlm_completion_tokens}, "
                    f"{total_tokens}\n"
                )
                token_stats_file.flush()
                
                # Timing statistics
                wall_total = _time.time() - wall_start_time
                timing_stats_file.write(
                    f"{episode}, {iter_llm_time:.4f}, {iter_vlm_time:.4f}, {cpu_time:.4f}, {wall_total:.4f}\n"
                )
                timing_stats_file.flush()
                
                print(f"\n[SUCCESS] Trial {trial_idx + 1} succeeded")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  CPU Time: {cpu_time:.2f}s")
                print(f"  API Time: {api_time:.2f}s (LLM: {agent.api_call_time:.2f}s, VLM: {agent.vlm_api_time:.2f}s)")
                print(f"  Tokens (Iteration) - LLM: prompt={llm_pt}, completion={llm_ct} | VLM: prompt={vlm_pt}, completion={vlm_ct}")
                print(
                    f"  Tokens (Cumulative) - LLM: prompt={agent.total_llm_prompt_tokens}, "
                    f"completion={agent.total_llm_completion_tokens} | "
                    f"VLM: prompt={agent.total_vlm_prompt_tokens}, "
                    f"completion={agent.total_vlm_completion_tokens}"
                )
                print(f"  Cumulative Tokens: {total_tokens}")
                update_best_result(f"iteration {episode}")
                print(f"  Best Reward Found: {best_reward:.6f} ({best_source})")
                if max_total_tokens is not None and total_tokens >= max_total_tokens:
                    token_budget_reached = True
                    print(
                        f"[STOP] Total-token budget reached: "
                        f"{total_tokens:,}/{max_total_tokens:,}"
                    )
                break
                
            except Exception as e:
                print(f"\n[ERROR] Trial {trial_idx + 1}/5 failed: {e}")
                traceback.print_exc()
                
                if trial_idx == 4:
                    print(f"\n[FAILURE] Episode {episode} failed after 5 attempts")
                    overall_log_file.close()
                    vision_stats_file.close()
                    token_stats_file.close()
                    timing_stats_file.close()
                    return
                else:
                    continue

        if token_budget_reached:
            break
    
    overall_log_file.close()
    vision_stats_file.close()
    token_stats_file.close()
    timing_stats_file.close()
    
    # Final summary
    total_tokens = (agent.total_llm_prompt_tokens + agent.total_llm_completion_tokens +
                    agent.total_vlm_prompt_tokens + agent.total_vlm_completion_tokens)
    
    print('\n' + '='*70)
    print("[BMPS] Training Complete!")
    print('='*70)
    print(f"  Final Episode: {episode}")
    print(f"  Final Iteration Reward: {total_reward:.2f}")
    print(f"  Best Reward Found: {best_reward:.6f} ({best_source})")
    if best_parameters is not None:
        print(
            "  Best Parameters: "
            + np.array2string(best_parameters, separator=", ", threshold=50, edgeitems=5)
        )
    print(f"  Best result saved to: {logdir}/best_result.txt")
    print(f"  Total API Time: {api_time:.2f}s")
    print(f"    - LLM Time: {agent.api_call_time:.2f}s")
    print(f"    - VLM Time: {agent.vlm_api_time:.2f}s")
    print(f"  Total Tokens: {total_tokens}")
    print(f"    - LLM Prompt Tokens: {agent.total_llm_prompt_tokens}")
    print(f"    - LLM Completion Tokens: {agent.total_llm_completion_tokens}")
    print(f"    - VLM Prompt Tokens: {agent.total_vlm_prompt_tokens}")
    print(f"    - VLM Completion Tokens: {agent.total_vlm_completion_tokens}")
    if enable_vision:
        print(f"  Total VLM Calls: {len(agent.visual_analysis_history)}")
    print(f'\nLogs saved to: {logdir}')
