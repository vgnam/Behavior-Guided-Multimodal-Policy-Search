from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_q_table_semantics import LLMNumOptimQTableSemanticsAgent
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
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
    llm_api_key=None,
    llm_api_base=None,
    policy_type="linear",
    hidden_sizes=None,
    hidden_activation="tanh",
    output_activation="tanh",
    optimization_mode="direct",
    latent_dim=32,
    projection_seed=0,
    projection_scale=1.0,
    projection_refresh_interval=0,
):
    assert task in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    if task == "dist_state_llm_num_optim_semantics":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        agent = LLMNumOptimQTableSemanticsAgent(
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
            env_kwargs=env_kwargs,
            env_desc_file=env_desc_file,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
        )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        agent = LLMNumOptimSemanticAgent(
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
            env_desc_file=env_desc_file,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            policy_type=policy_type,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            optimization_mode=optimization_mode,
            latent_dim=latent_dim,
            projection_seed=projection_seed,
            projection_scale=projection_scale,
            projection_refresh_interval=projection_refresh_interval,
        )

    print('init done')
    if task == "cont_state_llm_num_optim_semantics":
        print(f'  Policy Type: {policy_type}')
        if policy_type == "mlp":
            print(f'  Hidden Sizes: {hidden_sizes}')
        print(f'  Optimization Mode: {optimization_mode}')
        if optimization_mode == "latent":
            print(f'  Latent Dimension: {agent.rank}/{agent.parameter_dim}')

    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        agent.random_warmup(world, warmup_dir, warmup_episodes)
    else:
        agent.replay_buffer.load(warmup_dir)
    
    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n")
    overall_log_file.flush()
    
    # Token statistics log
    token_stats_file = open(f"{logdir}/token_statistics.txt", "w", encoding="utf-8")
    token_stats_file.write("Iteration, LLM Prompt Tokens, LLM Completion Tokens, VLM Prompt Tokens, VLM Completion Tokens, "
                           "Total LLM Prompt Tokens, Total LLM Completion Tokens, Total VLM Prompt Tokens, Total VLM Completion Tokens, "
                           "Total Tokens\n")
    token_stats_file.flush()
    
    # Timing statistics log
    timing_stats_file = open(f"{logdir}/timing_statistics.txt", "w", encoding="utf-8")
    timing_stats_file.write("Iteration, LLM API Time (s), VLM API Time (s), CPU Time (s), Total Time (s)\n")
    timing_stats_file.flush()
    
    import time as _time
    wall_start_time = _time.time()
    
    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        for trial_idx in range(5):
            try:
                iter_start = _time.time()
                cpu_time, api_time, total_episodes, total_steps, total_reward, \
                    llm_pt, llm_ct, vlm_pt, vlm_ct = agent.train_policy(world, curr_episode_dir)
                iter_time = _time.time() - iter_start
                
                overall_log_file.write(f"{episode + 1}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}\n")
                overall_log_file.flush()
                
                # Token statistics
                total_tokens = (agent.total_llm_prompt_tokens + agent.total_llm_completion_tokens)
                token_stats_file.write(
                    f"{episode + 1}, {llm_pt}, {llm_ct}, {vlm_pt}, {vlm_ct}, "
                    f"{agent.total_llm_prompt_tokens}, {agent.total_llm_completion_tokens}, 0, 0, "
                    f"{total_tokens}\n"
                )
                token_stats_file.flush()
                
                # Timing statistics
                wall_total = _time.time() - wall_start_time
                timing_stats_file.write(
                    f"{episode + 1}, {api_time:.4f}, 0.0000, {cpu_time:.4f}, {wall_total:.4f}\n"
                )
                timing_stats_file.flush()
                
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(
                    f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                )
                traceback.print_exc()
                continue
        if trial_idx == 4:
            print(f"Episode {episode} failed to train after 5 attempts")
            break
    overall_log_file.close()
    token_stats_file.close()
    timing_stats_file.close()
