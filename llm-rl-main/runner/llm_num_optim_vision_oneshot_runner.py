"""
Runner for ProPS-V OneShot: Vision-Guided Prompted Policy Search
with constant context length.

Differences from llm_num_optim_vision_runner.py:
  - Uses LLMNumOptimVisionOneshotAgent instead of LLMNumOptimVisionAgent.
  - The LLM prompt always contains exactly ONE example (the replay-buffer best),
    so input size never grows with the number of training iterations.
  - VLM is invoked every iteration unconditionally (no stochastic annealing).

Supported task names:
  cont_state_llm_num_optim_vision_oneshot
  dist_state_llm_num_optim_vision_oneshot
"""

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy_vision_oneshot import (
    LLMNumOptimVisionOneshotAgent,
)
from agent.llm_num_optim_q_table_vision import LLMNumOptimQTableVisionAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback


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
    frame_sample_period=50,
    enable_vision=True,
    n_neighbors=5,
    poisson_lam=2.0,
    neighbor_step=0.1,
    ablate_anchor=None,
):
    """
    Run ProPS-V OneShot training loop.

    The agent only ever shows ONE example (best replay-buffer entry) to the LLM,
    keeping prompt length bounded regardless of training duration.

    Args are identical to llm_num_optim_vision_runner.run_training_loop.
    """
    assert task in [
        "cont_state_llm_num_optim_vision_oneshot",
        "dist_state_llm_num_optim_vision_oneshot",
    ], (
        f"ProPS-V OneShot runner only supports "
        f"'cont_state_llm_num_optim_vision_oneshot' or "
        f"'dist_state_llm_num_optim_vision_oneshot', got '{task}'"
    )

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    env_description = env_desc_file if env_desc_file else None

    # Vision requires rgb_array rendering
    if enable_vision and render_mode is None:
        render_mode = "rgb_array"
        print("[ProPS-V OneShot] Enabling rgb_array rendering for frame capture")

    # ------------------------------------------------------------------
    # Initialize world and agent
    # ------------------------------------------------------------------
    if task == "dist_state_llm_num_optim_vision_oneshot":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
        # Q-table variant reuses the base vision Q-table agent
        # (one-shot string building happens in the runner-level wrapper
        #  if needed; for discrete tasks the buffer is small by default)
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
            env_kwargs=env_kwargs,
            n_neighbors=n_neighbors,
            poisson_lam=poisson_lam,
            neighbor_step=neighbor_step,
            ablate_anchor=ablate_anchor,
        )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
        )
        agent = LLMNumOptimVisionOneshotAgent(
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
            n_neighbors=n_neighbors,
            poisson_lam=poisson_lam,
            neighbor_step=neighbor_step,
            ablate_anchor=ablate_anchor,
        )

    print("[ProPS-V OneShot] Initialization done")
    print(f"  LLM          : {llm_model_name}")
    print(f"  VLM          : {vlm_model_name}")
    print(f"  Vision       : {enable_vision}")
    print(f"  Prompt style : one-shot (best example only)")
    print(f"  VLM schedule : every iteration")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        print(f"\n[ProPS-V OneShot] Warmup: {warmup_episodes} episodes...")
        agent.random_warmup(world, warmup_dir, warmup_episodes)
        print("[ProPS-V OneShot] Warmup complete")
    else:
        print(f"[ProPS-V OneShot] Loading warmup from {warmup_dir}")
        agent.replay_buffer.load(warmup_dir)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    overall_log_file = open(
        f"{logdir}/overall_log.txt", "w", encoding="utf-8"
    )
    overall_log_file.write(
        "Iteration, CPU Time, API Time (LLM+VLM), "
        "Total Episodes, Total Steps, Total Reward\n"
    )
    overall_log_file.flush()

    vision_stats_file = open(
        f"{logdir}/vision_statistics.txt", "w", encoding="utf-8"
    )
    vision_stats_file.write(
        "Iteration, VLM Invoked, Num Frames\n"
    )
    vision_stats_file.flush()

    print("\n" + "=" * 70)
    print(f"[ProPS-V OneShot] Starting training — {num_episodes} episodes")
    print("=" * 70 + "\n")

    for episode in range(num_episodes):
        print("\n" + "=" * 70)
        print(f"ProPS-V OneShot — Episode {episode}/{num_episodes}")
        print("=" * 70)

        curr_episode_dir = f"{logdir}/episode_{episode}"
        os.makedirs(curr_episode_dir, exist_ok=True)

        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = (
                    agent.train_policy(world, curr_episode_dir)
                )

                overall_log_file.write(
                    f"{episode}, {cpu_time:.2f}, {api_time:.2f}, "
                    f"{total_episodes}, {total_steps}, {total_reward:.2f}\n"
                )
                overall_log_file.flush()

                # Vision stats
                if enable_vision:
                    vlm_invoked = (
                        len(agent.visual_analysis_history) > 0
                        and agent.visual_analysis_history[-1]["iteration"]
                        == episode
                    )
                    num_frames = (
                        agent.visual_analysis_history[-1]["num_frames"]
                        if vlm_invoked
                        else 0
                    )
                    vision_stats_file.write(
                        f"{episode}, {vlm_invoked}, {num_frames}\n"
                    )
                    vision_stats_file.flush()

                print(f"\n[SUCCESS] Trial {trial_idx + 1} succeeded")
                print(f"  Total Reward : {total_reward:.2f}")
                print(f"  CPU Time     : {cpu_time:.2f}s")
                print(
                    f"  API Time     : {api_time:.2f}s  "
                    f"(LLM: {agent.api_call_time:.2f}s, "
                    f"VLM: {agent.vlm_api_time:.2f}s)"
                )
                break

            except Exception as e:
                print(f"\n[ERROR] Trial {trial_idx + 1}/5 failed: {e}")
                traceback.print_exc()

                if trial_idx == 4:
                    print(f"\n[FAILURE] Episode {episode} failed after 5 attempts")
                    overall_log_file.close()
                    vision_stats_file.close()
                    return
                continue

    overall_log_file.close()
    vision_stats_file.close()
