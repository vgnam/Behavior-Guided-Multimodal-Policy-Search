"""
Runner for MLP-policy ProPS variants.

Supported task names:
  cont_space_llm_num_optim_mlp  — continuous-action environments (MuJoCo, etc.)
  atari_llm_num_optim_mlp       — Atari ALE (discrete actions, pixel observations)

Example config (LunarLander continuous with MLP):
    task: cont_space_llm_num_optim_mlp
    gym_env_name: LunarLanderContinuous-v2
    dim_states: 8
    dim_actions: 2
    hidden_dims: [32, 16]
    intrinsic_rank: 50     # optional: project MLP params to 50-d for LLM

Example config (Atari Pong):
    task: atari_llm_num_optim_mlp
    gym_env_name: ALE/Pong-v5
    dim_states: 1024       # = resize_h * resize_w (used only for reference)
    dim_actions: 6
    hidden_dims: [32]
    obs_projection_dim: 64  # random project 1024 → 64 before MLP
    intrinsic_rank: 50
    resize_h: 32
    resize_w: 32
"""

import os
import traceback
import numpy as np
from jinja2 import Environment, FileSystemLoader

from agent.llm_num_optim_mlp_policy import LLMNumOptimMLPAgent
from world.continuous_space_general_world import ContinualSpaceGeneralWorld

_ATARI_TASKS = {"atari_llm_num_optim_mlp"}
_CONT_TASKS  = {"cont_space_llm_num_optim_mlp"}


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    hidden_dims,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    optimum=1000,
    search_step_size=0.1,
    intrinsic_rank=None,
    obs_projection_dim=None,
    env_desc_file=None,
    env_kwargs=None,
    resize_h=32,
    resize_w=32,
    **kwargs,
):
    assert task in _ATARI_TASKS | _CONT_TASKS, (
        f"Unknown task '{task}'. Expected one of {_ATARI_TASKS | _CONT_TASKS}"
    )

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template      = jinja2_env.get_template(llm_si_template_name)
    llm_output_template  = jinja2_env.get_template(llm_output_conversion_template_name)

    # ── Build world ──────────────────────────────────────────────────────
    if task in _ATARI_TASKS:
        from world.atari_world import AtariWorld
        world = AtariWorld(
            gym_env_name, render_mode, max_traj_length,
            resize_h=resize_h, resize_w=resize_w,
        )
        obs_dim = world.obs_dim          # resize_h * resize_w
        is_discrete = True
    else:
        world = ContinualSpaceGeneralWorld(gym_env_name, render_mode, max_traj_length)
        obs_dim = dim_states
        is_discrete = False

    # ── Build agent ──────────────────────────────────────────────────────
    agent = LLMNumOptimMLPAgent(
        logdir=logdir,
        dim_action=dim_actions,
        dim_state=obs_dim,
        hidden_dims=hidden_dims,
        max_traj_count=max_traj_count,
        max_traj_length=max_traj_length,
        llm_si_template=llm_si_template,
        llm_output_conversion_template=llm_output_template,
        llm_model_name=llm_model_name,
        num_evaluation_episodes=num_evaluation_episodes,
        optimum=optimum,
        search_step_size=search_step_size,
        intrinsic_rank=intrinsic_rank,
        obs_projection_dim=obs_projection_dim,
        is_discrete=is_discrete,
    )

    # ── Warmup ───────────────────────────────────────────────────────────
    warmup_logdir = os.path.join(logdir, "warmup")
    os.makedirs(warmup_logdir, exist_ok=True)

    if warmup_dir:
        agent.replay_buffer.load(warmup_dir)
    elif warmup_episodes > 0:
        agent.random_warmup(world, warmup_logdir, warmup_episodes)

    # ── Training loop ────────────────────────────────────────────────────
    overall_log_path = os.path.join(logdir, "overall_log.txt")
    overall_log = open(overall_log_path, "w", encoding="utf-8")
    overall_log.write(
        "Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n"
    )
    overall_log.flush()

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"[MLP Runner] Episode: {episode}/{num_episodes}")
        print(f"{'='*60}")

        episode_dir = os.path.join(logdir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)

        for trial in range(5):
            try:
                cpu_time, api_time, total_eps, total_steps, reward = (
                    agent.train_policy(world, episode_dir)
                )
                overall_log.write(
                    f"{episode}, {cpu_time:.3f}, {api_time:.3f}, "
                    f"{total_eps}, {total_steps}, {reward:.4f}\n"
                )
                overall_log.flush()
                print(f"Episode {episode} done — reward: {reward:.3f}")
                break
            except Exception as e:
                print(f"Trial {trial + 1}/5 failed: {e}")
                traceback.print_exc()
                if trial == 4:
                    print(f"Episode {episode} failed after 5 attempts — stopping.")
                    overall_log.close()
                    return

    overall_log.close()
    print("\nTraining complete.")
