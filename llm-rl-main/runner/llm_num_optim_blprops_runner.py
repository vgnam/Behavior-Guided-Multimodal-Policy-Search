"""
Runner for BL-ProPS: Behavioral-Linguistic Prompted Policy Search.

Training loop:
  1. Initialize world and BLProPSAgent.
  2. Warmup: collect random episodes to seed the replay buffer and anchors.
  3. For each training episode:
       PRE-STEP → STEP 1 (LLM b_hyp) → STEP 2 (VLM b_post) → STEP 3 (LLM θ)
"""

import os
import traceback

from jinja2 import Environment, FileSystemLoader
from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from agent.llm_num_optim_blprops import BLProPSAgent


def run_training_loop(
    task: str,
    num_episodes: int,
    gym_env_name: str,
    render_mode: str,
    logdir: str,
    dim_actions: int,
    dim_states: int,
    max_traj_count: int,
    max_traj_length: int,
    template_dir: str,
    llm_hypothesis_template_name: str,
    llm_proposal_template_name: str,
    llm_model_name: str,
    num_evaluation_episodes: int,
    warmup_episodes: int,
    warmup_dir: str = None,
    bias: bool = True,
    optimum: float = 1000.0,
    search_step_size: float = 0.1,
    env_kwargs: dict = None,
    env_desc_file: str = None,
    vlm_model_name: str = "gemini-2.5-flash-lite",
    frame_sample_period: int = 50,
    **kwargs,
):
    """
    Run the BL-ProPS training loop.

    Args:
        task:                          Must be "blprops"
        num_episodes:                  Number of training iterations T
        gym_env_name:                  Gym environment ID
        render_mode:                   Render mode ("rgb_array" for frame capture)
        logdir:                        Root log directory
        dim_actions:                   Action space dimension
        dim_states:                    State space dimension
        max_traj_count:                Max episodes in replay buffer
        max_traj_length:               Max steps per episode
        template_dir:                  Directory with Jinja2 .j2 templates
        llm_hypothesis_template_name:  Template for STEP 1 (blprops_step1_hypothesis.j2)
        llm_proposal_template_name:    Template for STEP 3 (blprops_step3_proposal.j2)
        llm_model_name:                LiteLLM model string for the LLM
        num_evaluation_episodes:       Eval rollouts after each θ update
        warmup_episodes:               Random episodes before training
        warmup_dir:                    If set, load replay buffer from this directory
        bias:                          Whether linear policy uses a bias term
        optimum:                       Expected optimal reward (hint to LLM)
        search_step_size:              Exploration step size hint
        env_kwargs:                    Additional kwargs for the Gym environment
        env_desc_file:                 Jinja2 include path for environment description
        vlm_model_name:                VLM model name (Gemini)
        frame_sample_period:           Frame sampling interval P for VLM calls
    """
    assert task == "blprops", (
        f"BL-ProPS runner only supports task='blprops', got '{task}'"
    )

    # Force rgb_array render mode so frame capture works
    if render_mode != "rgb_array":
        render_mode = "rgb_array"
        print("[BL-ProPS] Overriding render_mode to 'rgb_array' for frame capture.")

    # Load Jinja2 templates
    jinja_env = Environment(loader=FileSystemLoader(template_dir))
    hypothesis_template = jinja_env.get_template(llm_hypothesis_template_name)
    proposal_template   = jinja_env.get_template(llm_proposal_template_name)

    # Initialize environment world
    world = ContinualSpaceGeneralWorld(gym_env_name, render_mode, max_traj_length)

    # Initialize agent
    agent = BLProPSAgent(
        logdir=logdir,
        dim_action=dim_actions,
        dim_state=dim_states,
        max_traj_count=max_traj_count,
        max_traj_length=max_traj_length,
        hypothesis_template=hypothesis_template,
        proposal_template=proposal_template,
        llm_model_name=llm_model_name,
        num_evaluation_episodes=num_evaluation_episodes,
        bias=bias,
        optimum=optimum,
        search_step_size=search_step_size,
        env_desc_file=env_desc_file,
        vlm_model_name=vlm_model_name,
        frame_sample_period=frame_sample_period,
        template_dir=template_dir,
    )

    print("[BL-ProPS] Initialization complete")
    print(f"  LLM       : {llm_model_name}")
    print(f"  VLM       : {vlm_model_name}")
    print(f"  env       : {gym_env_name}")
    print(f"  rank      : {agent.rank}")

    # Warmup phase
    # BL-ProPS always needs random_warmup to initialise anchor policies
    # (best / worst / current) for the VLM describe step.
    # When warmup_dir is supplied the replay buffer from that saved run is
    # loaded AFTER warmup so that the numerical history is richer, but we
    # still run a fresh warmup to seed the anchor frames.
    warmup_path = os.path.join(logdir, "warmup")
    os.makedirs(warmup_path, exist_ok=True)
    print(f"\n[BL-ProPS] Starting warmup with {warmup_episodes} episodes...")
    agent.random_warmup(world, warmup_path, warmup_episodes)
    print("[BL-ProPS] Warmup complete")

    if warmup_dir:
        print(f"[BL-ProPS] Loading additional replay buffer from {warmup_dir}")
        agent.replay_buffer.load(warmup_dir)

    # Open overall log
    os.makedirs(logdir, exist_ok=True)
    overall_log = open(os.path.join(logdir, "overall_log.txt"), "w", encoding="utf-8")
    overall_log.write(
        "Iteration, CPU Time, API Time (LLM+VLM), Total Episodes, Total Steps, Total Reward\n"
    )
    overall_log.flush()

    print(f"\n{'='*70}")
    print(f"[BL-ProPS] Training for {num_episodes} iterations")
    print(f"{'='*70}\n")

    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"[BL-ProPS] Episode {episode}/{num_episodes}")
        print(f"{'='*70}")

        ep_dir = os.path.join(logdir, f"episode_{episode}")
        os.makedirs(ep_dir, exist_ok=True)

        for trial in range(5):
            try:
                cpu_time, api_time, total_eps, total_steps, total_reward = agent.train_policy(
                    world, ep_dir
                )
                overall_log.write(
                    f"{episode}, {cpu_time:.2f}, {api_time:.2f}, "
                    f"{total_eps}, {total_steps}, {total_reward:.2f}\n"
                )
                overall_log.flush()
                print(
                    f"[BL-ProPS] ep={episode} | reward={total_reward:.2f} | "
                    f"cpu={cpu_time:.1f}s | api={api_time:.1f}s"
                )
                break
            except AssertionError as e:
                print(f"[BL-ProPS] Parse error on trial {trial+1}/5: {e}")
                if trial == 4:
                    print("[BL-ProPS] Giving up on this episode after 5 trials.")
                    traceback.print_exc()
            except Exception as e:
                print(f"[BL-ProPS] Error on trial {trial+1}/5: {e}")
                traceback.print_exc()
                if trial == 4:
                    print("[BL-ProPS] Giving up on this episode after 5 trials.")

    overall_log.close()
    print("\n[BL-ProPS] Training complete.")
    print(
        f"  Best policy reward: {agent._anchor_best['reward']:.2f}"
    )
