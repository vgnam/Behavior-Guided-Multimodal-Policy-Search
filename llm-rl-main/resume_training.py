"""
Resume training from existing logs.

This module provides functionality to continue training from where a previous
run left off, by:
  1. Parsing the overall_log.txt to determine the last completed episode.
  2. Loading warmup data into the replay buffer.
  3. Loading training episode rollouts into the replay buffer.
  4. Restoring policy parameters from the last episode's parameters.txt.
  5. Restoring cumulative counters (cpu_time, api_time, total_episodes, total_steps).
  6. Continuing the training loop from the next episode.

Usage:
    python resume_training.py --config config.yaml [--resume_from <logdir>]

If --resume_from is not specified, the logdir from the config is used.
"""

import yaml
import argparse
import importlib
import os
import re
import traceback
import numpy as np
from jinja2 import Environment, FileSystemLoader

# ── Imports for all runner/agent types ──────────────────────────────────
from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_linear_policy import LLMNumOptimAgent
from agent.llm_num_optim_linear_policy_rndm_proj import LLMNumOptimRndmPrjAgent
from agent.llm_num_optim_q_table import LLMNumOptimQTableAgent
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
from agent.llm_num_optim_q_table_semantics import LLMNumOptimQTableSemanticsAgent
from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent
from agent.llm_num_optim_q_table_vision import LLMNumOptimQTableVisionAgent
from agent.llm_num_optim_linear_policy_vision_oneshot import LLMNumOptimVisionOneshotAgent
from agent.openai_es_linear_policy import OpenAIESLinearPolicyAgent
try:
    from agent.cma_es_linear_policy import CMAESLinearPolicyAgent
except ModuleNotFoundError as exc:
    if exc.name == "cma":
        CMAESLinearPolicyAgent = None
    else:
        raise

from envs import nim, pong

try:
    from envs import grid2op_env  # noqa: F401
except ModuleNotFoundError:
    grid2op_env = None

try:
    from envs import robosuite_env  # noqa: F401
except ModuleNotFoundError:
    robosuite_env = None

try:
    importlib.import_module("fancy_gym")
except ModuleNotFoundError:
    fancy_gym = None

try:
    import highway_env  # noqa: F401
except ModuleNotFoundError:
    highway_env = None

try:
    import gymnasium_robotics  # noqa: F401
except ModuleNotFoundError:
    gymnasium_robotics = None


def _infer_dimension(value, name):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be an empty list")
        if len(value) == 1 and isinstance(value[0], list):
            return len(value[0])
        return len(value)
    raise ValueError(f"Unsupported type for {name}: {type(value)}")


def _is_discrete_problem(dim_actions, dim_states):
    return isinstance(dim_actions, list) or isinstance(dim_states, list)


# ─────────────────────────────────────────────────────────────────────────
# Helper: parse overall_log.txt
# ─────────────────────────────────────────────────────────────────────────

def parse_overall_log(logdir):
    """
    Parse overall_log.txt and return:
      - last_completed_episode (int): the 0-based episode index of the last
        successfully completed episode (-1 if none).
      - last_row dict with keys: iteration, cpu_time, api_time,
        total_episodes, total_steps, total_reward
      - all_rows: list of dicts for every row
    """
    log_path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No overall_log.txt found in {logdir}")

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header
    data_lines = [l.strip() for l in lines[1:] if l.strip()]
    if not data_lines:
        return -1, None, []

    all_rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        row = {
            "iteration": int(float(parts[0])),
            "cpu_time": float(parts[1]),
            "api_time": float(parts[2]),
            "total_episodes": int(parts[3]),
            "total_steps": int(parts[4]),
            "total_reward": float(parts[5]),
        }
        all_rows.append(row)

    last_row = all_rows[-1]

    # Determine the 0-based episode index.
    # Some runners write iteration as episode+1 (1-indexed), others as episode (0-indexed).
    # We detect by checking existing episode directories.
    last_iter = last_row["iteration"]

    # Check which episode directories actually exist
    completed_episodes = []
    for name in os.listdir(logdir):
        m = re.match(r"episode_(\d+)$", name)
        if m:
            ep_num = int(m.group(1))
            ep_dir = os.path.join(logdir, name)
            # An episode is "complete" if it has training_rollout.txt
            if os.path.exists(os.path.join(ep_dir, "training_rollout.txt")):
                completed_episodes.append(ep_num)

    if not completed_episodes:
        return -1, None, all_rows

    completed_episodes.sort()
    last_completed_episode = completed_episodes[-1]

    return last_completed_episode, last_row, all_rows


# ─────────────────────────────────────────────────────────────────────────
# Helper: load training rollouts into replay buffer
# ─────────────────────────────────────────────────────────────────────────

def load_training_rollouts_into_buffer(agent, task, logdir, max_episode):
    """
    Parse training_rollout.txt (and parameters.txt) from episode_0..episode_{max_episode}
    and add them to the agent's replay_buffer (and traj_buffer if present).
    """
    import ast
    qtable_tasks = {
        "dist_state_llm_num_optim",
        "dist_state_llm_num_optim_semantics",
        "dist_state_llm_num_optim_vision",
        "blprops_qtable",
    }
    is_qtable = task in qtable_tasks
    loaded = 0

    for ep in range(max_episode + 1):
        ep_dir = os.path.join(logdir, f"episode_{ep}")
        rollout_path = os.path.join(ep_dir, "training_rollout.txt")
        params_path = os.path.join(ep_dir, "parameters.txt")

        if not os.path.exists(rollout_path):
            continue

        try:
            with open(rollout_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse all "Total reward" values
            rewards = []
            for line in lines:
                if "Total reward" in line:
                    try:
                        rewards.append(float(line.split()[-1]))
                    except (ValueError, IndexError):
                        continue
            if not rewards:
                continue
            reward_mean = np.mean(rewards)

            params_flat = None

            if is_qtable:
                # Q-table: read action mapping from parameters.txt
                if os.path.exists(params_path):
                    with open(params_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    try:
                        mapping = ast.literal_eval(content)
                        params_flat = np.array(
                            [mapping[k] for k in sorted(mapping.keys())], dtype=float
                        )
                    except Exception as parse_err:
                        print(f"  Warning: Could not parse parameters.txt for episode_{ep}: {parse_err}")
            else:
                # Linear policy: params embedded in rollout before "parameter ends"
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(",")])
                    except (ValueError, IndexError):
                        continue
                if parameters:
                    params_flat = np.array(parameters).reshape(-1)

            if params_flat is None:
                continue

            agent.replay_buffer.add(params_flat, reward_mean)

            # For semantics agents: also populate traj_buffer from first rollout segment
            if hasattr(agent, "traj_buffer"):
                agent.traj_buffer.start_new_trajectory()
                in_traj = False
                for line in lines:
                    if "state | action | reward" in line.lower():
                        in_traj = True
                        continue
                    if in_traj and "Total reward" in line:
                        break
                    if in_traj:
                        parts = line.strip().split("|")
                        if len(parts) == 3:
                            try:
                                state = int(parts[0].strip())
                                action = int(parts[1].strip())
                                reward = float(parts[2].strip())
                                agent.traj_buffer.add_step(state, action, reward)
                            except ValueError:
                                continue

            loaded += 1

        except Exception as e:
            print(f"  Warning: Could not load episode_{ep}: {e}")
            continue

    return loaded


# ─────────────────────────────────────────────────────────────────────────
# Helper: parse parameters.txt to restore linear policy
# ─────────────────────────────────────────────────────────────────────────

def parse_linear_policy_parameters(params_path):
    """
    Parse parameters.txt for a linear policy.
    Format:
        Weights:
        w00, w01, ...
        w10, w11, ...
        ...
        Bias:
        b0, b1, ...

    Returns: weights (2D np.array), bias (1D np.array or None)
    """
    with open(params_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    weights = []
    bias = None
    section = None

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Weights"):
            section = "weights"
            continue
        if line.startswith("Bias"):
            section = "bias"
            continue

        if section == "weights":
            try:
                weights.append([float(x) for x in line.split(",")])
            except ValueError:
                continue
        elif section == "bias":
            try:
                bias = np.array([float(x) for x in line.split(",")])
            except ValueError:
                continue

    weights = np.array(weights) if weights else None
    return weights, bias


# ─────────────────────────────────────────────────────────────────────────
# Main: build agent + world, restore state, continue training
# ─────────────────────────────────────────────────────────────────────────

def resume_training(config, resume_logdir=None):
    """Resume training from existing logs."""
    task = config["task"]
    logdir = config["logdir"]
    resume_from = resume_logdir or logdir
    num_episodes = config["num_episodes"]

    # ── Step 1: Parse logs ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[Resume] Analyzing logs in: {resume_from}")
    print(f"{'='*70}")

    last_episode, last_row, all_rows = parse_overall_log(resume_from)

    if last_episode < 0:
        print("[Resume] No completed episodes found. Starting training from scratch.")
        start_episode = 0
    else:
        start_episode = last_episode + 1
        print(f"[Resume] Last completed episode: {last_episode}")
        print(f"  Total reward:   {last_row['total_reward']:.2f}")
        print(f"  Total episodes: {last_row['total_episodes']}")
        print(f"  Total steps:    {last_row['total_steps']}")
        print(f"  CPU time:       {last_row['cpu_time']:.2f}s")
        print(f"  API time:       {last_row['api_time']:.2f}s")
        print(f"[Resume] Will resume from episode {start_episode}")

    if start_episode >= num_episodes:
        print(f"[Resume] Training already complete ({start_episode}/{num_episodes} episodes).")
        return

    remaining = num_episodes - start_episode
    print(f"[Resume] Remaining episodes: {remaining}")

    # ── Step 2: Initialize agent + world (same as runners) ──────────────
    template_dir = config.get("template_dir", "agent/policy/templates")
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))

    dim_actions = config["dim_actions"]
    dim_states = config["dim_states"]
    max_traj_count = config.get("max_traj_count", 1000)
    max_traj_length = config.get("max_traj_length", 1000)
    gym_env_name = config["gym_env_name"]
    render_mode = config.get("render_mode")
    llm_model_name = config.get("llm_model_name")
    num_evaluation_episodes = config.get("num_evaluation_episodes", 20)
    bias = config.get("bias", True)
    optimum = config.get("optimum", 1000)
    search_step_size = config.get("search_step_size", 0.1)
    env_kwargs = config.get("env_kwargs")
    env_desc_file = config.get("env_desc_file")

    world = None
    agent = None

    if task in ["cont_space_openai_es", "dist_state_openai_es", "openai_es_baseline"]:
        discrete_problem = _is_discrete_problem(dim_actions, dim_states)
        inferred_actions = _infer_dimension(dim_actions, "dim_actions")
        inferred_states = _infer_dimension(dim_states, "dim_states")

        if discrete_problem:
            world = DiscreteStateGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )
        else:
            world = ContinualSpaceGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )

        agent = OpenAIESLinearPolicyAgent(
            logdir=logdir,
            dim_action=inferred_actions,
            dim_state=inferred_states,
            max_traj_length=max_traj_length,
            num_evaluation_episodes=num_evaluation_episodes,
            bias=bias,
            population_size=config.get("population_size", 32),
            sigma=config.get("sigma", 0.1),
            noise_stdev=config.get("noise_stdev", None),
            learning_rate=config.get("learning_rate", 0.03),
            candidate_evaluation_episodes=config.get("candidate_evaluation_episodes", 1),
            return_proc_mode=config.get("return_proc_mode", "centered_rank"),
            optimizer_type=config.get("optimizer_type", "adam"),
            use_centered_ranks=config.get("use_centered_ranks", True),
            use_adam=config.get("use_adam", True),
            adam_beta1=config.get("adam_beta1", 0.9),
            adam_beta2=config.get("adam_beta2", 0.999),
            adam_epsilon=config.get("adam_epsilon", 1e-8),
            weight_decay=config.get("weight_decay", 0.0),
            l2coeff=config.get("l2coeff", None),
            grad_batch_size=config.get("grad_batch_size", 500),
            sgd_momentum=config.get("sgd_momentum", 0.9),
            seed=config.get("seed", None),
        )

    elif task in ["cont_space_cma_es", "dist_state_cma_es", "cma_es_baseline"]:
        if CMAESLinearPolicyAgent is None:
            raise ModuleNotFoundError(
                "CMA-ES requires the `cma` package. Install dependencies from requirements.txt."
            )
        discrete_problem = _is_discrete_problem(dim_actions, dim_states)
        inferred_actions = _infer_dimension(dim_actions, "dim_actions")
        inferred_states = _infer_dimension(dim_states, "dim_states")

        if discrete_problem:
            world = DiscreteStateGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )
        else:
            world = ContinualSpaceGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )

        agent = CMAESLinearPolicyAgent(
            logdir=logdir,
            dim_action=inferred_actions,
            dim_state=inferred_states,
            max_traj_length=max_traj_length,
            num_evaluation_episodes=num_evaluation_episodes,
            bias=bias,
            population_size=config.get("population_size", 32),
            sigma=config.get("sigma", 0.1),
            elite_count=config.get("elite_count", None),
            candidate_evaluation_episodes=config.get("candidate_evaluation_episodes", 1),
            covariance_type=config.get("covariance_type", "auto"),
            full_covariance_max_dim=config.get("full_covariance_max_dim", 256),
            decomposition_frequency=config.get("decomposition_frequency", None),
            min_sigma=config.get("min_sigma", 1e-12),
            max_sigma=config.get("max_sigma", None),
            seed=config.get("seed", None),
        )

    elif task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj"]:
        llm_si_template = jinja2_env.get_template(config["llm_si_template_name"])
        llm_output_template = jinja2_env.get_template(config["llm_output_conversion_template_name"])

        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        if task == "cont_space_llm_num_optim":
            agent = LLMNumOptimAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, bias, optimum, search_step_size,
            )
        else:
            rank = config.get("rank")
            agent = LLMNumOptimRndmPrjAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, rank, bias, optimum, search_step_size,
            )

    elif task == "dist_state_llm_num_optim":
        llm_si_template = jinja2_env.get_template(config["llm_si_template_name"])
        llm_output_template = jinja2_env.get_template(config["llm_output_conversion_template_name"])

        world = DiscreteStateGeneralWorld(
            gym_env_name, render_mode, max_traj_length, env_kwargs=env_kwargs,
        )
        agent = LLMNumOptimQTableAgent(
            logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
            llm_si_template, llm_output_template, llm_model_name,
            num_evaluation_episodes, optimum, env_kwargs=env_kwargs,
        )

    elif task in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_si_template = jinja2_env.get_template(config["llm_si_template_name"])
        llm_output_template = jinja2_env.get_template(config["llm_output_conversion_template_name"])

        if task == "dist_state_llm_num_optim_semantics":
            world = DiscreteStateGeneralWorld(
                gym_env_name, render_mode, max_traj_length, env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimQTableSemanticsAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, optimum,
                env_kwargs=env_kwargs, env_desc_file=env_desc_file,
            )
        else:
            world = ContinualSpaceGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimSemanticAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, bias, optimum, search_step_size,
                env_desc_file=env_desc_file,
            )

    elif task in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision"]:
        llm_si_template = jinja2_env.get_template(config["llm_si_template_name"])
        llm_output_template = jinja2_env.get_template(config["llm_output_conversion_template_name"])

        vlm_model_name = config.get("vlm_model_name", "gpt-4o")
        decay_horizon = config.get("decay_horizon", 100)
        frame_sample_period = config.get("frame_sample_period", 50)
        enable_vision = config.get("enable_vision", True)
        n_neighbors = config.get("n_neighbors", 5)
        poisson_lam = config.get("poisson_lam", 2.0)
        neighbor_step = config.get("neighbor_step", 0.1)

        if enable_vision and render_mode is None:
            render_mode = "rgb_array"

        if task == "dist_state_llm_num_optim_vision":
            world = DiscreteStateGeneralWorld(
                gym_env_name, render_mode, max_traj_length, env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimQTableVisionAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, optimum,
                env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon, frame_sample_period=frame_sample_period,
                enable_vision=enable_vision, env_kwargs=env_kwargs,
                n_neighbors=n_neighbors, poisson_lam=poisson_lam,
                neighbor_step=neighbor_step,
            )
        else:
            world = ContinualSpaceGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimVisionAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, bias, optimum, search_step_size,
                env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon, frame_sample_period=frame_sample_period,
                enable_vision=enable_vision, n_neighbors=n_neighbors,
                poisson_lam=poisson_lam, neighbor_step=neighbor_step,
            )

    elif task in ["cont_state_llm_num_optim_vision_oneshot", "dist_state_llm_num_optim_vision_oneshot"]:
        llm_si_template = jinja2_env.get_template(config["llm_si_template_name"])
        llm_output_template = jinja2_env.get_template(config["llm_output_conversion_template_name"])

        vlm_model_name = config.get("vlm_model_name", "gpt-4o")
        decay_horizon = config.get("decay_horizon", 100)
        frame_sample_period = config.get("frame_sample_period", 50)
        enable_vision = config.get("enable_vision", True)
        n_neighbors = config.get("n_neighbors", 5)
        poisson_lam = config.get("poisson_lam", 2.0)
        neighbor_step = config.get("neighbor_step", 0.1)

        if enable_vision and render_mode is None:
            render_mode = "rgb_array"

        if task == "dist_state_llm_num_optim_vision_oneshot":
            world = DiscreteStateGeneralWorld(
                gym_env_name, render_mode, max_traj_length, env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimQTableVisionAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, optimum,
                env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon, frame_sample_period=frame_sample_period,
                enable_vision=enable_vision, env_kwargs=env_kwargs,
                n_neighbors=n_neighbors, poisson_lam=poisson_lam,
                neighbor_step=neighbor_step,
            )
        else:
            world = ContinualSpaceGeneralWorld(
                gym_env_name,
                render_mode,
                max_traj_length,
                env_kwargs=env_kwargs,
            )
            agent = LLMNumOptimVisionOneshotAgent(
                logdir, dim_actions, dim_states, max_traj_count, max_traj_length,
                llm_si_template, llm_output_template, llm_model_name,
                num_evaluation_episodes, bias, optimum, search_step_size,
                env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
                decay_horizon=decay_horizon, frame_sample_period=frame_sample_period,
                enable_vision=enable_vision, n_neighbors=n_neighbors,
                poisson_lam=poisson_lam, neighbor_step=neighbor_step,
            )

    elif task == "blprops":
        from agent.llm_num_optim_blprops import BLProPSAgent
        hypothesis_template = jinja2_env.get_template(config["llm_hypothesis_template_name"])
        proposal_template = jinja2_env.get_template(config["llm_proposal_template_name"])

        vlm_model_name = config.get("vlm_model_name", "gemini-2.5-flash-lite")
        frame_sample_period = config.get("frame_sample_period", 50)

        if render_mode != "rgb_array":
            render_mode = "rgb_array"

        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
        agent = BLProPSAgent(
            logdir=logdir, dim_action=dim_actions, dim_state=dim_states,
            max_traj_count=max_traj_count, max_traj_length=max_traj_length,
            hypothesis_template=hypothesis_template,
            proposal_template=proposal_template,
            llm_model_name=llm_model_name,
            num_evaluation_episodes=num_evaluation_episodes,
            bias=bias, optimum=optimum, search_step_size=search_step_size,
            env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
            frame_sample_period=frame_sample_period,
            template_dir=template_dir,
        )

    elif task == "blprops_qtable":
        from agent.llm_num_optim_q_table_blprops import BLProPSQTableAgent
        hypothesis_template = jinja2_env.get_template(config["llm_hypothesis_template_name"])
        proposal_template = jinja2_env.get_template(config["llm_proposal_template_name"])

        vlm_model_name = config.get("vlm_model_name", "gemini-2.5-flash-lite")
        frame_sample_period = config.get("frame_sample_period", 10)

        if render_mode != "rgb_array":
            render_mode = "rgb_array"

        world = DiscreteStateGeneralWorld(
            gym_env_name, render_mode, max_traj_length, env_kwargs=env_kwargs,
        )
        agent = BLProPSQTableAgent(
            logdir=logdir, actions=dim_actions, states=dim_states,
            max_traj_count=max_traj_count, max_traj_length=max_traj_length,
            hypothesis_template=hypothesis_template,
            proposal_template=proposal_template,
            llm_model_name=llm_model_name,
            num_evaluation_episodes=num_evaluation_episodes,
            optimum=optimum, search_step_size=search_step_size,
            env_desc_file=env_desc_file, vlm_model_name=vlm_model_name,
            frame_sample_period=frame_sample_period,
            template_dir=template_dir,
        )

    else:
        raise ValueError(f"Task '{task}' not recognized for resume.")

    if agent is None or world is None:
        raise RuntimeError("Failed to initialize agent or world.")

    print(f"[Resume] Agent and world initialized for task: {task}")

    # ── Step 3: Restore replay buffer ───────────────────────────────────
    if hasattr(agent, "replay_buffer"):
        warmup_dir = os.path.join(resume_from, "warmup")
        if os.path.exists(warmup_dir):
            print(f"[Resume] Loading warmup data from {warmup_dir}")
            agent.replay_buffer.load(warmup_dir)
            print(f"  Replay buffer size after warmup: {len(agent.replay_buffer.buffer)}")
        else:
            print("[Resume] WARNING: No warmup directory found — replay buffer starts empty.")

        if last_episode >= 0:
            loaded = load_training_rollouts_into_buffer(agent, task, resume_from, last_episode)
            print(f"[Resume] Loaded {loaded} training episodes into replay buffer")
            print(f"  Replay buffer size: {len(agent.replay_buffer.buffer)}")
    else:
        print("[Resume] Task does not use a replay buffer — skipping replay restore.")

    # ── Step 4: Restore policy parameters ───────────────────────────────
    qtable_tasks = {
        "dist_state_llm_num_optim",
        "dist_state_llm_num_optim_semantics",
        "dist_state_llm_num_optim_vision",
        "blprops_qtable",
    }
    if last_episode >= 0:
        import ast
        last_params_path = os.path.join(resume_from, f"episode_{last_episode}", "parameters.txt")
        if task in qtable_tasks:
            # Restore Q-table mapping from parameters.txt (dict string)
            if os.path.exists(last_params_path):
                with open(last_params_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                try:
                    mapping = ast.literal_eval(content)
                    agent.q_table.mapping = mapping
                    print(f"[Resume] Restored Q-table from episode_{last_episode}")
                except Exception as e:
                    print(f"[Resume] WARNING: Could not restore Q-table: {e}")
            else:
                print(f"[Resume] WARNING: No parameters.txt found for episode_{last_episode}")
        elif task in ["cont_space_cma_es", "dist_state_cma_es", "cma_es_baseline"]:
            cma_state_path = os.path.join(resume_from, "cma_state_latest.pkl")
            if os.path.exists(cma_state_path):
                try:
                    agent.load_state(cma_state_path)
                    print(f"[Resume] Restored CMA-ES strategy state from {cma_state_path}")
                except Exception as e:
                    print(f"[Resume] WARNING: Could not restore CMA-ES state: {e}")
            elif os.path.exists(last_params_path):
                weights, bias_val = parse_linear_policy_parameters(last_params_path)
                if weights is not None:
                    if bias_val is not None:
                        params = np.concatenate([weights.reshape(-1), bias_val.reshape(-1)])
                    else:
                        params = weights.reshape(-1)
                    agent.policy.update_policy(params)
                    agent.theta = agent.policy.get_parameters().reshape(-1).copy()
                    agent.strategy.mean = agent.theta.astype(np.float64)
                    print(f"[Resume] Restored CMA-ES mean policy from episode_{last_episode}")
                else:
                    print(f"[Resume] WARNING: Could not parse parameters from episode_{last_episode}")
            else:
                print(f"[Resume] WARNING: No CMA-ES state or parameters.txt found for episode_{last_episode}")
        elif os.path.exists(last_params_path):
            weights, bias_val = parse_linear_policy_parameters(last_params_path)
            if weights is not None:
                if bias_val is not None:
                    params = np.concatenate([weights.reshape(-1), bias_val.reshape(-1)])
                else:
                    params = weights.reshape(-1)
                agent.policy.update_policy(params)
                if hasattr(agent, "theta"):
                    agent.theta = agent.policy.get_parameters().reshape(-1).copy()
                print(f"[Resume] Restored policy parameters from episode_{last_episode}")
            else:
                print(f"[Resume] WARNING: Could not parse parameters from episode_{last_episode}")
        else:
            print(f"[Resume] WARNING: No parameters.txt found for episode_{last_episode}")

    # ── Step 5: Restore cumulative counters ─────────────────────────────
    if last_row is not None:
        agent.total_episodes = last_row["total_episodes"]
        agent.total_steps = last_row["total_steps"]
        agent.api_call_time = last_row["api_time"]
        agent.training_episodes = start_episode

        # Adjust start_time so that cpu_time offsets correctly
        import time
        agent.start_time = time.process_time() - last_row["cpu_time"]

        print(f"[Resume] Restored counters:")
        print(f"  training_episodes: {agent.training_episodes}")
        print(f"  total_episodes:    {agent.total_episodes}")
        print(f"  total_steps:       {agent.total_steps}")
        print(f"  api_call_time:     {agent.api_call_time:.2f}s")

    # ── Step 6: Vision-specific counter restore ─────────────────────────
    if task in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision",
                "cont_state_llm_num_optim_vision_oneshot", "dist_state_llm_num_optim_vision_oneshot"]:
        if hasattr(agent, "visual_guidance") and hasattr(agent.visual_guidance, "iteration"):
            agent.visual_guidance.iteration = start_episode
            print(f"[Resume] Restored visual guidance iteration to {start_episode}")
        if hasattr(agent, "vlm_api_time"):
            agent.vlm_api_time = 0  # VLM time is included in api_time, reset to accumulate fresh

    # ── Step 7: Continue training loop ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[Resume] Continuing training from episode {start_episode} to {num_episodes - 1}")
    print(f"{'='*70}\n")

    # Open overall_log in append mode
    overall_log_path = os.path.join(logdir, "overall_log.txt")
    if start_episode == 0:
        # Fresh start — overwrite
        overall_log_file = open(overall_log_path, "w", encoding="utf-8")
        overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n")
    else:
        # Append to existing
        overall_log_file = open(overall_log_path, "a", encoding="utf-8")

    overall_log_file.flush()

    # Vision stats file (if applicable)
    vision_stats_file = None
    enable_vision = config.get("enable_vision", False)
    if task in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision",
                "cont_state_llm_num_optim_vision_oneshot", "dist_state_llm_num_optim_vision_oneshot"]:
        vision_stats_path = os.path.join(logdir, "vision_statistics.txt")
        if start_episode == 0:
            vision_stats_file = open(vision_stats_path, "w", encoding="utf-8")
            vision_stats_file.write("Iteration, Lambda, VLM Invoked, Phase, Num Frames\n")
        else:
            vision_stats_file = open(vision_stats_path, "a", encoding="utf-8")
        vision_stats_file.flush()

    # Determine iteration indexing style from existing log
    # ProPS runners use 1-indexed (episode+1), ProPS-V/BL-ProPS use 0-indexed
    uses_1_indexed = task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj",
                               "dist_state_llm_num_optim",
                               "dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]

    for episode in range(start_episode, num_episodes):
        print(f"\n{'='*70}")
        print(f"[Resume] Episode: {episode}/{num_episodes}")
        print(f"{'='*70}")

        curr_episode_dir = os.path.join(logdir, f"episode_{episode}")
        os.makedirs(curr_episode_dir, exist_ok=True)

        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = agent.train_policy(
                    world, curr_episode_dir
                )

                # Write log entry matching the runner's format
                iter_val = episode + 1 if uses_1_indexed else episode
                overall_log_file.write(
                    f"{iter_val}, {cpu_time}, {api_time}, "
                    f"{total_episodes}, {total_steps}, {total_reward}\n"
                )
                overall_log_file.flush()
                if task in ["cont_space_cma_es", "dist_state_cma_es", "cma_es_baseline"]:
                    agent.save_state(os.path.join(logdir, "cma_state_latest.pkl"))

                # Vision statistics (if applicable)
                if vision_stats_file and enable_vision:
                    try:
                        stats = agent.visual_guidance.get_statistics()
                        vlm_invoked = (
                            len(agent.visual_analysis_history) > 0 and
                            agent.visual_analysis_history[-1]["iteration"] == episode
                        )
                        num_frames = (
                            agent.visual_analysis_history[-1]["num_frames"]
                            if vlm_invoked else 0
                        )
                        vision_stats_file.write(
                            f"{episode}, {stats['current_lambda']:.3f}, {vlm_invoked}, "
                            f"{stats['phase']}, {num_frames}\n"
                        )
                        vision_stats_file.flush()
                    except Exception:
                        pass

                print(f"[Resume] Episode {episode} completed — reward: {total_reward:.2f}")
                break

            except Exception as e:
                print(f"[Resume] Trial {trial_idx + 1}/5 failed: {e}")
                traceback.print_exc()

                if trial_idx == 4:
                    print(f"[Resume] Episode {episode} failed after 5 attempts.")
                    if task in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj",
                                "dist_state_llm_num_optim"]:
                        # Original ProPS runner exits on failure
                        overall_log_file.close()
                        if vision_stats_file:
                            vision_stats_file.close()
                        print("[Resume] Training terminated due to repeated failures.")
                        return
                    # Semantics/vision runners just break and move on
                    break
                continue

    overall_log_file.close()
    if vision_stats_file:
        vision_stats_file.close()

    print(f"\n{'='*70}")
    print(f"[Resume] Training complete! Episodes {start_episode}-{num_episodes - 1} done.")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Resume training from existing logs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file (same as used for original training)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to the log directory to resume from. Defaults to config's logdir.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set API keys (same as main.py)
    os.environ.setdefault("OPENROUTER_API_KEY", "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")
    os.environ.setdefault("NVIDIA_NIM_API_KEY", "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")
    os.environ.setdefault("MISTRAL_API_KEY", "wjLJ7TRAHtcDNv2VrIgE7dreAhVyYQBD")
    os.environ.setdefault("GEMINI_API_KEY", "AIzaSyDoMFuM881ierGjBmpT4O-tQyIrayjBPcw")

    resume_training(config, resume_logdir=args.resume_from)


if __name__ == "__main__":
    main()
