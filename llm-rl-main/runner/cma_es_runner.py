"""Runner for the CMA-ES baseline (linear policy)."""

import os
import shutil
import traceback

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
try:
    from agent.cma_es_linear_policy import CMAESLinearPolicyAgent
except ModuleNotFoundError as exc:
    if exc.name == "cma":
        CMAESLinearPolicyAgent = None
    else:
        raise


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
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir=None,
    bias=True,
    population_size=32,
    sigma=0.1,
    elite_count=None,
    candidate_evaluation_episodes=1,
    covariance_type="auto",
    full_covariance_max_dim=256,
    decomposition_frequency=None,
    min_sigma=1e-12,
    max_sigma=None,
    seed=None,
    env_kwargs=None,
    **kwargs,
):
    del max_traj_count
    del kwargs

    assert task in ["cont_space_cma_es", "cma_es_baseline", "dist_state_cma_es"], (
        "CMA-ES runner supports task in ['cont_space_cma_es', 'cma_es_baseline', 'dist_state_cma_es']"
    )
    if CMAESLinearPolicyAgent is None:
        raise ModuleNotFoundError(
            "CMA-ES requires the `cma` package. Install dependencies from requirements.txt."
        )

    discrete_problem = _is_discrete_problem(dim_actions, dim_states)
    dim_actions = _infer_dimension(dim_actions, "dim_actions")
    dim_states = _infer_dimension(dim_states, "dim_states")

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
        dim_action=dim_actions,
        dim_state=dim_states,
        max_traj_length=max_traj_length,
        num_evaluation_episodes=num_evaluation_episodes,
        bias=bias,
        population_size=population_size,
        sigma=sigma,
        elite_count=elite_count,
        candidate_evaluation_episodes=candidate_evaluation_episodes,
        covariance_type=covariance_type,
        full_covariance_max_dim=full_covariance_max_dim,
        decomposition_frequency=decomposition_frequency,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        seed=seed,
    )

    os.makedirs(logdir, exist_ok=True)

    if warmup_dir:
        print("[CMA-ES] warmup is disabled; warmup_dir is ignored.")
    elif warmup_episodes and warmup_episodes > 0:
        print("[CMA-ES] warmup is disabled; warmup_episodes is ignored.")

    overall_log_file = open(f"{logdir}/overall_log.txt", "w", encoding="utf-8")
    overall_log_file.write(
        "Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n"
    )
    overall_log_file.flush()
    best_reward = float("-inf")
    best_episode = None
    best_episode_dir = None

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        curr_episode_dir = f"{logdir}/episode_{episode}"
        os.makedirs(curr_episode_dir, exist_ok=True)

        for trial_idx in range(5):
            try:
                cpu_time, api_time, total_episodes, total_steps, total_reward = (
                    agent.train_policy(world, curr_episode_dir)
                )
                overall_log_file.write(
                    f"{episode}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}\n"
                )
                overall_log_file.flush()
                agent.save_state(os.path.join(logdir, "cma_state_latest.pkl"))
                if total_reward > best_reward:
                    best_reward = float(total_reward)
                    best_episode = int(episode)
                    best_episode_dir = curr_episode_dir
                    print(
                        f"[CMA-ES best] episode={best_episode} "
                        f"mean_reward_{num_evaluation_episodes}_rollouts={best_reward:.6f}"
                    )
                print(f"{trial_idx + 1}th trial attempt succeeded in training")
                break
            except Exception as e:
                print(f"{trial_idx + 1}th trial attempt failed with error in training: {e}")
                traceback.print_exc()

                if trial_idx == 4:
                    print(f"All {trial_idx + 1} trials failed. Train terminated")
                    overall_log_file.close()
                    raise
                continue

    overall_log_file.close()
    if best_episode_dir is not None:
        best_summary_path = os.path.join(logdir, "best_summary.txt")
        with open(best_summary_path, "w", encoding="utf-8") as f:
            f.write(f"best_episode: {best_episode}\n")
            f.write(f"best_mean_reward_{num_evaluation_episodes}_rollouts: {best_reward:.6f}\n")

        files_to_copy = {
            "parameters.txt": "best_parameters.txt",
            "parameters_flat.txt": "best_parameters_flat.txt",
            "training_rollout.txt": "best_training_rollout.txt",
            "cma_diagnostics.txt": "best_cma_diagnostics.txt",
        }
        for src_name, dst_name in files_to_copy.items():
            src_path = os.path.join(best_episode_dir, src_name)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, os.path.join(logdir, dst_name))

        print("\n[CMA-ES final best]")
        print(f"  Best episode      : {best_episode}")
        print(f"  Best mean reward  : {best_reward:.6f}")
        print(f"  Eval rollouts     : {num_evaluation_episodes}")
