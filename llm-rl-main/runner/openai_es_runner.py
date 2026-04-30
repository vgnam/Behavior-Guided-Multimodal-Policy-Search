"""Runner for the OpenAI-ES baseline (linear policy)."""

import os
import shutil
import traceback

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.openai_es_linear_policy import OpenAIESLinearPolicyAgent


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
    noise_stdev=None,
    learning_rate=0.03,
    candidate_evaluation_episodes=1,
    return_proc_mode="centered_rank",
    optimizer_type="adam",
    use_centered_ranks=True,
    use_adam=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    l2coeff=None,
    grad_batch_size=500,
    sgd_momentum=0.9,
    seed=None,
    env_kwargs=None,
    **kwargs,
):
    del max_traj_count

    assert task in ["cont_space_openai_es", "openai_es_baseline", "dist_state_openai_es"], (
        "OpenAI-ES runner supports task in ['cont_space_openai_es', 'openai_es_baseline', 'dist_state_openai_es']"
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

    agent = OpenAIESLinearPolicyAgent(
        logdir=logdir,
        dim_action=dim_actions,
        dim_state=dim_states,
        max_traj_length=max_traj_length,
        num_evaluation_episodes=num_evaluation_episodes,
        bias=bias,
        population_size=population_size,
        sigma=sigma,
        noise_stdev=noise_stdev,
        learning_rate=learning_rate,
        candidate_evaluation_episodes=candidate_evaluation_episodes,
        return_proc_mode=return_proc_mode,
        optimizer_type=optimizer_type,
        use_centered_ranks=use_centered_ranks,
        use_adam=use_adam,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        l2coeff=l2coeff,
        grad_batch_size=grad_batch_size,
        sgd_momentum=sgd_momentum,
        seed=seed,
    )

    os.makedirs(logdir, exist_ok=True)

    if warmup_dir:
        print("[OpenAI-ES] warmup_dir is ignored (this baseline has no replay buffer).")
    elif warmup_episodes and warmup_episodes > 0:
        warmup_logdir = os.path.join(logdir, "warmup")
        os.makedirs(warmup_logdir, exist_ok=True)
        agent.random_warmup(world, warmup_logdir, warmup_episodes)

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
                if total_reward > best_reward:
                    best_reward = float(total_reward)
                    best_episode = int(episode)
                    best_episode_dir = curr_episode_dir
                    print(
                        f"[OpenAI-ES best] episode={best_episode} "
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
            f.write(
                f"best_mean_reward_{num_evaluation_episodes}_rollouts: {best_reward:.6f}\n"
            )

        files_to_copy = {
            "parameters.txt": "best_parameters.txt",
            "training_rollout.txt": "best_training_rollout.txt",
            "es_diagnostics.txt": "best_es_diagnostics.txt",
        }
        for src_name, dst_name in files_to_copy.items():
            src_path = os.path.join(best_episode_dir, src_name)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, os.path.join(logdir, dst_name))

        print("\n[OpenAI-ES final best]")
        print(f"  Best episode      : {best_episode}")
        print(f"  Best mean reward  : {best_reward:.6f}")
        print(f"  Eval rollouts     : {num_evaluation_episodes}")
