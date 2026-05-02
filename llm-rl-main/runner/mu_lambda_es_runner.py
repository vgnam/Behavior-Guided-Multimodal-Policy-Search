"""Runner for the (mu, lambda)-ES baseline (linear policy)."""

import os
import shutil
import traceback

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.mu_lambda_es_linear_policy import MuLambdaESLinearPolicyAgent
from agent.mu_lambda_es_value_based import MuLambdaESValueBasedAgent


def _count_discrete(space_list):
    total = 1
    for sub in space_list:
        total *= len(sub)
    return total


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


def _maybe_infer_from_env(value, name, gym_env_name, env_kwargs):
    """Auto-infer dimension from the gym environment when value is None."""
    if value is not None:
        return value
    import gymnasium as gym

    kwargs = dict(env_kwargs or {})
    kwargs.pop("render_mode", None)

    if gym_env_name == "maze-sample-3x3-v0":
        env = gym.make(gym_env_name, enable_render=None, **kwargs)
    else:
        env = gym.make(gym_env_name, **kwargs)

    if name == "dim_states":
        space = env.observation_space
    elif name == "dim_actions":
        space = env.action_space
    else:
        env.close()
        raise ValueError(f"Unknown dimension name: {name}")

    env.close()

    if isinstance(space, gym.spaces.Discrete):
        return [list(range(space.n))]
    if isinstance(space, gym.spaces.Box):
        dim = int(space.shape[0]) if space.shape else 1
        return dim
    raise ValueError(f"Cannot auto-infer {name} from environment space: {type(space)}")


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
    mu=8,
    lam=32,
    sigma=0.1,
    sigma_decay=1.0,
    min_sigma=1e-12,
    max_sigma=None,
    param_bound=10.0,
    cxmode="blend",
    alpha=0.5,
    cxpb=0.6,
    mutpb=0.3,
    smin=0.01,
    smax=0.5,
    clip=True,
    ncores=1,
    candidate_evaluation_episodes=1,
    es_backend="auto",
    seed=None,
    env_kwargs=None,
    **kwargs,
):
    del max_traj_count
    del kwargs

    assert task in [
        "cont_space_mu_lambda_es",
        "mu_lambda_es_baseline",
        "dist_state_mu_lambda_es",
        "dist_state_mu_lambda_es_qvalue",
    ], (
        "(mu, lambda)-ES runner supports task in "
        "['cont_space_mu_lambda_es', 'mu_lambda_es_baseline', 'dist_state_mu_lambda_es', 'dist_state_mu_lambda_es_qvalue']"
    )

    dim_actions = _maybe_infer_from_env(dim_actions, "dim_actions", gym_env_name, env_kwargs)
    dim_states = _maybe_infer_from_env(dim_states, "dim_states", gym_env_name, env_kwargs)

    if task == "dist_state_mu_lambda_es_qvalue":
        if not isinstance(dim_actions, list) or not isinstance(dim_states, list):
            raise ValueError(
                "dist_state_mu_lambda_es_qvalue expects list-form action/state spaces, for example [[0,1,2,3]]."
            )
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )
        agent = MuLambdaESValueBasedAgent(
            logdir=logdir,
            n_states=_count_discrete(dim_states),
            n_actions=_count_discrete(dim_actions),
            max_traj_length=max_traj_length,
            num_evaluation_episodes=num_evaluation_episodes,
            mu=mu,
            lam=lam,
            sigma=sigma,
            sigma_decay=sigma_decay,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            param_bound=param_bound,
            cxmode=cxmode,
            alpha=alpha,
            cxpb=cxpb,
            mutpb=mutpb,
            smin=smin,
            smax=smax,
            clip=clip,
            ncores=ncores,
            candidate_evaluation_episodes=candidate_evaluation_episodes,
            es_backend=es_backend,
            seed=seed,
        )
    else:
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

        agent = MuLambdaESLinearPolicyAgent(
            logdir=logdir,
            dim_action=dim_actions,
            dim_state=dim_states,
            max_traj_length=max_traj_length,
            num_evaluation_episodes=num_evaluation_episodes,
            bias=bias,
            mu=mu,
            lam=lam,
            sigma=sigma,
            sigma_decay=sigma_decay,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            param_bound=param_bound,
            cxmode=cxmode,
            alpha=alpha,
            cxpb=cxpb,
            mutpb=mutpb,
            smin=smin,
            smax=smax,
            clip=clip,
            ncores=ncores,
            candidate_evaluation_episodes=candidate_evaluation_episodes,
            es_backend=es_backend,
            seed=seed,
        )

    os.makedirs(logdir, exist_ok=True)

    if warmup_dir:
        print("[(mu, lambda)-ES] warmup_dir is ignored (this baseline has no replay buffer).")
    elif warmup_episodes and warmup_episodes > 0:
        print("[(mu, lambda)-ES] warmup is disabled; warmup_episodes is ignored.")

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
                        f"[(mu, lambda)-ES best] episode={best_episode} "
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
            "parameters_flat.txt": "best_parameters_flat.txt",
            "training_rollout.txt": "best_training_rollout.txt",
            "es_diagnostics.txt": "best_es_diagnostics.txt",
        }
        for src_name, dst_name in files_to_copy.items():
            src_path = os.path.join(best_episode_dir, src_name)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, os.path.join(logdir, dst_name))

        print("\n[(mu, lambda)-ES final best]")
        print(f"  Best episode      : {best_episode}")
        print(f"  Best mean reward  : {best_reward:.6f}")
        print(f"  Eval rollouts     : {num_evaluation_episodes}")
