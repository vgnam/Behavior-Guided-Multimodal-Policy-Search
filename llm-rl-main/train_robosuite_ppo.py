"""Train Stable-Baselines3 PPO on the discrete robosuite benchmarks.

Examples (run from ``llm-rl-main``):

    py -3.10 train_robosuite_ppo.py --task lift --total-timesteps 1000000
    py -3.10 train_robosuite_ppo.py --task door --reward-mode dense
    py -3.10 train_robosuite_ppo.py --task pick-place-can

The policy consumes low-dimensional robot/object state. RGB rendering remains
available for evaluation, but is intentionally disabled during PPO training.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Importing this module registers the custom Gymnasium IDs.
from envs import robosuite_env  # noqa: F401


TASKS = {
    "lift": {
        "env_id": "RoboSuiteLiftDiscrete-v0",
        "horizon": 200,
        "state_dim": 64,
        "include_rotation_actions": False,
    },
    "door": {
        "env_id": "RoboSuiteDoorDiscrete-v0",
        "horizon": 400,
        "state_dim": 64,
        "include_rotation_actions": True,
    },
    "pick-place": {
        "env_id": "RoboSuitePickPlaceDiscrete-v0",
        "horizon": 800,
        "state_dim": 106,
        "include_rotation_actions": True,
    },
    "pick-place-can": {
        "env_id": "RoboSuitePickPlaceCanDiscrete-v0",
        "horizon": 400,
        "state_dim": 64,
        "include_rotation_actions": True,
    },
}


def parse_hidden_sizes(value):
    """Parse a comma-separated MLP width list."""
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden sizes must be integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return sizes


def make_env(task, reward_mode, seed, monitor_file=None, render_mode=None):
    """Construct one registered robosuite environment."""
    task_config = TASKS[task]
    env = gym.make(
        task_config["env_id"],
        render_mode=render_mode,
        horizon=task_config["horizon"],
        state_dim=task_config["state_dim"],
        reward_shaping=reward_mode == "dense",
        translation_delta=0.5,
        rotation_delta=0.5,
        include_rotation_actions=task_config["include_rotation_actions"],
        seed=seed,
    )
    if monitor_file is not None:
        env = Monitor(
            env,
            filename=str(monitor_file),
            info_keywords=("is_success",),
        )
    return env


def make_vec_env(task, reward_mode, seed, monitor_file, normalize_observation, normalize_reward):
    """Create a single-process vector environment with optional normalization."""
    vec_env = DummyVecEnv(
        [lambda: make_env(task, reward_mode, seed, monitor_file=monitor_file)]
    )
    return VecNormalize(
        vec_env,
        norm_obs=normalize_observation,
        norm_reward=normalize_reward,
        clip_obs=10.0,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train SB3 PPO on a discrete robosuite manipulation task."
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="lift")
    parser.add_argument("--reward-mode", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=[64, 64])
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--normalize-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the SB3 environment checker before training.",
    )
    return parser


def validate_args(parser, args):
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive")
    if args.n_steps <= 1:
        parser.error("--n-steps must be greater than 1")
    if args.batch_size <= 1 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [2, n_steps]")
    if args.eval_freq < 0 or args.checkpoint_freq < 0:
        parser.error("callback frequencies cannot be negative")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.check_env:
        validation_env = make_env(args.task, args.reward_mode, args.seed)
        try:
            check_env(validation_env, warn=True, skip_render_check=True)
        finally:
            validation_env.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.output_dir or f"logs/ppo_{args.task}_{args.reward_mode}"
    run_dir = Path(f"{base_dir}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "arguments.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    train_env = make_vec_env(
        args.task,
        args.reward_mode,
        args.seed,
        run_dir / "train_monitor.csv",
        args.normalize_observation,
        args.normalize_reward,
    )
    eval_env = None

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_robosuite",
                save_vecnormalize=True,
                verbose=1,
            )
        )

    if args.eval_freq > 0:
        eval_env = make_vec_env(
            args.task,
            args.reward_mode,
            args.seed + 10_000,
            run_dir / "eval_monitor.csv",
            args.normalize_observation,
            False,
        )
        eval_env.training = False
        eval_env.norm_reward = False
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            )
        )

    policy_kwargs = {
        "net_arch": {
            "pi": args.hidden_sizes,
            "vf": args.hidden_sizes,
        },
        "activation_fn": torch.nn.Tanh,
    }
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.set_logger(configure(str(run_dir / "sb3"), ["stdout", "csv"]))

    callback = CallbackList(callbacks) if callbacks else None
    print(f"[PPO] Task: {args.task} ({TASKS[args.task]['env_id']})")
    print(f"[PPO] Reward mode: {args.reward_mode}")
    print(f"[PPO] Run directory: {run_dir}")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        model.save(run_dir / "final_model")
        train_env.save(run_dir / "vecnormalize.pkl")
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    print(f"[PPO] Training complete. Model saved to {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
