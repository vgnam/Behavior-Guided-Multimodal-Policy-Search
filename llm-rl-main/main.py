import argparse
import importlib
import os

import yaml

# import gym_maze
# import gym_navigation
try:
    from envs import nim  # noqa: F401
except ModuleNotFoundError:
    nim = None

try:
    from envs import pong  # noqa: F401
except ModuleNotFoundError:
    pong = None

# Register stable_gym oscillator environments manually to avoid the
# apply_api_compatibility incompatibility in stable_gym's __init__.py
from gymnasium.envs.registration import register as _gym_register

_gym_register(
    id="stable_gym/Oscillator-v1",
    entry_point="stable_gym.envs.biological.oscillator.oscillator:Oscillator",
    max_episode_steps=400,
)
_gym_register(
    id="stable_gym/OscillatorComplicated-v1",
    entry_point="stable_gym.envs.biological.oscillator_complicated.oscillator_complicated:OscillatorComplicated",
    max_episode_steps=400,
)

os.environ["OPENROUTER_API_KEY"] = "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"

os.environ["NVIDIA_NIM_API_KEY"] = "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"
os.environ["MISTRAL_API_KEY"] = "wjLJ7TRAHtcDNv2VrIgE7dreAhVyYQBD"

os.environ["GEMINI_API_KEY"] = "AIzaSyBLtoejOAWxIkV5R1hV369pDopvdNqRkQk"


TASK_TO_RUNNER = {
    "cont_space_llm_num_optim": "runner.llm_num_optim_runner",
    "cont_space_llm_num_optim_rndm_proj": "runner.llm_num_optim_runner",
    "dist_state_llm_num_optim": "runner.llm_num_optim_runner",
    "dist_state_llm_num_optim_semantics": "runner.llm_num_optim_semantics_runner",
    "cont_state_llm_num_optim_semantics": "runner.llm_num_optim_semantics_runner",
    "cont_state_llm_num_optim_vision": "runner.llm_num_optim_vision_runner",
    "dist_state_llm_num_optim_vision": "runner.llm_num_optim_vision_runner",
    "cont_space_llm_num_optim_mlp": "runner.llm_num_optim_mlp_runner",
    "atari_llm_num_optim_mlp": "runner.llm_num_optim_mlp_runner",
    "cont_space_llm_num_optim_mlp_semantics": "runner.llm_num_optim_mlp_semantic_runner",
    "atari_llm_num_optim_mlp_semantics": "runner.llm_num_optim_mlp_semantic_runner",
    "cont_space_llm_num_optim_mlp_vision": "runner.llm_num_optim_mlp_vision_runner",
    "atari_llm_num_optim_mlp_vision": "runner.llm_num_optim_mlp_vision_runner",
    "cont_state_llm_num_optim_vision_oneshot": "runner.llm_num_optim_vision_oneshot_runner",
    "dist_state_llm_num_optim_vision_oneshot": "runner.llm_num_optim_vision_oneshot_runner",
    "cont_space_openai_es": "runner.openai_es_runner",
    "dist_state_openai_es": "runner.openai_es_runner",
    "openai_es_baseline": "runner.openai_es_runner",
}


def _load_runner(task):
    try:
        module_path = TASK_TO_RUNNER[task]
    except KeyError as exc:
        raise ValueError(f"Task {task} not recognized.") from exc
    return importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing logs (auto-detects last completed episode)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to log directory to resume from (defaults to config's logdir)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.resume:
        from resume_training import resume_training
        resume_training(config, resume_logdir=args.resume_from)
        return

    runner_module = _load_runner(config["task"])
    runner_module.run_training_loop(**config)


if __name__ == "__main__":
    main()
