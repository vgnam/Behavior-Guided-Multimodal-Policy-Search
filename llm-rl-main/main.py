import yaml
import argparse
from runner import (
    llm_num_optim_runner,
)

import litellm
from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner
from runner import llm_num_optim_vision_runner
from runner import llm_num_optim_mlp_runner
from runner import llm_num_optim_mlp_semantic_runner
from runner import llm_num_optim_mlp_vision_runner
from runner import llm_num_optim_vision_oneshot_runner
# import gym_maze
# import gym_navigation
from envs import nim, pong

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
import os
os.environ["OPENROUTER_API_KEY"] = "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"

os.environ["NVIDIA_NIM_API_KEY"] = "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"
os.environ["MISTRAL_API_KEY"] = "wjLJ7TRAHtcDNv2VrIgE7dreAhVyYQBD"

os.environ["GEMINI_API_KEY"] = "AIzaSyBLtoejOAWxIkV5R1hV369pDopvdNqRkQk"

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

    if config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        llm_num_optim_runner.run_training_loop(**config)
    elif config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_num_optim_semantics_runner.run_training_loop(**config)
    elif config["task"] in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision"]:
        llm_num_optim_vision_runner.run_training_loop(**config)
    elif config["task"] in ["cont_space_llm_num_optim_mlp", "atari_llm_num_optim_mlp"]:
        llm_num_optim_mlp_runner.run_training_loop(**config)
    elif config["task"] in ["cont_space_llm_num_optim_mlp_semantics", "atari_llm_num_optim_mlp_semantics"]:
        llm_num_optim_mlp_semantic_runner.run_training_loop(**config)
    elif config["task"] in ["cont_space_llm_num_optim_mlp_vision", "atari_llm_num_optim_mlp_vision"]:
        llm_num_optim_mlp_vision_runner.run_training_loop(**config)
    elif config["task"] in [
        "cont_state_llm_num_optim_vision_oneshot",
        "dist_state_llm_num_optim_vision_oneshot",
    ]:
        llm_num_optim_vision_oneshot_runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
