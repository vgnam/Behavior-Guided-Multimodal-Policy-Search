import yaml
import argparse
from runner import (
    llm_num_optim_runner,
)
from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner
from runner import llm_num_optim_vision_runner
# import gym_maze
# import gym_navigation
from envs import nim, pong
import os
os.environ["OPENROUTER_API_KEY"] = "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"

os.environ["NVIDIA_NIM_API_KEY"] = "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"
os.environ["MISTRAL_API_KEY"] = "wjLJ7TRAHtcDNv2VrIgE7dreAhVyYQBD"

os.environ["GEMINI_API_KEY"] = ("AIzaSyDoMFuM881ierGjBmpT4O-"
                                "tQyIrayjBPcw")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        llm_num_optim_runner.run_training_loop(**config)
    elif config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_num_optim_semantics_runner.run_training_loop(**config)
    elif config["task"] in ["cont_state_llm_num_optim_vision", "dist_state_llm_num_optim_vision"]:
        llm_num_optim_vision_runner.run_training_loop(**config)
    else:
        raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
