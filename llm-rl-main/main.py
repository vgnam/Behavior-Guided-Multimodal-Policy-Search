import argparse
import inspect
import importlib
import os

import yaml
from gymnasium.envs.registration import WrapperSpec as _WrapperSpec
from gymnasium.envs.registration import register as _gym_register
from gymnasium.error import Error as _GymError

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

try:
    from envs import grid2op_env  # noqa: F401
except ModuleNotFoundError:
    grid2op_env = None

try:
    importlib.import_module("fancy_gym")
except Exception:
    fancy_gym = None

try:
    import highway_env  # noqa: F401
except ModuleNotFoundError:
    highway_env = None

try:
    import gymnasium_robotics  # noqa: F401
except ModuleNotFoundError:
    gymnasium_robotics = None

_GYM_REGISTER_PARAMS = set(inspect.signature(_gym_register).parameters)


def _safe_register_stable_gym(**kwargs):
    try:
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in _GYM_REGISTER_PARAMS
        }
        _gym_register(**filtered_kwargs)
    except _GymError as exc:
        if "Cannot re-register id" not in str(exc):
            raise


# Register selected stable_gym environments manually to avoid importing
# stable_gym.__init__ (which can trigger compatibility issues in some setups).
_safe_register_stable_gym(
    id="stable_gym/Oscillator-v1",
    entry_point="stable_gym.envs.biological.oscillator.oscillator:Oscillator",
    max_episode_steps=400,
)
_safe_register_stable_gym(
    id="stable_gym/OscillatorComplicated-v1",
    entry_point="stable_gym.envs.biological.oscillator_complicated.oscillator_complicated:OscillatorComplicated",
    max_episode_steps=400,
)
_safe_register_stable_gym(
    id="stable_gym/FetchReachCost-v1",
    entry_point="stable_gym.envs.robotics.fetch.fetch_reach_cost.fetch_reach_cost:FetchReachCost",
    max_episode_steps=50,
)
_safe_register_stable_gym(
    id="stable_gym/MinitaurBulletCost-v1",
    entry_point="stable_gym.envs.robotics.minitaur.minitaur_bullet_cost.minitaur_bullet_cost:MinitaurBulletCost",
    max_episode_steps=500,
    disable_env_checker=True,
    apply_api_compatibility=True,
    additional_wrappers=(
        _WrapperSpec(
            name="MaxEpisodeStepsInjectionWrapper",
            entry_point="stable_gym.common.max_episode_steps_injection_wrapper:MaxEpisodeStepsInjectionWrapper",
            kwargs={},
        ),
    ),
)


TASK_TO_RUNNER = {
    "cont_space_llm_num_optim": "runner.llm_num_optim_runner",
    "dist_state_llm_num_optim": "runner.llm_num_optim_runner",
    "dist_state_llm_num_optim_semantics": "runner.llm_num_optim_semantics_runner",
    "cont_state_llm_num_optim_semantics": "runner.llm_num_optim_semantics_runner",
    "cont_state_llm_num_optim_vision": "runner.llm_num_optim_vision_runner",
    "dist_state_llm_num_optim_vision": "runner.llm_num_optim_vision_runner",
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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runner_module = _load_runner(config["task"])
    runner_module.run_training_loop(**config)


if __name__ == "__main__":
    main()
