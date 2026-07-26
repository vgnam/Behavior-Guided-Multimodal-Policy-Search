import argparse
from datetime import datetime
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


def _parse_cli_value(value):
    """Parse CLI values using YAML syntax (scalars, lists, and dictionaries)."""
    return yaml.safe_load(value)


def _cli_option_name(config_key):
    """Convert a config key such as ``vlm_frame_mode`` to a CLI option."""
    return "--" + config_key.replace("_", "-")


def _timestamped_logdir(logdir):
    """Append the run creation time to a configured log directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{os.path.normpath(logdir)}_{timestamp}"


def _load_runner(task):
    try:
        module_path = TASK_TO_RUNNER[task]
    except KeyError as exc:
        raise ValueError(f"Task {task} not recognized.") from exc
    return importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )

    # Parse only --config first so that the complete set of CLI options can be
    # generated from both the selected config and the runner signature.
    bootstrap_args, _ = parser.parse_known_args()

    with open(bootstrap_args.config, "r") as f:
        config = yaml.safe_load(f)

    runner_module = _load_runner(config["task"])
    runner_params = inspect.signature(runner_module.run_training_loop).parameters
    cli_keys = set(config) | set(runner_params)

    for key in sorted(cli_keys):
        if key == "config" or key == "kwargs":
            continue
        parser.add_argument(
            _cli_option_name(key),
            dest=key,
            type=_parse_cli_value,
            default=argparse.SUPPRESS,
            metavar="VALUE",
            help=f"Override config value: {key}",
        )

    parser.add_argument("-h", "--help", action="help", help="Show this help message")

    args = parser.parse_args()
    cli_overrides = vars(args)
    cli_overrides.pop("config", None)
    config.update(cli_overrides)

    # Give every run its own timestamped log directory without modifying the
    # YAML file. This also ensures runners with an explicitly supplied
    # warmup_dir can still open files in the log directory.
    config["logdir"] = _timestamped_logdir(config["logdir"])
    os.makedirs(config["logdir"], exist_ok=True)
    print(f"[BMPS] Log directory: {config['logdir']}")

    runner_module.run_training_loop(**config)


if __name__ == "__main__":
    main()
