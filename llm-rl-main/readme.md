# ProPS and ProPS+

This repo serves as the code base for ProPS and ProPS+.

ProPS uses LLMs to directly optimize a reinforcement learning policy.

ProPS+ is based on ProPS, but also adds the domain description as a context in the prompt, as a optimization guidance.

All the configuration files for each experiments are in the ./configs folder. There are 15 RL domains involved, each domain with a ProPS and ProPS+ training configuration.

In order to run an experiment, please run `python main.py --config <configuration_file>`.

For a concrete ProPS-V walkthrough on the maze task used by the default `config.yaml`, see `PROPSV_README.md`. The shortest path is:

`cd llm-rl-main`

`python -m pip install -r requirements.txt`

`python -m pip install -e .\envs\gym-maze-master`

`python main.py --config configs/maze/maze_propsv.yaml`

## OpenAI-ES Baseline

An OpenAI-ES baseline (linear policy, antithetic sampling, rank-normalized
updates) is available via:

- task: `cont_space_openai_es`
- example config: `configs/acrobot/acrobot_openai_es.yaml`

Implementation note:

- The ES update flow follows the core equations/patterns from the OpenAI
	evolution-strategies-starter repository (centered-rank return processing,
	weighted perturbation aggregation, optimizer update step).
- OpenAI-ES configs are aligned to the same `num_episodes` values as the
	corresponding ProPS+ (`*_propsp.yaml`) configs for each problem.

Run it with:

`python main.py --config configs/acrobot/acrobot_openai_es.yaml`

# Install the requirements

## RL Tasks

- The RL tasks are based on gymnasium. Please install according to `https://github.com/Farama-Foundation/Gymnasium`
- There are 2 customized environments in the folders `./envs/gym-maze-master` and `./envs/gym-navigation-main`. If you want to train the maze or navigation agent, please pip install the packages.
- Grid2Op is supported through the local wrapper `Grid2OpCase14-v0` and configs in `./configs/grid2op/`. On first use, Grid2Op may download the `l2rpn_case14_sandbox` dataset.
- robosuite is supported through the local wrapper `RoboSuiteLiftDiscrete-v0` and configs in `./configs/robosuite/`. This benchmark uses the `Lift` task with a Panda robot and a small discrete action set.
- fancy_gym environments are supported via Gymnasium registration (auto-imported in `main.py`) with configs split by problem folders (e.g. `./configs/fancy_simple_reacher/`, `./configs/fancy_long_simple_reacher/`, `./configs/fancy_reacher5d/`).
- stable_gym environments can be registered without importing `stable_gym.__init__` (see `main.py`) for compatibility. This now includes `stable_gym/MinitaurBulletCost-v1` and `stable_gym/FetchReachCost-v1`.

### Fancy Gym quick examples

Install dependency (already listed in `requirements.txt`):

`pip install fancy_gym`

Run a few included Fancy Gym problems:

`python main.py --config configs/fancy_simple_reacher/fancy_simple_reacher_propsp.yaml`

`python main.py --config configs/fancy_long_simple_reacher/fancy_long_simple_reacher_propsp.yaml`

`python main.py --config configs/fancy_reacher5d/fancy_reacher5d_propsp.yaml`

### Stable Gym cost-task examples

Install dependency (already listed in `requirements.txt`):

`pip install stable-gym`

Run Minitaur and FetchReachCost with OpenAI-ES baseline configs:

`python main.py --config configs/minitaur/minitaur_openai_es.yaml`

`python main.py --config configs/fetch_reach_cost/fetch_reach_cost_openai_es.yaml`

## The LLM APIs

We utilized the standard Google Gemini, Openai, and Anthropic APIs. Please install the packages accordingly.

- `https://ai.google.dev/gemini-api/docs`
- `https://platform.openai.com/docs/overview`
- `https://docs.anthropic.com/en/release-notes/api`
