# ProPS and ProPS+

This repo serves as the code base for ProPS and ProPS+.

ProPS uses LLMs to directly optimize a reinforcement learning policy.

ProPS+ is based on ProPS, but also adds the domain description as a context in the prompt, as a optimization guidance.

All the configuration files for each experiments are in the ./configs folder. There are 15 RL domains involved, each domain with a ProPS and ProPS+ training configuration.

In order to run an experiment, please run `python main.py --config <configuration_file>`.

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

## The LLM APIs

We utilized the standard Google Gemini, Openai, and Anthropic APIs. Please install the packages accordingly.

- `https://ai.google.dev/gemini-api/docs`
- `https://platform.openai.com/docs/overview`
- `https://docs.anthropic.com/en/release-notes/api`
