# Prompted Policy Search (ProPS, ProPS+, BMPS)

This repository contains the official implementation for the NeurIPS paper:

> **Behavior-Guided Multimodal Policy Search: Leveraging Vision-Language Models for Reinforcement Learning**

## Methods

This codebase implements four methods:

- **ProPS** (`props`): Uses LLMs to directly optimize a reinforcement learning policy via numerical parameter updates.
- **ProPS+** (`propsp`): Extends ProPS by incorporating domain-specific descriptions as semantic context in the prompt to guide optimization.
- **BMPS** (`bmps`): Behavior Landscape-guided Multi-modal Policy Search that integrates vision-language model feedback with policy optimization for visual RL tasks.
- **BMPS-CMA** (`bmps_cma`): Uses periodically sampled temporal-superposition images,
  score-free pairwise VLM comparisons, complete-policy LLM proposals, and a
  Mahalanobis trust region to guide CMA-ES without injecting remote LLM jumps
  into its covariance evolution paths.

## Repository Structure

```
.
├── main.py                           # Entry point for running experiments
├── config.yaml                       # Example configuration (FrozenLake + BMPS)
├── requirements.txt                  # Python dependencies
├── readme.md                         # This file
├── agent/                            # Agent implementations
│   ├── llm_num_optim_linear_policy.py
│   ├── llm_num_optim_q_table.py
│   ├── llm_num_optim_linear_policy_semantics.py
│   ├── llm_num_optim_q_table_semantics.py
│   ├── llm_num_optim_linear_policy_vision.py
│   ├── llm_num_optim_q_table_vision.py
│   └── policy/                       # Policy classes and prompts
│       ├── templates/                # Jinja2 prompt templates
│       └── ...
├── runner/                           # Training loop runners
│   ├── llm_num_optim_runner.py       # ProPS runner
│   ├── llm_num_optim_semantics_runner.py  # ProPS+ runner
│   └── llm_num_optim_vision_runner.py     # BMPS runner
├── world/                            # Environment wrappers
│   ├── continuous_space_general_world.py
│   └── discrete_state_general_world.py
└── configs/                          # Experiment configurations per domain
    ├── <domain>/<domain>_props.yaml
    ├── <domain>/<domain>_propsp.yaml
    └── <domain>/<domain>_bmps.yaml
```

## Installation

```bash
pip install -r requirements.txt

# For maze experiments
pip install -e ./envs/gym-maze-master

# For navigation experiments
pip install -e ./envs/gym-navigation-main
```

## Running Experiments

Each RL domain provides three configuration files (one per method):

```bash
# ProPS
python main.py --config configs/taxi/taxi_props.yaml

# ProPS+
python main.py --config configs/taxi/taxi_propsp.yaml

# BMPS
python main.py --config configs/frozenlake/frozenlake_bmps.yaml

# BMPS-CMA (continuous linear policy)
python main.py --config configs/hopper/hopper_linear_policy_bmps_cma.yaml
```

Every existing continuous-policy BMPS configuration has a sibling named
`*_bmps_cma.yaml` under the same environment directory. Regenerate the complete
set after changing or adding BMPS configs with:

```bash
python scripts/generate_bmps_cma_configs.py
```

### BMPS-CMA generation

BMPS-CMA retains the existing prompt representation
`params[0]: ...; ...; f(params): ...`. The LLM proposes a complete policy. The
proposal is projected into a trust region around the current numerical CMA mean,
then a Gaussian population is sampled and ranked only by environment return.
The initial frame, every `frame_sample_period` step, and the terminal/failure
frame from the first rollout of each candidate are sent as separate images by
default, matching BMPS. Set `stack_trajectory_frames: true` to collapse those
sampled frames into one temporal-superposition image. Frames are retained only
in memory for the VLM request and are not written into episode log directories.
The numerical proposal history contains the warmup policies followed only by
complete policies proposed by the LLM; CMA means and Gaussian candidates are
excluded. Pairwise VLM feedback from a generation is supplied to the LLM on the
next generation, preserving a single-population rollout budget.

## Quick Start

Run BMPS on FrozenLake (default `config.yaml`):

```bash
cd llm-rl-main
python main.py --config config.yaml
```

## Environments

The benchmark includes a diverse set of RL domains spanning discrete and continuous control, vision-based tasks, and robotic manipulation. All tasks use the standard [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API. Custom environments are provided under `./envs/`.

## LLM APIs

Our implementation supports standard LLM APIs. Please configure your API keys via environment variables before running experiments:

- `OPENROUTER_API_KEY`
- `NVIDIA_NIM_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`


## License

This project is released under the MIT License.
