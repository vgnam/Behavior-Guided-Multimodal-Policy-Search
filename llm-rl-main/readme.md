# Prompted Policy Search (ProPS, ProPS+, BMPS)

This repository contains the official implementation for the NeurIPS paper:

> **Behavior-Guided Multimodal Policy Search: Leveraging Vision-Language Models for Reinforcement Learning**

## Methods

This codebase implements three methods:

- **ProPS** (`props`): Uses LLMs to directly optimize a reinforcement learning policy via numerical parameter updates.
- **ProPS+** (`propsp`): Extends ProPS by incorporating domain-specific descriptions as semantic context in the prompt to guide optimization.
- **BMPS** (`bmps`): Behavior Landscape-guided Multi-modal Policy Search that integrates vision-language model feedback with policy optimization for visual RL tasks.

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

# BMPS with a 2x32 MLP and 32-D latent random-subspace search
python main.py --config configs/swimmer/swimmer_bmps_mlp.yaml
```

Every problem that has a `*_bmps.yaml` config also has a matching
`*_bmps_mlp.yaml` config. Discrete-state variants one-hot encode the state and
select the action with `argmax`; continuous variants consume the observation
vector directly.

## Quick Start

Run BMPS on FrozenLake (default `config.yaml`):

```bash
cd llm-rl-main
python main.py --config config.yaml
```

Any config/runner parameter can be overridden from the command line without
editing the config file. Use kebab-case names for CLI options:

```bash
python main.py --config configs/swimmer/swimmer_bmps.yaml \
  --vlm-frame-mode stacking \
  --num-episodes 400 \
  --enable-vision true \
  --hidden-sizes '[16, 16]'
```

Values use YAML syntax, so booleans, numbers, `null`, lists, and dictionaries
such as `--env-kwargs '{"flatten_observation": true}'` are supported. If an
option is omitted, the value from the config is used. The available VLM frame
modes are `individual`, `stacking`, and `overlay`.

Each run automatically gets a timestamped log directory, for example
`logs/swimmer_bmps_20260726_143015`.

## Training a robosuite PPO baseline

The robosuite wrappers expose low-dimensional state observations and discrete
Cartesian / gripper actions. Train the Stable-Baselines3 PPO baseline with
Python 3.10 from the `llm-rl-main` directory:

```bash
# Lift (64-D state, 9 actions)
py -3.10 train_robosuite_ppo.py --task lift --reward-mode sparse \
  --total-timesteps 1000000

# Door (64-D state, 15 actions)
py -3.10 train_robosuite_ppo.py --task door --reward-mode sparse \
  --total-timesteps 2000000

# Easier single-object PickPlace sanity check (64-D state, 15 actions)
py -3.10 train_robosuite_ppo.py --task pick-place-can --reward-mode sparse \
  --total-timesteps 2000000

# Full four-object PickPlace (106-D state, 15 actions)
py -3.10 train_robosuite_ppo.py --task pick-place --reward-mode dense \
  --total-timesteps 5000000
```

Use `--check-env` for an SB3 compatibility check before training. PPO
hyperparameters can be overridden directly, for example
`--hidden-sizes 128,128 --n-steps 4096 --batch-size 128`. Each run writes to a
timestamped directory under `logs/` and saves checkpoints, evaluation results,
the final model, and the observation-normalization statistics. Full PickPlace
with sparse rewards is substantially harder, so start with `pick-place-can` or
dense rewards when validating the setup.

## Environments

The benchmark includes a diverse set of RL domains spanning discrete and continuous control, vision-based tasks, and robotic manipulation. All tasks use the standard [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API. Custom environments are provided under `./envs/`.

## LLM APIs

Our implementation supports standard LLM APIs. Please configure your API keys via environment variables before running experiments:

- `OPENROUTER_API_KEY`
- `NVIDIA_NIM_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

## Scaling BMPS to Many Policy Parameters

For a policy vector `theta` with `D` parameters, direct prompting requires the
LLM to generate all `D` values. BMPS can instead optimize a local latent vector
`z` with `d << D` using an orthonormal random subspace.

At iteration `t`, construct the projection matrix as

```text
G_t[i,j] ~ Normal(0, 1),    G_t in R^(D x d)
A_t = qr(G_t),              A_t^T A_t = I_d
```

The LLM outputs only `z_t in R^d`. It is decoded to the full policy by

```text
theta_(t+1) = clip(theta_t + alpha A_t z_t, -6, 6).
```

Here `alpha` controls the full-space update magnitude. Because the columns of
`A_t` are orthonormal, `||A_t z_t||_2 = ||z_t||_2` before clipping. Historical
policy `theta_i` is shown to the LLM through its least-squares projection

```text
z_hat_i = clip(A_t^T (theta_i - theta_t) / alpha, -6, 6).
```

Neighborhood sampling also happens in latent space. For every coordinate,

```text
k_j ~ Poisson(lambda),      s_j ~ Uniform({-1, +1})
z_neighbor[j] = h s_j k_j
theta_neighbor = clip(theta_anchor + alpha A_t z_neighbor, -6, 6),
```

where `h = neighbor_step`. CURRENT, BEST, and WORST use the same `A_t` in one
iteration; `z=0` denotes each neighborhood's own anchor. The final LLM proposal
is always decoded around the CURRENT policy.

Enable it in a continuous BMPS config:

```yaml
optimization_mode: latent       # direct or latent
latent_dim: 8                    # d; automatically capped at D
projection_seed: 42
projection_scale: 1.0           # alpha
projection_refresh_interval: 20 # 0 keeps A fixed; N refreshes every N iterations
```

Start with `d` around `sqrt(D)` to `2*sqrt(D)`. Increase `d` when the search is
too constrained; decrease it when prompts are too large or updates are noisy.


## License

This project is released under the MIT License.
