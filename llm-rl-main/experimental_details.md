## D. Experimental Details

### D.1. Problem Taxonomy: Continuous vs. Discrete Spaces

We categorize all benchmark environments along two axes: **state space** (continuous / discrete) and **action space** (continuous / discrete). Table\~\ref{tab:space_taxonomy} summarizes how each algorithm family maps onto these spaces.

| Family | Continuous State | Discrete State | Continuous Action | Discrete Action |
|---|---|---|---|---|
| **Evolution Strategies (Linear)** | Linear policy $\\mathbf{a}=\\mathbf{W}\\mathbf{s}+\\mathbf{b}$ | One-hot state $\\mathbf{s}\\in\\{0,1\\}^{|S|}$ → linear → output logits | Direct output $\\mathbf{a}\\in\\mathbb{R}^{d_a}$ | Output logits → $\\arg\\max$ |
| **Evolution Strategies (Q-Value)** | — | Tabular $Q(s,a)$ vector, $\\pi(s)=\\arg\\max_a \\theta[s,a]$ | — | $\\arg\\max$ over discrete actions |
| **Evolution Strategies (Q-Table Direct)** | — | Tabular lookup $\\pi(s)=\\text{round}(\\theta[s])$ | — | Integer action index |
| **LLM-Guided (PROPSV)** | LLM proposes flat params for linear policy | LLM proposes flat params for Q-table / linear+one-hot | Same as ES linear | Same as ES linear / Q-table |
| **LLM-Guided (PROPSP)** | LLM proposes params with semantic trajectory context | Same, with env description + trajectory semantics | Same | Same |
| **LLM-Guided (PROPS / MLP)** | MLP policy $\\mathbf{a}=\\text{MLP}(\\mathbf{s})$ | One-hot or raw int fed into MLP | Direct output | Output logits → $\\arg\\max$ |

### D.2. Evolution Strategies (ES) Baselines

All ES algorithms share the following evaluation protocol: at each training iteration we evaluate the current candidate policy on $N_{\\text{eval}}=20$ independent rollouts and report the **mean episodic reward**. Training runs for a fixed number of iterations (100–400 depending on the environment). Unless otherwise noted, all ES baselines use a single CPU core (`ncores=1`) and the same random seeding scheme (seeds $0$–$9$ for statistical reporting).

#### D.2.1. Policy Representations

We implement **three policy representations** for ES algorithms, selected via the `task` field in the configuration:

1. **Linear Policy (continuous / discrete).** Parameterized by weight matrix $\\mathbf{W}\\in\\mathbb{R}^{d_s \\times d_a}$ and optional bias $\\mathbf{b}\\in\\mathbb{R}^{d_a}$. For discrete environments (CliffWalking, FrozenLake), raw integer states are converted to **one-hot vectors** $\\mathbf{e}_s\\in\\{0,1\\}^{|S|}$ before the linear projection. The action is selected by $\\arg\\max$ over the output logits.

2. **Q-Value Tabular (discrete-only).** A flat vector $\\theta\\in\\mathbb{R}^{|S|\\cdot|A|}$ representing $Q(s,a)$. The deterministic policy is $\\pi(s)=\\arg\\max_a \\theta[s,a]$. This representation is newly introduced in our work to provide a fair value-based baseline for discrete environments.

3. **Q-Table Direct (discrete-only).** A flat vector $\\theta\\in\\mathbb{R}^{|S|}$ where each entry stores a continuous value that is rounded to the nearest valid action index. This is the original discrete baseline in the codebase.

#### D.2.2. Augmented Random Search (ARS)

ARS samples $N$ random search directions, evaluates positive and negative perturbations, keeps the top-performing $k$ directions, normalizes by reward standard deviation, and performs a gradient step.

| Hyperparameter | CliffWalking | FrozenLake | Pong |
|---|---|---|---|
| `n_directions` | 32 | 32 | 32 |
| `deltas_used` | 16 | 16 | 16 |
| `step_size` | 0.03 | 0.03 | 0.03 |
| `delta_std` | 0.1 | 0.1 | 0.1 |
| `shift` | 0.0 | 0.0 | 0.0 |
| `candidate_evaluation_episodes` | 1 | 1 | 1 |
| `reward_normalization_epsilon` | $10^{-8}$ | $10^{-8}$ | $10^{-8}$ |
| `grad_batch_size` | 500 | 500 | 500 |

**Task assignments:**
- CliffWalking: `dist_state_ars_qvalue` (Q-value tabular, $|S|\\times|A|=192$ params).
- FrozenLake: `dist_state_ars_qvalue` (Q-value tabular, $|S|\\times|A|=64$ params).
- Pong: `cont_space_ars` (linear policy, $d_s=5$, $d_a=3$).

#### D.2.3. Covariance Matrix Adaptation ES (CMA-ES)

We use the `pycma` implementation with a Gaussian search distribution. Covariance type is set to `auto` (full covariance when dimension $\\le 256$, otherwise diagonal).

| Hyperparameter | Value |
|---|---|
| `population_size` | 32 |
| `sigma` | 0.1 |
| `elite_count` | `null` (auto-detected) |
| `candidate_evaluation_episodes` | 1 |
| `covariance_type` | `auto` |
| `full_covariance_max_dim` | 256 |
| `min_sigma` | $10^{-12}$ |

**Task assignments:**
- CliffWalking: `dist_state_cma_es_qvalue` (192 params).
- FrozenLake: `dist_state_cma_es_qvalue` (64 params), plus `dist_state_cma_es_qtable` for comparison (48 params).
- Pong: `cont_space_cma_es` (linear policy).

#### D.2.4. OpenAI-ES

OpenAI-ES uses antithetic Gaussian perturbations and Adam/SGD optimizer on the ES gradient estimate. We use centered-rank return processing.

| Hyperparameter | Value |
|---|---|
| `population_size` | 32 |
| `sigma` / `noise_stdev` | 0.1 |
| `learning_rate` | 0.03 |
| `candidate_evaluation_episodes` | 1 |
| `return_proc_mode` | `centered_rank` |
| `optimizer_type` | `adam` |
| `adam_beta1` | 0.9 |
| `adam_beta2` | 0.999 |
| `adam_epsilon` | $10^{-8}$ |
| `weight_decay` | 0.0 |
| `grad_batch_size` | 500 |

**Task assignments:**
- CliffWalking: `dist_state_openai_es_qvalue` (192 params).
- FrozenLake: `dist_state_openai_es_qvalue` (64 params).
- Pong: `cont_space_openai_es` (linear policy).

#### D.2.5. ($\\mu$, $\\lambda$)-ES

Traditional evolutionary strategy with parent selection, BLX-$\\alpha$ crossover, and Gaussian mutation. The backend is selected automatically: EvoLib $\\rightarrow$ neorl $\\rightarrow$ built-in NumPy fallback.

| Hyperparameter | Value |
|---|---|
| `mu` | 16 |
| `lam` | 32 |
| `sigma` | 0.1 |
| `sigma_decay` | 1.0 |
| `min_sigma` | $10^{-12}$ |
| `param_bound` | 10.0 |
| `cxmode` | `blend` (BLX-$\\alpha$) |
| `alpha` | 0.5 |
| `cxpb` | 0.6 |
| `mutpb` | 0.3 |
| `smin` / `smax` | 0.01 / 0.5 |
| `clip` | `True` |
| `candidate_evaluation_episodes` | 1 |

**Task assignments:**
- CliffWalking: `dist_state_mu_lambda_es_qvalue` (192 params).
- FrozenLake: `dist_state_mu_lambda_es_qvalue` (64 params).
- Pong: `cont_space_mu_lambda_es` (linear policy).

### D.3. LLM-Guided Policy Search (PROPS)

All LLM-guided methods use **GPT-4** (`gpt-4-1106-preview` or `gpt-4o` for vision) accessed via the OpenAI API. The LLM receives a prompt containing (i) a natural-language description of the environment, (ii) a history of previously proposed parameter vectors and their rewards, and (iii) formatting instructions. The LLM responds with a new parameter proposal which is parsed, instantiated as a policy, and evaluated on the environment.

#### D.3.1. PROPSV: Value-Based Numerical Optimization

**Continuous environments.** The LLM directly proposes a flat list of real numbers representing the concatenated parameters of a **linear policy** ($\\mathbf{W}$ and $\\mathbf{b}$). The prompt includes the last $k$ parameter-reward pairs so the LLM can perform in-context numerical optimization.

**Discrete environments.** The LLM can operate in two modes:
1. **Q-Table mode** (original): the LLM proposes $|S|$ integer or floating-point values representing the action index for each state. This is used for small discrete environments such as FrozenLake.
2. **Linear + one-hot mode**: the LLM proposes parameters for a linear policy, but the environment state is first converted to a one-hot vector (same mechanism as the ES linear baselines).

**Prompt templates** are stored in `agent/policy/templates/env_descriptions/{env}.j2`. The output format enforced by regex is:
```
params[0]: 1.23; params[1]: -0.45; ...; params[n]: 0.67
```

| Hyperparameter | Value |
|---|---|
| `llm_model_name` | `gpt-4-1106-preview` |
| `num_evaluation_episodes` | 20 |
| `search_step_size` | 0.1 |
| `max_traj_count` | 1000 (replay buffer size) |

#### D.3.2. PROPSP: Semantics-Enhanced Value Optimization

PROPSP extends PROPSV by injecting **semantic information** into the LLM prompt. Specifically, we append (i) a detailed environment description file (`env_desc_file`), and (ii) trajectory summaries (state-action-reward sequences) from the replay buffer. This gives the LLM contextual understanding of environment dynamics rather than pure black-box numbers.

The policy representation and parsing mechanism are identical to PROPSV. The only difference is the prompt construction function (`llm_update_parameters_num_optim_semantics` vs. `llm_update_parameters_num_optim`).

| Hyperparameter | Value |
|---|---|
| `llm_model_name` | `gpt-4-1106-preview` |
| `env_desc_file` | `env_descriptions/{env}_more_hints.j2` |
| `search_step_size` | 0.1 |

#### D.3.3. PROPS: General MLP-Based Framework

PROPS uses a **Multi-Layer Perceptron (MLP)** as the policy architecture, allowing it to scale beyond linear policies. The LLM proposes a flat parameter vector which is reshaped into MLP weights and biases. Depending on the environment, the MLP can operate in:

- **Continuous action mode**: MLP outputs raw action values $\\mathbf{a}\\in\\mathbb{R}^{d_a}$.
- **Discrete action mode**: MLP outputs logits $\\mathbf{z}\\in\\mathbb{R}^{|A|}$, and the action is selected by $\\arg\\max_a z_a$ (via `np.argmax`).

**Optional extensions:**
- **Intrinsic random projection**: when the MLP parameter count exceeds a threshold, the LLM proposes parameters in a low-dimensional intrinsic space which are projected back to the full space via a fixed random orthogonal matrix.
- **Observation projection**: high-dimensional observations (e.g., vision) can be randomly projected before feeding into the MLP.
- **ProPS-V (vision)**: periodic VLM (GPT-4o) analysis of rendered frames to guide neighborhood exploration.

| Hyperparameter | Value |
|---|---|
| `llm_model_name` | `gpt-4-1106-preview` |
| `hidden_dims` | `[64, 64]` (default MLP) |
| `search_step_size` | 0.1 |
| `intrinsic_rank` | `null` (disabled by default) |
| `is_discrete` | `True` for Atari/gridworld, `False` for MuJoCo |
| `vlm_model_name` | `gpt-4o` (ProPS-V only) |

### D.4. Environment Specifications

| Environment | State Space | Action Space | Max Steps | Reward Range | Task Tag |
|---|---|---|---|---|---|
| CliffWalking-v0 | Discrete (48) | Discrete (4) | 100 | $[-\\infty, -100]$ | `dist_state_*` |
| FrozenLake-v1 | Discrete (16) | Discrete (4) | 100 | $\\{0, 1\\}$ | `dist_state_*` |
| Pong-v0 | Continuous (5-dim features) | Discrete (3) | 1000 | $[-21, 21]$ | `cont_space_*` |

**CliffWalking-v0**: A $4\\times12$ grid world. The agent starts at the bottom-left corner and must reach the bottom-right corner. Each step incurs a reward of $-1$. Falling into the "cliff" (the bottom row excluding start and goal) incurs a penalty of $-100$ and resets the agent to the start.

**FrozenLake-v1**: A $4\\times4$ stochastic grid world (`is_slippery=True`). The agent navigates from the top-left to the bottom-right. Holes cause episode termination with reward $0$; reaching the goal gives reward $+1$.

**Pong-v0**: The agent receives hand-engineered 5-dimensional state features (ball position, velocity, paddle position). The action space is discrete: $\{0, 1, 2\\}$ corresponding to `NOOP`, `UP`, `DOWN`. Episodes last up to 1000 steps.

### D.5. Evaluation Protocol

| Setting | Value |
|---|---|
| Training iterations | 100 (FrozenLake), 300 (CliffWalking), 400 (Pong) |
| Evaluation rollouts per iteration | 20 |
| Independent random seeds | 10 (seeds $0$–$9$) |
| Reported metric | Best mean reward $\\bar{R}=\\max_t \\frac{1}{20}\\sum_{i=1}^{20} R_{t,i}$ |
| Statistical format | Mean $\\pm$ Std over 10 seeds |
| Compute | Single CPU core per run |

For the value-based ES variants, the Q-value vector is optimized directly in the continuous space $\\mathbb{R}^{|S|\\cdot|A|}$, and the policy is derived greedily via $\\arg\\max$ at evaluation time.

### D.6. Software and Reproducibility

- **Language**: Python 3.10+
- **Core packages**: `gymnasium==0.29.1`, `numpy==1.26.0`, `cma==3.3.0`, `pyyaml==6.0.1`
- **LLM API**: OpenAI GPT-4 / GPT-4o via official Python client
- **Code structure**:
  - `runner/`: Training loops for each algorithm family.
  - `agent/`: Policy implementations (linear, Q-table, Q-value, MLP) and LLM brain.
  - `configs/`: YAML experiment configurations.
  - `world/`: Environment wrappers (`ContinualSpaceGeneralWorld`, `DiscreteStateGeneralWorld`).
