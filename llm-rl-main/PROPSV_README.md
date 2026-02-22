# ProPS-V: Vision-Guided Prompted Policy Search

This directory contains the implementation of **ProPS-V**, a novel reinforcement learning method that combines:

1. **Numerical Optimization** (from ProPS)
2. **Semantic Reasoning** (from ProPS+)
3. **Vision-Guided Feedback** (new in ProPS-V)

## Overview

ProPS-V leverages Vision-Language Models (VLMs) to provide visual diagnostic feedback during policy search. Unlike traditional RL methods that rely solely on scalar rewards, ProPS-V can understand visual failure modes, spatial relationships, and behavioral patterns that are difficult to express numerically.

### Key Features

- **Critical Frame Sampling (Eq. 2)**: Intelligently selects pivotal moments from trajectories (initial state, terminal state, failures, significant transitions)
- **Adaptive Visual Guidance Annealing (Eq. 3)**: Gradually reduces VLM usage over time to balance exploration with computational efficiency
- **Vision-Language Integration (Eq. 4)**: Combines numerical rewards with visual diagnostics for more informed policy updates

## Implementation Structure

```
agent/
├── llm_num_optim_linear_policy_vision.py  # Main ProPS-V agent
├── policy/
│   ├── critical_frame_sampler.py          # Eq. 2: Critical frame sampling
│   ├── adaptive_visual_guidance.py        # Eq. 3: λ_t annealing schedule
│   ├── vlm_analyzer.py                    # VLM integration for visual feedback
│   ├── llm_brain_linear_policy.py         # Updated with vision method
│   └── templates/
│       └── num_optim_vision.j2            # ProPS-V prompt template
runner/
└── llm_num_optim_vision_runner.py         # Training loop for ProPS-V
configs/
└── [environment]/
    └── [environment]_propsv.yaml          # Configuration files
```

## Usage

### Running ProPS-V

```bash
# CartPole with ProPS-V
python main.py --config configs/cartpole/cartpole_propsv.yaml

# MountainCar Continuous with ProPS-V
python main.py --config configs/mountaincarcontinuous/mountaincar_continuous_propsv.yaml

# InvertedPendulum with ProPS-V
python main.py --config configs/invertedpendulum/invertedpendulum_propsv.yaml
```

### Configuration Parameters

ProPS-V adds the following parameters to the standard configuration:

```yaml
# Vision-specific parameters
vlm_model_name: gpt-4o              # Vision-Language Model
decay_horizon: 100                  # T_decay: iterations to phase out visual guidance
reward_change_threshold: 0.1        # δ: threshold for detecting transition states
enable_vision: true                 # Enable/disable vision-guided feedback
render_mode: rgb_array              # Required for frame capture
```

### Key Configuration Variables

- **decay_horizon** (`T_decay`): Controls how quickly visual guidance is phased out
  - Lower values (e.g., 50): Faster transition to pure numerical optimization
  - Higher values (e.g., 200): Longer vision-guided exploration phase
  
- **reward_change_threshold** (`δ`): Sensitivity for detecting important transitions
  - Lower values (e.g., 0.05): More frames marked as transitions
  - Higher values (e.g., 0.5): Only major reward changes captured

- **vlm_model_name**: Vision-Language Model for visual analysis
  - `gpt-4o`: Recommended for best vision understanding
  - `gpt-4-vision-preview`: Alternative option
  - Other multimodal models supported by LiteLLM

## How ProPS-V Works

### 1. Critical Frame Sampling (Eq. 2)

Instead of processing entire trajectories, ProPS-V samples critical frames:

$$\mathcal{I}_{\text{crit}} = \mathcal{I}_{\text{init}} \cup \mathcal{I}_{\text{term}} \cup \mathcal{I}_{\text{fail}} \cup \mathcal{I}_{\text{trans}}$$

- **I_init**: Initial state (t=0)
- **I_term**: Terminal state (last frame)
- **I_fail**: Failure states (early termination)
- **I_trans**: Transitions with |ΔR_t| > δ

### 2. Adaptive Visual Guidance (Eq. 3)

The probability of VLM invocation decreases over time:

$$\lambda_t = \max(0, 1 - \frac{t}{T_{\text{decay}}})$$

**Phase Progression:**
- λ > 0.75: **Early Exploration** - Prioritize visual feedback
- 0.5 < λ ≤ 0.75: **Balanced Guidance** - Mix visual and numerical
- 0.25 < λ ≤ 0.5: **Numerical Emphasis** - Focus on rewards
- λ ≤ 0.25: **Fine Tuning** - Pure numerical optimization

### 3. Vision-Guided Update (Eq. 4)

Policy parameters updated with integrated feedback:

$$\theta \leftarrow \text{LLM}(\Gamma, P, \Psi, \lambda_t)$$

Where:
- Γ: History of parameters and rewards
- P: Prompt template
- Ψ: Visual analysis history
- λ_t: Current guidance coefficient

## Output and Logs

ProPS-V generates the following logs:

```
logs/[environment]_propsv/
├── overall_log.txt              # Training statistics
├── vision_statistics.txt        # VLM invocation tracking
├── visual_analysis.txt          # All VLM analyses
├── warmup/                      # Warmup episodes
├── episode_0/
│   ├── training_rollout.txt     # Episode trajectory
│   ├── parameters.txt           # Policy parameters
│   └── parameters_reasoning.txt # LLM reasoning
└── ...
```

### Vision Statistics

The `vision_statistics.txt` file tracks:
- Iteration number
- λ_t value
- Whether VLM was invoked
- Current guidance phase
- Number of critical frames sampled

### Visual Analysis

The `visual_analysis.txt` file contains VLM reports including:
- Visual observations of agent behavior
- Failure mode analysis
- Success pattern identification
- Parameter adjustment recommendations

## Example VLM Analysis

```
==============================================================
Iteration 15 (λ=0.850)
==============================================================

**Visual Observations**:
The agent starts near the center but quickly moves left. The pole 
begins to tilt right and continues tilting throughout the episode. 
The cart does not apply sufficient force to counteract the tilt.

**Failure Analysis**:
The policy failed because it did not respond strongly enough to 
the pole angle. Visual evidence shows the pole angle exceeding 
the threshold at timestep 8, causing termination.

**Success Patterns**:
N/A - episode failed early

**Parameter Adjustment Recommendations**:
Increase params[4] and params[5] (pole angle to action weights) 
to make the policy more responsive to angular deviations. The 
cart position weights (params[0], params[1]) appear adequately tuned.
```

## Performance Comparison

ProPS-V is designed to:
- **Outperform ProPS** on visually-complex tasks (e.g., locomotion, navigation)
- **Match or exceed ProPS+** by adding visual grounding to semantic reasoning
- **Reduce sample complexity** through better failure diagnosis
- **Improve interpretability** via visual explanations

## API Cost Considerations

- **VLM calls are expensive**: Each call processes multiple images
- **Adaptive annealing reduces cost**: VLM usage decreases as training progresses
- **Typical VLM invocations**: ~50-100 calls for decay_horizon=100 (probabilistic)
- **Frame limit**: Critical sampling reduces frames to 3-10 per trajectory

## Troubleshooting

### No frames captured
- Ensure `render_mode: rgb_array` in config
- Check environment supports rendering
- Verify `enable_vision: true`

### VLM errors
- Check API keys for vision model
- Ensure VLM model supports image input
- Verify image encoding (should be PNG base64)

### High API costs
- Reduce `decay_horizon` for faster phase-out
- Increase `reward_change_threshold` to capture fewer transitions
- Set `enable_vision: false` to run as ProPS+

## Citation

If you use ProPS-V in your research, please cite:

```bibtex
@article{propsv2026,
  title={ProPS-V: Vision-Guided Prompted Policy Search},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2026}
}
```

## Requirements

- Python 3.8+
- gymnasium
- litellm (with vision model support)
- jinja2
- numpy
- PIL/Pillow

Install vision-specific dependencies:
```bash
pip install gymnasium litellm pillow jinja2 numpy
```

## License

[Your License Here]
