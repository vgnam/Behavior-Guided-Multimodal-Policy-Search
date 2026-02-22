"""
Adaptive Visual Guidance for ProPS-V

This module implements the visual guidance schedule using exponential decay:
λ_t = 0.995^t

Where λ_t controls VLM invocation probability (not prompt content).
The LLM receives a fixed, detailed instruction on how to balance visual and numerical feedback,
rather than phase-specific instructions.
"""

import numpy as np
import yaml


class AdaptiveVisualGuidance:
    """
    Manages the VLM invocation schedule for ProPS-V using exponential decay.
    
    Lambda (λ_t = 0.995^t) controls the probability of invoking VLM for visual analysis.
    - t=0: λ ≈ 1.0 (100% VLM calls)
    - t=100: λ ≈ 0.61 (61% VLM calls)
    - t=200: λ ≈ 0.37 (37% VLM calls)
    - t=300: λ ≈ 0.22 (22% VLM calls)
    - t=400: λ ≈ 0.13 (13% VLM calls)
    
    The LLM always receives the same detailed instruction on how to use visual and numerical feedback,
    regardless of iteration. The LLM makes its own decisions based on data quality and context.
    """
    
    def __init__(
        self, 
        decay_horizon: int = 100,
        min_lambda: float = 0.0,
        max_lambda: float = 1.0,
        prompts_config_path: str = "agent/policy/templates/vlm_prompts.yaml"
    ):
        """
        Initialize adaptive visual guidance scheduler.
        
        Args:
            decay_horizon: [DEPRECATED] Not used with exponential decay. Kept for backward compatibility.
            min_lambda: Minimum lambda value (floor for exponential decay)
            max_lambda: Maximum lambda value (initial value at t=0)
            prompts_config_path: Path to VLM prompts YAML configuration file
        """
        self.T_decay = decay_horizon  # Kept for backward compatibility
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.current_iteration = 0
        
        # Load prompts from YAML configuration
        self.prompts = self._load_prompts(prompts_config_path)
    
    def _load_prompts(self, config_path: str) -> dict:
        """
        Load VLM prompts from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary of prompt templates
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"[WARNING] VLM prompts config not found at {config_path}, using defaults")
            return self._get_default_prompts()
        except Exception as e:
            print(f"[WARNING] Error loading VLM prompts config: {e}, using defaults")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> dict:
        """Return default prompts if YAML file is not available."""
        return {
            'adaptive_visual_guidance': {
                'phases': {
                    'early_exploration': {'threshold': 0.75},
                    'balanced_guidance': {'threshold': 0.5},
                    'numerical_emphasis': {'threshold': 0.25},
                    'fine_tuning': {'threshold': 0.0}
                }
            }
        }
        
    def get_lambda(self, iteration: int = None) -> float:
        """
        Compute λ_t for given iteration using exponential decay.
        
        Args:
            iteration: Current iteration (uses internal counter if None)
            
        Returns:
            λ_t = 0.995^t, clamped to [min_lambda, max_lambda]
        """
        if iteration is None:
            iteration = self.current_iteration
            
        # Exponential decay: λ_t = 0.995^t
        lambda_t = self.max_lambda * (0.995 ** iteration)
        
        # Clamp to min_lambda
        lambda_t = max(self.min_lambda, lambda_t)
        
        return lambda_t
    
    def should_invoke_vlm(self, iteration: int = None, random_state: np.random.RandomState = None) -> bool:
        """
        Determine whether to invoke VLM based on λ_t probability.
        
        Args:
            iteration: Current iteration
            random_state: Random state for reproducibility
            
        Returns:
            True if VLM should be invoked, False otherwise
        """
        lambda_t = self.get_lambda(iteration)
        
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Invoke with probability λ_t
        return random_state.rand() < lambda_t
    
    def get_guidance_phase_description(self, iteration: int = None) -> str:
        """
        Get human-readable description of current guidance phase.
        
        Args:
            iteration: Current iteration
            
        Returns:
            Phase description string
        """
        lambda_t = self.get_lambda(iteration)
        return f"Visual Guidance Active (λ={lambda_t:.2f}, VLM invocation probability: {lambda_t*100:.0f}%)"
    
    def get_prompt_instruction(self, iteration: int = None) -> str:
        """
        Get the visual guidance instruction for LLM.
        
        This returns a fixed, detailed instruction on how to use visual and numerical feedback.
        The instruction doesn't change with iteration - LLM decides the balance based on data quality.
        
        Args:
            iteration: Current iteration (unused but kept for compatibility)
            
        Returns:
            Instruction string to include in LLM prompt
        """
        try:
            return self.prompts['visual_guidance']['instruction']
        except KeyError:
            # Fallback if YAML structure changed
            return """
# How to Use Visual and Numerical Feedback:

Use your judgment to balance visual analysis and numerical rewards:
- Prioritize visual feedback when behavior is clearly wrong
- Prioritize numerical feedback when behavior is good and you're optimizing
- Use both sources synergistically when available
"""
    
    def increment_iteration(self):
        """Increment the internal iteration counter."""
        self.current_iteration += 1
    
    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_iteration = 0
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the guidance schedule.
        
        Returns:
            Dictionary with schedule statistics
        """
        lambda_current = self.get_lambda()
        
        # Calculate iterations until pure numerical (λ < 0.01, i.e., 1% threshold)
        threshold = 0.01
        if lambda_current > threshold:
            iterations_until_pure = int(np.ceil(
                np.log(threshold / self.max_lambda) / np.log(0.995)
            )) - self.current_iteration
        else:
            iterations_until_pure = 0
        
        return {
            'current_iteration': self.current_iteration,
            'current_lambda': lambda_current,
            'decay_rate': 0.995,
            'phase': self.get_guidance_phase_description(),
            'vlm_invocation_probability': lambda_current,
            'iterations_until_pure_numerical': max(0, iterations_until_pure),
            'expected_vlm_calls_next_100': sum(0.995 ** (self.current_iteration + i) for i in range(100))
        }
