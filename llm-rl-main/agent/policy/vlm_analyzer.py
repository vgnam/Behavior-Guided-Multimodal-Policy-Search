"""
Vision-Language Model Analyzer for ProPS-V

This module provides VLM integration for analyzing critical frames
and generating diagnostic feedback for policy optimization.
"""

import base64
import io
import time
import yaml
import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from litellm import completion


class VLMAnalyzer:
    """
    Analyzes episode frames using Vision-Language Models to provide
    visual diagnostic feedback for policy search.
    """
    
    def __init__(
        self,
        vlm_model_name: str = "gpt-4o",
        max_retries: int = 3,
        timeout: int = 60,
        prompts_config_path: str = "agent/policy/templates/vlm_prompts.yaml"
    ):
        """
        Initialize VLM analyzer.
        
        Args:
            vlm_model_name: Name of the VLM model to use
            max_retries: Maximum number of retry attempts on failure
            timeout: Timeout in seconds for API calls
            prompts_config_path: Path to VLM prompts YAML configuration file
        """
        self.vlm_model_name = vlm_model_name
        self.max_retries = max_retries
        self.timeout = timeout
        
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
            'vlm_analyzer': {
                'analysis_prompt': {
                    'system_instruction': "You are an expert RL policy analyst.",
                    'task_instruction': "Analyze the frames and provide insights."
                }
            }
        }
        
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert numpy frame to base64-encoded string.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
            
        Returns:
            Base64-encoded image string
        """
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Encode to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def create_analysis_prompt(
        self,
        critical_frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool
    ) -> str:
        """
        Create the text prompt for VLM analysis.
        
        Args:
            critical_frames: List of critical frame dictionaries
            env_description: Description of the environment
            episode_reward: Total episodic reward
            terminated_early: Whether episode terminated early (failure)
            
        Returns:
            Prompt string for VLM
        """
        config = self.prompts['vlm_analyzer']['analysis_prompt']
        
        # Build prompt from YAML templates
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(config['system_instruction'])
        prompt_parts.append("")
        
        # Environment section
        prompt_parts.append(config['environment_section'].format(
            env_description=env_description
        ))
        prompt_parts.append("")
        
        # Episode summary section
        termination_status = (
            config['termination_status']['early'] if terminated_early 
            else config['termination_status']['normal']
        )
        prompt_parts.append(config['episode_summary_section'].format(
            episode_reward=episode_reward,
            termination_status=termination_status,
            num_frames=len(critical_frames)
        ))
        prompt_parts.append("")
        
        # Critical frames intro
        prompt_parts.append(config['critical_frames_intro'].format(
            num_frames=len(critical_frames)
        ))
        prompt_parts.append("")
        
        # Task instruction
        prompt_parts.append(config['task_instruction'])
        
        # Add frame-specific context
        for i, frame_data in enumerate(critical_frames):
            frame_info = config['frame_info_template'].format(
                frame_num=i+1,
                timestep=frame_data['timestep'],
                frame_type=frame_data['frame_type'],
                reward=frame_data['reward']
            )
            prompt_parts.append(frame_info)
            
            if 'state' in frame_data:
                prompt_parts.append(config['frame_state_template'].format(
                    state=frame_data['state']
                ))
            if 'action' in frame_data:
                prompt_parts.append(config['frame_action_template'].format(
                    action=frame_data['action']
                ))
        
        return "\n".join(prompt_parts)
    
    def analyze_critical_frames(
        self,
        critical_frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool = False
    ) -> tuple[str, float]:
        """
        Analyze critical frames using VLM and return diagnostic feedback.
        
        Args:
            critical_frames: List of critical frame dictionaries with 'frame' key
            env_description: Description of the environment
            episode_reward: Total episodic reward  
            terminated_early: Whether episode terminated early
            
        Returns:
            Tuple of (analysis_text, api_time)
        """
        # Create text prompt
        text_prompt = self.create_analysis_prompt(
            critical_frames,
            env_description,
            episode_reward,
            terminated_early
        )
        
        # Prepare messages with images
        messages = [{"role": "user", "content": []}]
        
        # Add text
        messages[0]["content"].append({
            "type": "text",
            "text": text_prompt
        })
        
        # Add images
        for i, frame_data in enumerate(critical_frames):
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
        
        # Query VLM
        for attempt in range(self.max_retries):
            try:
                api_start_time = time.time()
                response = completion(
                    model=self.vlm_model_name,
                    messages=messages,
                    temperature=0.7,
                    timeout=self.timeout,
                )
                api_time = time.time() - api_start_time
                
                analysis = response["choices"][0]["message"]["content"]
                return analysis, api_time
                
            except Exception as e:
                error_msg = self.prompts['error_messages']['vlm_error_attempt'].format(
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=e
                )
                print(error_msg)
                if attempt == self.max_retries - 1:
                    final_error = self.prompts['error_messages']['vlm_analysis_failed'].format(
                        max_retries=self.max_retries,
                        error=str(e)
                    )
                    return final_error, 0.0
                time.sleep(5)
        
        return self.prompts['error_messages']['vlm_analysis_unavailable'], 0.0
    
    def analyze_trajectory_comparison(
        self,
        frames_current: List[Dict[str, Any]],
        frames_previous: List[Dict[str, Any]],
        reward_current: float,
        reward_previous: float,
        env_description: str
    ) -> tuple[str, float]:
        """
        Compare two trajectories visually to identify improvements or regressions.
        
        Args:
            frames_current: Critical frames from current trajectory
            frames_previous: Critical frames from previous trajectory
            reward_current: Reward of current trajectory
            reward_previous: Reward of previous trajectory
            env_description: Environment description
            
        Returns:
            Tuple of (comparative_analysis, api_time)
        """
        config = self.prompts['vlm_analyzer']['trajectory_comparison_prompt']
        
        # Build prompt from YAML templates
        prompt_parts = []
        prompt_parts.append(config['system_instruction'])
        prompt_parts.append("")
        prompt_parts.append(config['environment_section'].format(
            env_description=env_description
        ))
        prompt_parts.append("")
        prompt_parts.append(config['comparison_section'].format(
            reward_previous=reward_previous,
            reward_current=reward_current,
            reward_change=reward_current - reward_previous
        ))
        prompt_parts.append("")
        prompt_parts.append(config['task_instruction'])
        
        prompt = "\n".join(prompt_parts)
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        # Add previous frames
        for frame_data in frames_previous[:3]:  # Limit to 3 frames
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                messages[0]["content"].append({
                    "type": "image_url",  
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
        
        # Add current frames
        for frame_data in frames_current[:3]:  # Limit to 3 frames
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
        
        # Query VLM
        for attempt in range(self.max_retries):
            try:
                api_start_time = time.time()
                response = completion(
                    model=self.vlm_model_name,
                    messages=messages,
                    temperature=0.7,
                    timeout=self.timeout,
                )
                api_time = time.time() - api_start_time
                
                analysis = response["choices"][0]["message"]["content"]
                return analysis, api_time
                
            except Exception as e:
                error_msg = self.prompts['error_messages']['vlm_comparison_error'].format(
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=e
                )
                print(error_msg)
                if attempt == self.max_retries - 1:
                    final_error = self.prompts['error_messages']['comparison_failed'].format(
                        error=str(e)
                    )
                    return final_error, 0.0
                time.sleep(5)
        
        return self.prompts['error_messages']['comparison_unavailable'], 0.0
