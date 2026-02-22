"""
Vision-Language Model Analyzer for ProPS-V

This module provides VLM integration for analyzing episode frames
and generating diagnostic feedback for policy optimization.
Prompts are loaded from Jinja2 .j2 template files.
Uses NVIDIA API (meta/llama-4-scout-17b-16e-instruct) via requests.
"""

import base64
import io
import time
import os
import requests
import json
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from jinja2 import Environment, FileSystemLoader


class VLMAnalyzer:
    """
    Analyzes episode frames using Vision-Language Models to provide
    visual diagnostic feedback for policy search.
    Uses NVIDIA API (meta/llama-4-scout-17b-16e-instruct) via requests.
    """
    
    # NVIDIA API Configuration
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    NVIDIA_API_KEY = "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"
    NVIDIA_MODEL = "mistralai/mistral-large-3-675b-instruct-2512"

    def __init__(
        self,
        vlm_model_name: str = "meta/llama-4-scout-17b-16e-instruct",
        max_retries: int = 3,
        timeout: int = 60,
        template_dir: str = "agent/policy/templates",
        enable_reasoning: bool = False,
    ):
        """
        Initialize VLM analyzer using NVIDIA API.

        Args:
            vlm_model_name: Name of the VLM model (ignored, uses NVIDIA model)
            max_retries: Maximum number of retry attempts on failure
            timeout: Timeout in seconds for API calls
            template_dir: Directory containing Jinja2 .j2 prompt templates
            enable_reasoning: If True, prepend chain-of-thought instruction to the prompt
        """
        self.vlm_model_name = self.NVIDIA_MODEL  # Always use NVIDIA model
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_reasoning = enable_reasoning

        self._jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )
        
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
    
    def _call_nvidia_api(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> tuple[str, float]:
        """
        Call NVIDIA API with streaming support.
        
        Args:
            messages: List of message dicts with role and content
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (response_text, api_time)
        """
        headers = {
            "Authorization": f"Bearer {self.NVIDIA_API_KEY}",
            "Accept": "text/event-stream"
        }
        
        payload = {
            "model": self.NVIDIA_MODEL,
            "messages": messages,
            "max_tokens": 512,
            "temperature": temperature,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True
        }
        
        api_start_time = time.time()
        response_text = ""
        
        try:
            response = requests.post(
                self.NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Parse streaming response
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    response_text += delta["content"]
                        except json.JSONDecodeError:
                            continue
            
            api_time = time.time() - api_start_time
            return response_text.strip(), api_time
            
        except requests.exceptions.RequestException as e:
            api_time = time.time() - api_start_time
            raise e
    
    def _call_nvidia_api_simple(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> tuple[str, float]:
        """
        Call NVIDIA API without streaming (fallback).
        
        Args:
            messages: List of message dicts with role and content
            temperature: Temperature for sampling
            
        Returns:
            Tuple of (response_text, api_time)
        """
        headers = {
            "Authorization": f"Bearer {self.NVIDIA_API_KEY}",
            "Accept": "application/json"
        }
        
        payload = {
            "model": self.NVIDIA_MODEL,
            "messages": messages,
            "max_tokens": 512,
            "temperature": temperature,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False
        }
        
        api_start_time = time.time()
        
        try:
            response = requests.post(
                self.NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
            api_time = time.time() - api_start_time
            
            return response_text, api_time
            
        except requests.exceptions.RequestException as e:
            api_time = time.time() - api_start_time
            raise e

    def create_analysis_prompt(
        self,
        frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool,
    ) -> str:
        """
        Render the VLM analysis prompt from vlm_analysis_prompt.j2.

        Args:
            frames: List of sampled frame dicts
            env_description: Description of the environment
            episode_reward: Total episodic reward
            terminated_early: Whether episode terminated early (failure)

        Returns:
            Rendered prompt string
        """
        template = self._jinja_env.get_template("vlm_analysis_prompt.j2")
        return template.render(
            env_description=env_description,
            episode_reward=episode_reward,
            terminated_early=terminated_early,
            frames=frames,
        )
    
    def analyze_frames(
        self,
        frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool = False
    ) -> tuple[str, float]:
        """
        Analyze sampled episode frames using VLM and return diagnostic feedback.

        Args:
            frames: List of frame dicts with 'frame' key
            env_description: Description of the environment
            episode_reward: Total episodic reward
            terminated_early: Whether episode terminated early

        Returns:
            Tuple of (analysis_text, api_time)
        """
        # Create text prompt
        text_prompt = self.create_analysis_prompt(
            frames,
            env_description,
            episode_reward,
            terminated_early
        )

        # Build message with inline images (NVIDIA format)
        image_tags = ""
        for frame_data in frames:
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                image_tags += f'<img src="data:image/png;base64,{img_base64}" /> '

        messages = [
            {
                "role": "user",
                "content": f"{text_prompt} {image_tags}"
            }
        ]
        
        # Query VLM
        for attempt in range(self.max_retries):
            try:
                analysis, api_time = self._call_nvidia_api(messages, temperature=0.7)
                return analysis, api_time
                
            except Exception as e:
                print(f"[VLM ERROR] Attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    return f"VLM analysis failed after {self.max_retries} attempts: {e}", 0.0
                time.sleep(5)

        return "VLM analysis unavailable", 0.0

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
            frames_current: Sampled frames from current trajectory
            frames_previous: Sampled frames from previous trajectory
            reward_current: Reward of current trajectory
            reward_previous: Reward of previous trajectory
            env_description: Environment description

        Returns:
            Tuple of (comparative_analysis, api_time)
        """
        template = self._jinja_env.get_template("vlm_comparison_prompt.j2")
        prompt = template.render(
            env_description=env_description,
            reward_previous=reward_previous,
            reward_current=reward_current,
        )
        
        # Build message with inline images (NVIDIA format)
        prev_images = ""
        for frame_data in frames_previous[:3]:
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                prev_images += f'<img src="data:image/png;base64,{img_base64}" /> '

        curr_images = ""
        for frame_data in frames_current[:3]:
            if 'frame' in frame_data and frame_data['frame'] is not None:
                img_base64 = self.frame_to_base64(frame_data['frame'])
                curr_images += f'<img src="data:image/png;base64,{img_base64}" /> '

        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nPrevious trajectory frames: {prev_images}\nCurrent trajectory frames: {curr_images}"
            }
        ]
        
        # Query VLM
        num_prev = sum(1 for f in frames_previous[:3] if f.get('frame') is not None)
        num_curr = sum(1 for f in frames_current[:3] if f.get('frame') is not None)
        print(f"[VLM] Comparison call | prev_frames={num_prev} reward={reward_previous:.2f} | curr_frames={num_curr} reward={reward_current:.2f}")
        for attempt in range(self.max_retries):
            try:
                analysis, api_time = self._call_nvidia_api(messages, temperature=0.7)
                print(f"[VLM] Comparison response received in {api_time:.1f}s ({len(analysis)} chars)")
                return analysis, api_time
            except Exception as e:
                print(f"[VLM ERROR] Comparison attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    return f"VLM comparison failed after {self.max_retries} attempts: {e}", 0.0
                time.sleep(5)

        return "VLM comparison unavailable", 0.0
