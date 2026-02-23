"""
Vision-Language Model Analyzer for ProPS-V

This module provides VLM integration for analyzing episode frames
and generating diagnostic feedback for policy optimization.
Prompts are loaded from Jinja2 .j2 template files.
Uses Google Gen AI SDK (gemini-2.5-flash-lite).
"""

import io
import time
import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from google import genai
from google.genai import types


class VLMAnalyzer:
    """
    Analyzes episode frames using Vision-Language Models to provide
    visual diagnostic feedback for policy search.
    Uses Google Gen AI SDK (gemini-2.5-flash-lite).
    """

    GEMINI_MODEL = "gemini-2.5-flash-lite"

    def __init__(
        self,
        vlm_model_name: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        timeout: int = 60,
        template_dir: str = "agent/policy/templates",
        enable_reasoning: bool = False,
    ):
        """
        Initialize VLM analyzer using Google Gen AI SDK.

        Args:
            vlm_model_name: Name of the VLM model (ignored, uses GEMINI_MODEL)
            max_retries: Maximum number of retry attempts on failure
            timeout: Timeout in seconds for API calls
            template_dir: Directory containing Jinja2 .j2 prompt templates
            enable_reasoning: If True, prepend chain-of-thought instruction to the prompt
        """
        self.vlm_model_name = self.GEMINI_MODEL
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_reasoning = enable_reasoning

        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self._jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            keep_trailing_newline=True,
        )

    def frame_to_png_bytes(self, frame: np.ndarray) -> bytes:
        """
        Convert numpy frame to raw PNG bytes.

        Args:
            frame: RGB frame as numpy array (H, W, 3)

        Returns:
            PNG image bytes
        """
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

    def _build_parts(
        self,
        text_prompt: str,
        image_frames: List[np.ndarray],
    ) -> List[types.Part]:
        """
        Build a list of SDK Part objects from a text prompt and image frames.

        Args:
            text_prompt: The text portion of the message
            image_frames: List of RGB numpy arrays

        Returns:
            List of types.Part (text first, then images)
        """
        parts: List[types.Part] = [types.Part.from_text(text=text_prompt)]
        for frame in image_frames:
            png_bytes = self.frame_to_png_bytes(frame)
            parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
        return parts

    def _call_gemini_api(
        self,
        parts: List[types.Part],
        temperature: float = 0.7,
    ) -> tuple[str, float]:
        """
        Call Gemini via Google Gen AI SDK.

        Args:
            parts: List of types.Part (text + images)
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, api_time)
        """
        api_start_time = time.time()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )

        response = self._client.models.generate_content(
            model=self.GEMINI_MODEL,
            contents=parts,
            config=config,
        )

        api_time = time.time() - api_start_time
        return response.text.strip(), api_time

    def create_analysis_prompt(
        self,
        frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool,
        current_params=None,
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
            current_params=current_params,
        )

    def analyze_frames(
        self,
        frames: List[Dict[str, Any]],
        env_description: str,
        episode_reward: float,
        terminated_early: bool = False,
        current_params=None,
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
            terminated_early,
            current_params=current_params,
        )

        image_frames = [
            fd["frame"]
            for fd in frames
            if "frame" in fd and fd["frame"] is not None
        ]
        parts = self._build_parts(text_prompt, image_frames)

        for attempt in range(self.max_retries):
            try:
                analysis, api_time = self._call_gemini_api(parts, temperature=0.7)
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

        prev_image_frames = [
            fd["frame"]
            for fd in frames_previous[:3]
            if "frame" in fd and fd["frame"] is not None
        ]
        curr_image_frames = [
            fd["frame"]
            for fd in frames_current[:3]
            if "frame" in fd and fd["frame"] is not None
        ]

        full_prompt = (
            f"{prompt}\n\n"
            f"Previous trajectory ({len(prev_image_frames)} frames below):\n"
            f"Current trajectory ({len(curr_image_frames)} frames below):"
        )
        parts = self._build_parts(full_prompt, prev_image_frames + curr_image_frames)

        num_prev = len(prev_image_frames)
        num_curr = len(curr_image_frames)
        print(
            f"[VLM] Comparison call | prev_frames={num_prev} reward={reward_previous:.2f} "
            f"| curr_frames={num_curr} reward={reward_current:.2f}"
        )
        for attempt in range(self.max_retries):
            try:
                analysis, api_time = self._call_gemini_api(parts, temperature=0.7)
                print(f"[VLM] Comparison response received in {api_time:.1f}s ({len(analysis)} chars)")
                return analysis, api_time
            except Exception as e:
                print(f"[VLM ERROR] Comparison attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    return f"VLM comparison failed after {self.max_retries} attempts: {e}", 0.0
                time.sleep(5)

        return "VLM comparison unavailable", 0.0

    def analyze_candidate_diversity(
        self,
        candidates: List[Dict[str, Any]],
        env_description: str,
        frames_per_candidate: int = 2,
    ) -> tuple[str, float]:
        """
        Assess behavioral diversity across multiple policy candidates visually.

        Sends one representative frame per candidate to the VLM and asks whether
        the candidates are producing distinct behavioral strategies or collapsing
        to the same visual mode.

        Args:
            candidates: List of dicts, each with keys:
                - 'params': policy parameters (any type, or None)
                - 'reward': float episode reward
                - 'frames': List of frame dicts containing a 'frame' key (np.ndarray)
            env_description: Environment description string
            frames_per_candidate: Max frames to include per candidate (default 2)

        Returns:
            Tuple of (diversity_analysis_text, api_time)
        """
        if len(candidates) < 2:
            return "Diversity analysis requires at least 2 candidates.", 0.0

        # Render template
        template = self._jinja_env.get_template("vlm_diversity_prompt.j2")
        template_candidates = [
            {
                "reward": c["reward"],
                "params": c.get("params", None),
            }
            for c in candidates
        ]
        prompt = template.render(
            env_description=env_description,
            candidates=template_candidates,
        )

        all_image_frames: List[np.ndarray] = []
        label_lines: List[str] = []
        for idx, c in enumerate(candidates, start=1):
            frames = c.get("frames", [])
            count = 0
            for frame_data in frames:
                if count >= frames_per_candidate:
                    break
                if "frame" in frame_data and frame_data["frame"] is not None:
                    all_image_frames.append(frame_data["frame"])
                    count += 1
            label_lines.append(f"  [C{idx}] reward={c['reward']:.2f}, {count} frame(s)")

        full_prompt = prompt + "\n\nCandidate frames order:\n" + "\n".join(label_lines)
        parts = self._build_parts(full_prompt, all_image_frames)

        n_candidates = len(candidates)
        rewards_str = ", ".join(f"{c['reward']:.2f}" for c in candidates)
        print(f"[VLM] Diversity call | candidates={n_candidates} rewards=[{rewards_str}]")

        for attempt in range(self.max_retries):
            try:
                analysis, api_time = self._call_gemini_api(parts, temperature=0.7)
                print(
                    f"[VLM] Diversity response received in {api_time:.1f}s "
                    f"({len(analysis)} chars)"
                )
                return analysis, api_time
            except Exception as e:
                print(
                    f"[VLM ERROR] Diversity attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt == self.max_retries - 1:
                    return (
                        f"VLM diversity analysis failed after {self.max_retries} attempts: {e}",
                        0.0,
                    )
                time.sleep(5)

        return "VLM diversity analysis unavailable", 0.0
