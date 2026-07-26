"""
Vision-Language Model Analyzer for BMPS

This module provides VLM integration for analyzing episode frames
and generating diagnostic feedback for policy optimization.
Prompts are loaded from Jinja2 .j2 template files.
Uses LiteLLM for multi-provider model access.
"""

import base64
import io
import math
import time
import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from jinja2 import Environment, FileSystemLoader
import litellm


class VLMAnalyzer:
    """
    Analyzes episode frames using Vision-Language Models to provide
    visual diagnostic feedback for policy search.
    Uses LiteLLM for multi-provider model access (model read from config).
    """

    def __init__(
        self,
        vlm_model_name: str = "openrouter/google/gemma-3-27b-it",
        max_retries: int = 3,
        timeout: int = 60,
        template_dir: str = "agent/policy/templates",
        enable_reasoning: bool = False,
        vlm_api_key: str = None,
        vlm_api_base: str = None,
        vlm_frame_mode: str = "overlay",
    ):
        """
        Initialize VLM analyzer using LiteLLM.

        Args:
            vlm_model_name: LiteLLM model identifier (e.g. "gemini/gemini-2.5-flash-lite",
                            "openai/gpt-4o", "nvidia_nim/meta/llama-4-scout-17b-16e-instruct")
            max_retries: Maximum number of retry attempts on failure
            timeout: Timeout in seconds for API calls
            template_dir: Directory containing Jinja2 .j2 prompt templates
            enable_reasoning: If True, prepend chain-of-thought instruction to the prompt
            vlm_api_key: Optional API key for VLM provider
            vlm_api_base: Optional API base URL for VLM provider
            vlm_frame_mode: "individual" to send each frame separately,
                            "stacking" to send one contact sheet per rollout,
                            or "overlay" to average all frames pixel by pixel
        """
        self.vlm_model_name = vlm_model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_reasoning = enable_reasoning
        self.vlm_api_key = vlm_api_key
        self.vlm_api_base = vlm_api_base
        self.vlm_frame_mode = vlm_frame_mode.lower()
        if self.vlm_frame_mode not in {"individual", "stacking", "overlay"}:
            raise ValueError(
                "vlm_frame_mode must be 'individual', 'stacking', or 'overlay'"
            )

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

    def stack_frames(
        self,
        image_frames: List[np.ndarray],
        columns: int = 4,
        max_tile_width: int = 384,
        max_tile_height: int = 384,
    ) -> np.ndarray:
        """Combine a rollout's frames into one chronological contact sheet."""
        if not image_frames:
            raise ValueError("Cannot stack an empty frame list")

        pil_frames = []
        for frame in image_frames:
            if frame.dtype != np.uint8:
                frame = (
                    (frame * 255).astype(np.uint8)
                    if frame.max() <= 1.0
                    else frame.astype(np.uint8)
                )
            pil_frames.append(Image.fromarray(frame).convert("RGB"))

        scale = min(
            1.0,
            max_tile_width / max(image.width for image in pil_frames),
            max_tile_height / max(image.height for image in pil_frames),
        )
        tile_size = (
            max(1, round(pil_frames[0].width * scale)),
            max(1, round(pil_frames[0].height * scale)),
        )
        rows = math.ceil(len(pil_frames) / columns)
        sheet = Image.new(
            "RGB",
            (columns * tile_size[0], rows * tile_size[1]),
            color="black",
        )

        for index, image in enumerate(pil_frames):
            image = image.copy()
            image.thumbnail(tile_size, Image.Resampling.LANCZOS)
            x = (index % columns) * tile_size[0]
            y = (index // columns) * tile_size[1]
            offset_x = (tile_size[0] - image.width) // 2
            offset_y = (tile_size[1] - image.height) // 2
            sheet.paste(image, (x + offset_x, y + offset_y))

        return np.asarray(sheet)

    def overlay_frames(self, image_frames: List[np.ndarray]) -> np.ndarray:
        """Average a rollout's frames pixel by pixel into one image."""
        if not image_frames:
            raise ValueError("Cannot overlay an empty frame list")

        first_frame = image_frames[0]
        if first_frame.dtype != np.uint8:
            first_frame = (
                (first_frame * 255).astype(np.uint8)
                if first_frame.max() <= 1.0
                else first_frame.astype(np.uint8)
            )
        first_image = Image.fromarray(first_frame).convert("RGB")
        target_size = first_image.size
        composite = np.asarray(first_image, dtype=np.float32).copy()

        for frame_count, frame in enumerate(image_frames[1:], start=2):
            if frame.dtype != np.uint8:
                frame = (
                    (frame * 255).astype(np.uint8)
                    if frame.max() <= 1.0
                    else frame.astype(np.uint8)
                )
            image = Image.fromarray(frame).convert("RGB")
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            frame_array = np.asarray(image, dtype=np.float32)
            composite += (frame_array - composite) / frame_count

        return np.clip(composite, 0, 255).astype(np.uint8)

    def _build_messages(
        self,
        text_prompt: str,
        image_frames: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Build OpenAI-compatible message list from a text prompt and image frames.

        Args:
            text_prompt: The text portion of the message
            image_frames: List of RGB numpy arrays

        Returns:
            List of messages in OpenAI chat format with vision content
        """
        if self.vlm_frame_mode == "stacking" and len(image_frames) > 1:
            image_frames = [self.stack_frames(image_frames)]
            text_prompt = (
                "The attached image is a chronological contact sheet for this "
                "single rollout. Read cells from left to right, then top to bottom.\n\n"
                + text_prompt
            )
        elif self.vlm_frame_mode == "overlay" and len(image_frames) > 1:
            image_frames = [self.overlay_frames(image_frames)]
            text_prompt = (
                "The attached image is a pixel-level temporal overlay of all "
                "frames from this single rollout. Static regions remain sharp; "
                "transparent trails and blurred regions indicate movement over time.\n\n"
                + text_prompt
            )

        content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        for frame in image_frames:
            png_bytes = self.frame_to_png_bytes(frame)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            })
        return [{"role": "user", "content": content}]

    def _call_vlm_api(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
    ) -> tuple[str, float, int, int]:
        """
        Call VLM via LiteLLM.

        Args:
            messages: OpenAI-compatible message list
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, api_time, prompt_tokens, completion_tokens)
        """
        api_start_time = time.time()

        kwargs = {
            "model": self.vlm_model_name,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if self.vlm_api_key is not None:
            kwargs["api_key"] = self.vlm_api_key
        if self.vlm_api_base is not None:
            kwargs["api_base"] = self.vlm_api_base

        response = litellm.completion(**kwargs)

        api_time = time.time() - api_start_time
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage') and response.usage is not None:
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
        return response.choices[0].message.content.strip(), api_time, prompt_tokens, completion_tokens

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
    ) -> tuple[str, float, int, int]:
        """
        Analyze sampled episode frames using VLM and return diagnostic feedback.

        Args:
            frames: List of frame dicts with 'frame' key
            env_description: Description of the environment
            episode_reward: Total episodic reward
            terminated_early: Whether episode terminated early

        Returns:
            Tuple of (analysis_text, api_time, vlm_prompt_tokens, vlm_completion_tokens)
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
        messages = self._build_messages(text_prompt, image_frames)

        for attempt in range(self.max_retries):
            try:
                analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens = self._call_vlm_api(messages, temperature=0.7)
                return analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens
            except Exception as e:
                print(f"[VLM ERROR] Attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    return f"VLM analysis failed after {self.max_retries} attempts: {e}", 0.0, 0, 0
                time.sleep(5)

        return "VLM analysis unavailable", 0.0, 0, 0

    def analyze_trajectory_comparison(
        self,
        frames_current: List[Dict[str, Any]],
        frames_previous: List[Dict[str, Any]],
        reward_current: float,
        reward_previous: float,
        env_description: str
    ) -> tuple[str, float, int, int]:
        """
        Compare two trajectories visually to identify improvements or regressions.

        Args:
            frames_current: Sampled frames from current trajectory
            frames_previous: Sampled frames from previous trajectory
            reward_current: Reward of current trajectory
            reward_previous: Reward of previous trajectory
            env_description: Environment description

        Returns:
            Tuple of (comparative_analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens)
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
        messages = self._build_messages(full_prompt, prev_image_frames + curr_image_frames)

        num_prev = len(prev_image_frames)
        num_curr = len(curr_image_frames)
        print(
            f"[VLM] Comparison call | prev_frames={num_prev} reward={reward_previous:.2f} "
            f"| curr_frames={num_curr} reward={reward_current:.2f}"
        )
        for attempt in range(self.max_retries):
            try:
                analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens = self._call_vlm_api(messages, temperature=0.7)
                print(f"[VLM] Comparison response received in {api_time:.1f}s ({len(analysis)} chars)")
                return analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens
            except Exception as e:
                print(f"[VLM ERROR] Comparison attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    return f"VLM comparison failed after {self.max_retries} attempts: {e}", 0.0, 0, 0
                time.sleep(5)

        return "VLM comparison unavailable", 0.0, 0, 0

    def analyze_candidate_diversity(
        self,
        candidates: List[Dict[str, Any]],
        env_description: str,
        frames_per_candidate: int = 2,
    ) -> tuple[str, float, int, int]:
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
            Tuple of (diversity_analysis_text, api_time, vlm_prompt_tokens, vlm_completion_tokens)
        """
        if len(candidates) < 2:
            return "Diversity analysis requires at least 2 candidates.", 0.0, 0, 0

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
        messages = self._build_messages(full_prompt, all_image_frames)

        n_candidates = len(candidates)
        rewards_str = ", ".join(f"{c['reward']:.2f}" for c in candidates)
        print(f"[VLM] Diversity call | candidates={n_candidates} rewards=[{rewards_str}]")

        for attempt in range(self.max_retries):
            try:
                analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens = self._call_vlm_api(messages, temperature=0.7)
                print(
                    f"[VLM] Diversity response received in {api_time:.1f}s "
                    f"({len(analysis)} chars)"
                )
                return analysis, api_time, vlm_prompt_tokens, vlm_completion_tokens
            except Exception as e:
                print(
                    f"[VLM ERROR] Diversity attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt == self.max_retries - 1:
                    return (
                        f"VLM diversity analysis failed after {self.max_retries} attempts: {e}",
                        0.0, 0, 0,
                    )
                time.sleep(5)

        return "VLM diversity analysis unavailable", 0.0, 0, 0
