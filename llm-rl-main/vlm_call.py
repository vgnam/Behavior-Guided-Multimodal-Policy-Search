"""
vlm_call.py
-----------
Standalone test for the VLMAnalyzer.analyze_single_frame() API call.

Run from the llm-rl-main/ directory:
    python vlm_call.py

Options:
    --model MODEL   litellm model string (default: nvidia_nim/meta/llama-4-maverick-17b-128e-instruct)
    --reasoning     Enable chain-of-thought reasoning prefix

Required env var for NVIDIA NIM:
    NVIDIA_NIM_API_KEY  (or set NVIDIA_API_KEY — litellm accepts both)
"""

import sys
import os
import argparse
import time
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
from agent.policy.vlm_analyzer import VLMAnalyzer

ENV_DESCRIPTION = (
    "The CartPole environment: a pole is attached to a cart on a frictionless track. "
    "State = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]. "
    "Action 0 = push left, action 1 = push right. "
    "Goal: keep the pole upright and the cart within [-2.4, 2.4]. "
    "+1 reward per timestep the pole stays up."
)


def capture_cartpole_frame(steps: int = 30) -> tuple:
    """
    Run CartPole for `steps` steps with a simple heuristic policy and
    return (frame, state, action, episode_reward, terminated_early, timestep).
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    terminated = False
    frame = None
    last_state = obs
    last_action = 0

    for t in range(steps):
        frame = env.render()
        # Simple heuristic: push toward pole lean
        action = 1 if obs[2] > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        last_state = obs
        last_action = action
        if terminated or truncated:
            break

    env.close()
    return frame, last_state, last_action, total_reward, terminated, t


def run_test(model: str, enable_reasoning: bool = False):
    print("=" * 60)
    print(f"VLM Single-Frame Test")
    print(f"  model     : {model}")
    print(f"  reasoning : {'ON' if enable_reasoning else 'OFF'}")
    print("=" * 60)

    # ── 1. Capture frame ──────────────────────────────────────────
    print("\n[1] Rendering CartPole-v1 frame...")
    frame, state, action, ep_reward, terminated_early, timestep = capture_cartpole_frame(steps=30)
    print(f"    Frame shape : {frame.shape}  dtype={frame.dtype}")
    print(f"    State       : {np.round(state, 4)}")
    print(f"    Action      : {action}  ({'push right' if action == 1 else 'push left'})")
    print(f"    Ep. reward  : {ep_reward:.1f}  |  terminated early: {terminated_early}")
    print(f"    Timestep    : {timestep}")

    # Optionally save frame for inspection
    try:
        from PIL import Image
        out_path = "test_vlm_frame.png"
        Image.fromarray(frame).save(out_path)
        print(f"    Saved frame → {out_path}")
    except Exception:
        pass

    # ── 2. Instantiate VLMAnalyzer ────────────────────────────────
    print(f"\n[2] Initialising VLMAnalyzer (model={model}, timeout=120s)...")
    analyzer = VLMAnalyzer(
        vlm_model_name=model,
        max_retries=1,
        timeout=120,
        enable_reasoning=enable_reasoning,
    )
    print("    Done.")

    # ── 3. Call analyze_single_frame ──────────────────────────────
    print("\n[3] Calling analyze_single_frame()...")
    t0 = time.time()
    analysis, api_time = analyzer.analyze_single_frame(
        frame=frame,
        env_description=ENV_DESCRIPTION,
        episode_reward=ep_reward,
        terminated_early=terminated_early,
        timestep=timestep,
        state=state,
        action=action,
    )
    wall_time = time.time() - t0

    # ── 4. Results ────────────────────────────────────────────────
    print(f"\n[4] Response received  (api_time={api_time:.2f}s  wall={wall_time:.2f}s)")
    print("-" * 60)
    print(analysis)
    print("-" * 60)

    # Write to file
    safe_model = model.replace("/", "_").replace(":", "-")
    out_file = f"test_vlm_output_{safe_model}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"model    : {model}\n")
        f.write(f"reasoning: {enable_reasoning}\n")
        f.write(f"api_time : {api_time:.2f}s\n")
        f.write(f"wall_time: {wall_time:.2f}s\n\n")
        f.write(analysis)
    print(f"\nFull output saved → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VLM single-frame call")
    parser.add_argument(
        "--model",
        default="nvidia_nim/meta/llama-4-maverick-17b-128e-instruct",
        help="litellm model string (default: nvidia_nim/meta/llama-4-maverick-17b-128e-instruct)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable chain-of-thought reasoning prefix",
    )
    args = parser.parse_args()
    run_test(model=args.model, enable_reasoning=args.reasoning)
