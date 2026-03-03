"""
BL-ProPS Q-Table: Behavioral-Linguistic Prompted Policy Search for Discrete State Spaces

Implements the BL-ProPS algorithm on top of a Q-table policy:
  PRE-STEP : VLM describes three anchor policies (worst / current / best)
  STEP 1   : LLM generates a Behavioral Hypothesis b_hyp  :  P(b | r_{1:t})
  STEP 2   : VLM produces a Behavioral Posterior   b_post :  P(b | b_hyp, V_curr, V_best, V_worst)
  STEP 3   : LLM proposes new Q-table θ_{t+1}             :  P(θ | b_post, r_{1:t})
"""

import io
import itertools
import os
import re
import time
import numpy as np

from agent.policy.q_table import QTable
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from agent.policy.frame_sampler import FrameSampler
from agent.policy.vlm_analyzer import VLMAnalyzer
from world.base_world import BaseWorld


class BLProPSQTableAgent:
    """
    BL-ProPS Agent for discrete state spaces using a Q-table policy.

    Uses a three-step loop per iteration:
      1. LLM generates behavioral hypothesis   (blprops_q_table_step1_hypothesis.j2)
      2. VLM verifies and refines into b_post  (blprops_step2_posterior.j2)
      3. LLM maps b_post → new Q-table         (blprops_q_table_step3_proposal.j2)

    Each params[i] represents the action assigned to state i.
    """

    def __init__(
        self,
        logdir: str,
        actions: list,
        states: list,
        max_traj_count: int,
        max_traj_length: int,
        hypothesis_template,        # Jinja2 template for Step 1
        proposal_template,          # Jinja2 template for Step 3
        llm_model_name: str,
        num_evaluation_episodes: int,
        optimum: float,
        search_step_size: float = 0.1,
        env_desc_file: str = None,
        vlm_model_name: str = "gemini-2.5-flash-lite",
        frame_sample_period: int = 50,
        template_dir: str = "agent/policy/templates",
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0.0
        self.vlm_api_time = 0.0
        self.total_steps = 0
        self.total_episodes = 0

        self.actions = actions
        self.states = states
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.env_desc_file = env_desc_file
        self.max_traj_length = max_traj_length
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes

        # Q-table policy
        self.q_table = QTable(actions=actions, states=states)
        self.rank = len(self.q_table.mapping)

        # Valid action bounds (for clamping LLM output)
        if len(actions) == 1:
            valid_actions = actions[0]
        else:
            valid_actions = list(itertools.product(*actions))
        self.min_action = min(valid_actions)
        self.max_action = max(valid_actions)

        # Replay buffer (numerical history only)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)

        # Two separate LLM brains (different templates, same model)
        dummy = hypothesis_template
        self.hypothesis_brain = LLMBrain(hypothesis_template, dummy, llm_model_name)
        self.proposal_brain   = LLMBrain(proposal_template,   dummy, llm_model_name)

        # VLM analyzer
        self.vlm_analyzer = VLMAnalyzer(
            vlm_model_name=vlm_model_name,
            template_dir=template_dir,
        )

        # Frame sampler
        self.frame_sampler = FrameSampler(sample_period=frame_sample_period)

        # Anchor states (initialised in random_warmup)
        self._anchor_best  = None   # dict: params_arr, params_str, reward, description, frames
        self._anchor_worst = None
        self._anchor_curr  = None

        self.training_episodes = 0

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _params_arr(self) -> np.ndarray:
        """Return current Q-table as a flat array of actions (indexed by state index)."""
        return np.array([self.q_table.mapping[s] for s in self.q_table.states])

    def _params_str(self, params_arr) -> str:
        """Format params array as 'params[0]: a, params[1]: b, ...' string."""
        return ", ".join(f"params[{i}]: {int(v)}" for i, v in enumerate(params_arr))

    def _format_history(self) -> str:
        text = ""
        for weights, reward in self.replay_buffer.buffer:
            p = np.array(weights).reshape(-1)
            line = "  ".join(f"params[{i}]: {int(v)}" for i, v in enumerate(p))
            text += f"{line}  ->  f(params): {reward:.2f}\n"
        return text

    def _parse_parameters(self, input_text: str) -> np.ndarray:
        """Parse 'params[i]: value' from first line of LLM output → integer action array."""
        s = input_text.split("\n")[0]
        pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
        matches = pattern.findall(s)
        results = [float(m[1]) for m in matches]
        assert len(results) == self.rank, (
            f"Expected {self.rank} params, got {len(results)} — raw: {s!r}"
        )
        return np.array(results)

    # ------------------------------------------------------------------ #
    #  Rollout                                                             #
    # ------------------------------------------------------------------ #

    def _rollout_with_frames(self, world: BaseWorld, logging_file):
        """Roll out one episode with rendered frame capture for VLM."""
        state = world.reset()
        logging_file.write("state | action | reward\n")

        done = False
        trajectory = []
        step_idx = 0

        while not done:
            action = self.q_table.get_action(state)
            action = int(np.reshape(action, (1,)))

            frame = None
            if hasattr(world.env, "render"):
                try:
                    frame = world.env.render()
                    if not isinstance(frame, np.ndarray):
                        frame = None
                except Exception:
                    frame = None

            next_state, reward, done = world.step(action)
            logging_file.write(f"{state} | {action} | {reward}\n")

            trajectory.append({
                "state": state,
                "action": action,
                "reward": reward,
                "frame": frame,
                "episode_num": 0,
                "timestep": step_idx,
                "frame_type": "periodic",
            })

            state = next_state
            step_idx += 1
            self.total_steps += 1

            if step_idx >= self.max_traj_length:
                break

        total_reward = world.get_accu_reward()
        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        terminated_early = step_idx < self.max_traj_length
        return total_reward, trajectory, terminated_early

    def _rollout_eval(self, world: BaseWorld, logging_file) -> float:
        """Roll out one episode for evaluation only (no frame capture)."""
        state = world.reset()
        done = False
        step_idx = 0
        while not done:
            action = self.q_table.get_action(state)
            action = int(np.reshape(action, (1,)))
            next_state, reward, done = world.step(action)
            state = next_state
            step_idx += 1
            self.total_steps += 1
            if step_idx >= self.max_traj_length:
                break
        self.total_episodes += 1
        return world.get_accu_reward()

    def _sample_frames(self, trajectory, terminated_early):
        indices = self.frame_sampler.sample_frames(
            trajectory, terminated_early, self.max_traj_length
        )
        return self.frame_sampler.get_frames(trajectory, indices)

    def _vlm_describe(self, frames, reward, params_str, logdir=None):
        """Call VLM describe_behavior; returns description string."""
        desc, vlm_t = self.vlm_analyzer.describe_behavior(
            frames,
            self.env_desc_file if self.env_desc_file else "RL Environment",
            reward,
            params=params_str,
        )
        self.vlm_api_time += vlm_t
        self.api_call_time += vlm_t
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            with open(f"{logdir}/vlm_description.txt", "w", encoding="utf-8") as f:
                f.write(desc)
        return desc

    def _build_anchor_dict(self, anchor_state):
        if anchor_state is None:
            return {"params": "N/A", "reward": 0.0, "description": "N/A"}
        return {
            "params": anchor_state["params_str"],
            "reward": anchor_state["reward"],
            "description": anchor_state["description"],
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def random_warmup(self, world: BaseWorld, logdir: str, num_episodes: int):
        """
        Random warmup — initialises replay buffer and bootstraps anchor policies.
        """
        os.makedirs(logdir, exist_ok=True)
        first_anchor = None

        for episode in range(num_episodes):
            self.q_table.initialize_policy()
            print(f"Rolling out warmup episode {episode}...")
            lf = io.StringIO()
            reward, trajectory, terminated_early = self._rollout_with_frames(world, lf)
            print(f"Result: {reward}")

            params_arr = self._params_arr()
            params_s   = self._params_str(params_arr)

            frames = self._sample_frames(trajectory, terminated_early)
            has_frames = any(f.get("frame") is not None for f in frames)

            desc = ""
            if has_frames:
                desc = self._vlm_describe(frames, reward, params_s)

            anchor = {
                "params_arr": params_arr.copy(),
                "params_str": params_s,
                "reward": reward,
                "description": desc,
                "frames": frames,
            }
            self.replay_buffer.add(params_arr, reward)

            if first_anchor is None:
                first_anchor = anchor
                self._anchor_best  = anchor
                self._anchor_worst = anchor
                self._anchor_curr  = anchor
            else:
                if reward > self._anchor_best["reward"]:
                    self._anchor_best = anchor
                if reward < self._anchor_worst["reward"]:
                    self._anchor_worst = anchor
                self._anchor_curr = anchor

            print(
            print(
                f"  best={self._anchor_best['reward']:.2f} | worst={self._anchor_worst['reward']:.2f}"
            )
            )

    def train_policy(self, world: BaseWorld, logdir: str):
        """
        One BL-ProPS Q-table iteration.

        Returns:
            (cpu_time, api_time, total_episodes, total_steps, total_reward)
        """
        t = self.training_episodes
        print(f"\n{'='*60}")
        print(f"BL-ProPS-QTable iteration {t}")
        print(f"{'='*60}")

        # ──────────────────────────────────────────────
        # PRE-STEP: VLM describes three anchor policies
        # ──────────────────────────────────────────────
        print("[BL-ProPS-QTable PRE-STEP] Rolling out current policy for VLM description...")
        pre_log = os.path.join(logdir, "pre_rollout.txt")
        with open(pre_log, "w") as lf:
            r_curr, traj_curr, term_curr = self._rollout_with_frames(world, lf)

        params_curr_arr = self._params_arr()
        params_curr_str = self._params_str(params_curr_arr)
        frames_curr     = self._sample_frames(traj_curr, term_curr)

        b_curr = self._vlm_describe(
            frames_curr, r_curr, params_curr_str,
            logdir=os.path.join(logdir, "vlm_curr"),
        )

        # Refresh best / worst descriptions from stored frames
        b_best  = self._anchor_best["description"]
        b_worst = self._anchor_worst["description"]

        if self._anchor_best.get("frames"):
            b_best = self._vlm_describe(
                self._anchor_best["frames"],
                self._anchor_best["reward"],
                self._anchor_best["params_str"],
            )
            self._anchor_best["description"] = b_best

        if self._anchor_worst.get("frames"):
            b_worst = self._vlm_describe(
                self._anchor_worst["frames"],
                self._anchor_worst["reward"],
                self._anchor_worst["params_str"],
            )
            self._anchor_worst["description"] = b_worst

        self._anchor_curr = {
            "params_arr": params_curr_arr.copy(),
            "params_str": params_curr_str,
            "reward": r_curr,
            "description": b_curr,
            "frames": frames_curr,
        }

        anchors = {
            "worst":   self._build_anchor_dict(self._anchor_worst),
            "current": self._build_anchor_dict(self._anchor_curr),
            "best":    self._build_anchor_dict(self._anchor_best),
        }

        print(f"  b_curr  ({r_curr:.2f}): {b_curr[:80]}...")
        print(f"  b_best  ({self._anchor_best['reward']:.2f}): {b_best[:80]}...")
        print(f"  b_worst ({self._anchor_worst['reward']:.2f}): {b_worst[:80]}...")

        history_str = self._format_history()

        # ──────────────────────────────────────────────
        # STEP 1: LLM → Behavioral Hypothesis b_hyp
        # ──────────────────────────────────────────────
        print("\n[BL-ProPS-QTable STEP 1] LLM generating behavioral hypothesis...")
        b_hyp, hyp_log, hyp_time = self.hypothesis_brain.llm_generate_behavioral_hypothesis(
            anchors=anchors,
            history_str=history_str,
            step_number=t,
            env_description=self.env_desc_file,
            rank=self.rank,
            actions=self.actions,
        )
        self.api_call_time += hyp_time
        print(f"  b_hyp ({hyp_time:.1f}s): {b_hyp[:120]}...")

        with open(os.path.join(logdir, "step1_b_hyp.txt"), "w", encoding="utf-8") as f:
            f.write(hyp_log)

        # ──────────────────────────────────────────────
        # STEP 2: VLM → Behavioral Posterior b_post
        # ──────────────────────────────────────────────
        print("\n[BL-ProPS-QTable STEP 2] VLM computing behavioral posterior...")
        frames_best  = self._anchor_best.get("frames", [])
        frames_worst = self._anchor_worst.get("frames", [])

        b_post, post_time = self.vlm_analyzer.build_behavioral_posterior(
            b_hyp=b_hyp,
            frames_worst=frames_worst,
            frames_curr=frames_curr,
            frames_best=frames_best,
            r_worst=self._anchor_worst["reward"],
            r_curr=r_curr,
            r_best=self._anchor_best["reward"],
            env_description=self.env_desc_file,
        )
        self.vlm_api_time += post_time
        self.api_call_time += post_time
        print(f"  b_post ({post_time:.1f}s): {b_post[:120]}...")

        with open(os.path.join(logdir, "step2_b_post.txt"), "w", encoding="utf-8") as f:
            f.write(b_post)

        # ──────────────────────────────────────────────
        # STEP 3: LLM → Q-table Proposal θ_{t+1}
        # ──────────────────────────────────────────────
        print("\n[BL-ProPS-QTable STEP 3] LLM proposing new Q-table...")
        new_params, prop_log, prop_time = self.proposal_brain.llm_propose_parameters_blprops(
            b_post=b_post,
            anchors=anchors,
            history_str=history_str,
            rank=self.rank,
            step_number=t,
            parse_parameters=self._parse_parameters,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            env_description=self.env_desc_file,
            actions=self.actions,
        )
        self.api_call_time += prop_time
        print(f"  New params ({prop_time:.1f}s): {new_params}")

        with open(os.path.join(logdir, "step3_proposal.txt"), "w", encoding="utf-8") as f:
            f.write(prop_log)

        # ──────────────────────────────────────────────
        # EXECUTE AND UPDATE
        # ──────────────────────────────────────────────
        self.q_table.update_policy(new_params)

        # Log parameters and reasoning (mirroring propsp)
        with open(os.path.join(logdir, "parameters.txt"), "w") as f:
            f.write(str(self.q_table.mapping))
        with open(os.path.join(logdir, "parameters_reasoning.txt"), "w", encoding="utf-8") as f:
            f.write("=== STEP 1: Behavioral Hypothesis ===\n")
            f.write(hyp_log)
            f.write("\n\n=== STEP 2: Behavioral Posterior ===\n")
            f.write(b_post)
            f.write("\n\n=== STEP 3: Parameter Proposal ===\n")
            f.write(prop_log)
        print("Policy updated!")

        # Evaluate new policy
        print(f"\n[BL-ProPS-QTable EVAL] Evaluating new policy ({self.num_evaluation_episodes} eps)...")
        eval_rewards = []
        eval_log = os.path.join(logdir, "training_rollout.txt")
        with open(eval_log, "w") as lf:
            for _ in range(self.num_evaluation_episodes):
                r = self._rollout_eval(world, lf)
                eval_rewards.append(r)
        result = float(np.mean(eval_rewards))
        print(f"  mean reward = {result:.2f}")

        # Add to numerical history
        new_params_arr = self._params_arr()
        self.replay_buffer.add(new_params_arr, result)

        new_params_str = self._params_str(new_params_arr)
        new_anchor_candidate = {
            "params_arr": new_params_arr.copy(),
            "params_str": new_params_str,
            "reward": result,
            "description": "",
            "frames": [],
        }

        # Eagerly capture frames for anchor update if new best/worst
        if result > self._anchor_best["reward"] or result < self._anchor_worst["reward"]:
            print("[BL-ProPS-QTable] Re-rolling out to capture frames for anchor update...")
            capture_log = os.path.join(logdir, "anchor_capture_rollout.txt")
            with open(capture_log, "w") as lf:
                _, traj_new, term_new = self._rollout_with_frames(world, lf)
            frames_new = self._sample_frames(traj_new, term_new)
            has_frames = any(f.get("frame") is not None for f in frames_new)
            desc_new = ""
            if has_frames:
                desc_new = self._vlm_describe(frames_new, result, new_params_str)
            new_anchor_candidate["frames"] = frames_new
            new_anchor_candidate["description"] = desc_new

        if result > self._anchor_best["reward"]:
            print(f"  [BL-ProPS-QTable] New BEST: {result:.2f} > {self._anchor_best['reward']:.2f}")
            self._anchor_best = new_anchor_candidate

        if result < self._anchor_worst["reward"]:
            print(f"  [BL-ProPS-QTable] New WORST: {result:.2f} < {self._anchor_worst['reward']:.2f}")
            self._anchor_worst = new_anchor_candidate

        self.training_episodes += 1

        cpu_time = time.process_time() - self.start_time
        return cpu_time, self.api_call_time, self.total_episodes, self.total_steps, result

    def evaluate_policy(self, world: BaseWorld, logdir: str):
        results = []
        for idx in range(self.num_evaluation_episodes):
            log_path = os.path.join(logdir, f"eval_{idx}.txt")
            with open(log_path, "w") as lf:
                results.append(self._rollout_eval(world, lf))
        return results
