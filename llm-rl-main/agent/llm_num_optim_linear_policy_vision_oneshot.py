"""
ProPS-V OneShot: Vision-Guided Prompted Policy Search with bounded context.

Identical to the base ProPS-V agent (llm_num_optim_linear_policy_vision.py)
except that only the BEST (params, reward) pair from the replay buffer is
passed to the LLM each iteration, keeping the prompt size O(rank) regardless
of training duration.  Neighborhood sampling and VLM scheduling are unchanged.
"""

from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent
from world.base_world import BaseWorld
import numpy as np
import re
import time


class LLMNumOptimVisionOneshotAgent(LLMNumOptimVisionAgent):
    """
    ProPS-V OneShot Agent.

    Same as LLMNumOptimVisionAgent but the LLM only ever sees the single
    best (params, reward) example — keeping context length constant.
    """

    def train_policy(self, world: BaseWorld, logdir):
        """
        ProPS-V OneShot training step.

        Identical to LLMNumOptimVisionAgent.train_policy except
        str_nd_examples is replaced by str_best_example (one entry only).

        Returns:
            Tuple (cpu_time, api_time, total_episodes, total_steps, total_reward)
        """

        def parse_parameters(input_text):
            """Parse parameters from LLM output."""
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)
            results = []
            for match in matches:
                results.append(float(match[1]))
            print(results)
            assert len(results) == self.rank, f"Expected {self.rank} params, got {len(results)}"
            return np.array(results).reshape(-1)

        def str_best_example(replay_buffer, n):
            """One-shot prompt: only the single best (params, reward) entry."""
            if not replay_buffer.buffer:
                return "(no examples yet)\n"
            best_weights, best_reward = max(replay_buffer.buffer, key=lambda x: x[1])
            parameters = best_weights.reshape(-1)
            line = ""
            for i in range(n):
                line += f"params[{i}]: {parameters[i]:.5g}; "
            line += f"f(params): {best_reward:.2f}\n"
            return line

        # ===== STEP 1: Determine lambda and whether VLM will run =====
        visual_analysis = None
        lambda_t = 0.0
        use_vision_this_iter = False

        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            use_vision_this_iter = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                random_state=np.random.RandomState(self.training_episodes)
            )
            print(f"ProPS-V OneShot Iteration {self.training_episodes}")
            print(f"λ_t = {lambda_t:.3f}")
            print(f"VLM invocation: {use_vision_this_iter}")

        # ===== STEP 2: Neighborhood Behavioral Sampling + VLM ======
        neighborhood_analysis = None
        if use_vision_this_iter:
            saved_params = self.policy.get_parameters().reshape(-1).copy()
            print(f"[Neighborhood] Running {self.n_neighbors} neighbors per anchor "
                  f"(Poisson λ={self.poisson_lam}, step={self.neighbor_step})...")

            # --- CURRENT anchor ---
            print("[Neighborhood] Anchor: CURRENT")
            cur_anchor, cur_neighbors = self._rollout_neighbors(
                world, saved_params, "current", logdir
            )
            visual_analysis = cur_anchor["analysis"]

            if visual_analysis is not None:
                self.visual_analysis_history.append({
                    'iteration': self.training_episodes,
                    'lambda': lambda_t,
                    'reward': cur_anchor["reward"],
                    'analysis': visual_analysis,
                    'num_frames': self.frame_sampler.sample_period,
                    'params': cur_anchor["params_str"],
                })
                with open(f"{logdir}/vlm_analysis.txt", "w", encoding="utf-8") as vf:
                    vf.write(visual_analysis)
                print(f"[VLM] CURRENT anchor analysis done ({len(visual_analysis)} chars)")

            # --- BEST and WORST anchors from replay buffer ---
            best_nb_anchor = None
            best_nb_neighbors = []
            worst_nb_anchor = None
            worst_nb_neighbors = []

            if len(self.replay_buffer.buffer) > 0:
                rb = self.replay_buffer.buffer
                best_rb_params, best_rb_reward = max(rb, key=lambda x: x[1])
                worst_rb_params, worst_rb_reward = min(rb, key=lambda x: x[1])

                if self.ablate_anchor != "best":
                    print(f"[Neighborhood] Anchor: BEST (reward={best_rb_reward:.2f})")
                    best_nb_anchor, best_nb_neighbors = self._rollout_neighbors(
                        world, np.array(best_rb_params).reshape(-1), "best", logdir
                    )

                if self.ablate_anchor != "worst" and worst_rb_reward != best_rb_reward:
                    print(f"[Neighborhood] Anchor: WORST (reward={worst_rb_reward:.2f})")
                    worst_nb_anchor, worst_nb_neighbors = self._rollout_neighbors(
                        world, np.array(worst_rb_params).reshape(-1), "worst", logdir
                    )

            # Restore original policy
            self.policy.update_policy(saved_params)

            # ===== STEP 3: Build neighborhood analysis summary for LLM =====
            def _anchor_block(label, anchor_res, neighbors):
                lines = [f"### Anchor [{label}]  reward={anchor_res['reward']:.2f}"]
                lines.append(f"  Params : {anchor_res['params_str']}")
                if anchor_res["analysis"]:
                    lines.append(f"  Behavior: {anchor_res['analysis']}")
                else:
                    lines.append(f"  Behavior: (no VLM analysis available)")
                if neighbors:
                    lines.append(f"\n  ---- Neighbors of [{label}] ----")
                    for rank_i, nb in enumerate(neighbors, start=1):
                        lines.append(f"  Neighbor #{rank_i}  reward={nb['reward']:.2f}")
                        lines.append(f"    Params : {nb['params_str']}")
                        if nb["analysis"]:
                            lines.append(f"    Behavior: {nb['analysis']}")
                        else:
                            lines.append(f"    Behavior: (no VLM analysis available)")
                return "\n".join(lines)

            blocks = ["## Neighborhood Behavioral Landscape\n"]
            if self.ablate_anchor != "current":
                blocks.append(_anchor_block("CURRENT", cur_anchor, cur_neighbors))
            if self.ablate_anchor != "best" and best_nb_anchor is not None:
                blocks.append(_anchor_block("BEST (replay buffer)", best_nb_anchor, best_nb_neighbors))
            if self.ablate_anchor != "worst" and worst_nb_anchor is not None:
                blocks.append(_anchor_block("WORST (replay buffer)", worst_nb_anchor, worst_nb_neighbors))
            neighborhood_analysis = "\n\n".join(blocks)

            with open(f"{logdir}/neighborhood_analysis.txt", "w", encoding="utf-8") as nf:
                nf.write(neighborhood_analysis)
            print(f"[Neighborhood] Landscape summary saved ({len(neighborhood_analysis)} chars)")

        # ===== STEP 4: Update policy — ONE example only (bounded context) =====
        print("\nUpdating policy with LLM (one-shot)...")
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_vision(
            str_best_example(self.replay_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            neighborhood_analysis=neighborhood_analysis,
        )
        self.api_call_time += api_time

        self.policy.update_policy(new_parameter_list)
        with open(f"{logdir}/parameters.txt", "w") as logging_q_file:
            logging_q_file.write(str(self.policy))
        with open(f"{logdir}/parameters_reasoning.txt", "w", encoding="utf-8") as q_reasoning_file:
            q_reasoning_file.write(reasoning)
        print("Policy updated!")

        # ===== STEP 5: Evaluate new policy =====
        print(f"Rolling out episode {self.training_episodes}...")
        results = []
        with open(f"{logdir}/training_rollout.txt", "w") as logging_file:
            for idx in range(self.num_evaluation_episodes):
                result = self.rollout_episode(
                    world, logging_file,
                    record=(idx == 0),
                    capture_frames=False
                )
                results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)

        # ==================================================================
        # STEP 5 — Update replay buffer and counters
        # ==================================================================
        self.replay_buffer.add(new_parameter_list, result)
        self.training_episodes += 1
        if self.enable_vision:
            self.visual_guidance.increment_iteration()

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time + self.vlm_api_time
        return _cpu_time, _api_time, self.total_episodes, self.total_steps, result
