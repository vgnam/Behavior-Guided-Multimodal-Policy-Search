"""BMPS-CMA: pairwise visual guidance with trust-region CMA policy search."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import re
import time
from typing import Optional

import numpy as np
from PIL import Image

from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent
from agent.policy.pairwise_preference import (
    BradleyTerryAggregator,
    PairwisePreference,
    parse_pairwise_response,
)
from agent.policy.temporal_frame_stacker import TemporalFrameStacker
from agent.policy.trust_region_cma import TrustRegionCMA
from world.base_world import BaseWorld


@dataclass
class EvaluatedCandidate:
    identifier: str
    role: str
    params: np.ndarray
    reward: float
    visual_images: list[np.ndarray]
    num_frames: int
    terminated_early: bool


class LLMNumOptimBMPSCMAAgent(LLMNumOptimVisionAgent):
    """Continuous-policy BMPS variant guided by an LLM proposal and CMA-ES.

    The LLM keeps the existing BMPS contract and returns a complete ``params``
    vector.  Its proposal is projected into the current CMA Mahalanobis trust
    region. Periodically sampled trajectory frames are sent separately by
    default, matching BMPS, or optionally stacked into one temporal
    superposition. The resulting score-free preference landscape guides the
    following generation, avoiding a second rollout population per iteration.
    """

    def __init__(
        self,
        *args,
        cma_population_size=13,
        cma_sigma=0.5,
        cma_covariance_mode="auto",
        cma_full_covariance_max_dim=128,
        cma_lower_bound=-6.0,
        cma_upper_bound=6.0,
        trust_region_radius=3.0,
        llm_guidance_alpha=1.0,
        cma_acceptance_margin=0.0,
        cma_restart_distance=8.0,
        cma_restart_margin=0.0,
        candidate_evaluation_episodes=1,
        pairwise_max_comparisons=4,
        stack_trajectory_frames=False,
        stack_motion_threshold=18.0,
        stack_background_learning_rate=0.01,
        stack_tint_strength=0.45,
        stack_occupancy_scale=4.0,
        cma_seed=None,
        **kwargs,
    ):
        configured_frame_period = int(kwargs.get("frame_sample_period", 50))
        super().__init__(*args, **kwargs)
        if candidate_evaluation_episodes < 1:
            raise ValueError("candidate_evaluation_episodes must be at least 1")
        if pairwise_max_comparisons < 1:
            raise ValueError("pairwise_max_comparisons must be at least 1")

        self.cma_population_size = int(cma_population_size)
        self.cma_sigma_initial = float(cma_sigma)
        self.cma_covariance_mode = cma_covariance_mode
        self.cma_full_covariance_max_dim = int(cma_full_covariance_max_dim)
        self.cma_lower_bound = float(cma_lower_bound)
        self.cma_upper_bound = float(cma_upper_bound)
        self.trust_region_radius = float(trust_region_radius)
        self.llm_guidance_alpha = float(llm_guidance_alpha)
        self.cma_acceptance_margin = float(cma_acceptance_margin)
        self.cma_restart_distance = float(cma_restart_distance)
        self.cma_restart_margin = float(cma_restart_margin)
        self.candidate_evaluation_episodes = int(candidate_evaluation_episodes)
        self.pairwise_max_comparisons = int(pairwise_max_comparisons)
        self.stack_trajectory_frames = bool(stack_trajectory_frames)
        self.cma_seed = cma_seed
        self.stack_frame_period = max(1, configured_frame_period)

        self.stack_kwargs = {
            "motion_threshold": stack_motion_threshold,
            "background_learning_rate": stack_background_learning_rate,
            "tint_strength": stack_tint_strength,
            "occupancy_scale": stack_occupancy_scale,
        }
        self.preference_aggregator = BradleyTerryAggregator()
        self.cma: Optional[TrustRegionCMA] = None
        self.pending_behavioral_landscape: Optional[str] = None
        self.last_generation_results: list[EvaluatedCandidate] = []

    def _ensure_cma_initialized(self) -> None:
        if self.cma is not None:
            return
        if self.replay_buffer.buffer:
            initial_mean, initial_reward = max(
                self.replay_buffer.buffer, key=lambda entry: entry[1]
            )
            initial_mean = np.asarray(initial_mean, dtype=float).reshape(-1)
            print(f"[BMPS-CMA] Initial mean from best warmup reward={initial_reward:.2f}")
        else:
            initial_mean = self.policy.get_parameters().reshape(-1).copy()
            print("[BMPS-CMA] Initial mean from current policy")

        self.cma = TrustRegionCMA(
            mean=initial_mean,
            sigma=self.cma_sigma_initial,
            population_size=self.cma_population_size,
            covariance_mode=self.cma_covariance_mode,
            full_covariance_max_dim=self.cma_full_covariance_max_dim,
            lower_bound=self.cma_lower_bound,
            upper_bound=self.cma_upper_bound,
            seed=self.cma_seed,
        )
        self.policy.update_policy(self.cma.mean)

    def _render_environment_description(self) -> str:
        if not self.env_desc_file:
            return "Reinforcement-learning environment"
        try:
            environment = self.llm_brain.llm_si_template.environment
            return environment.get_template(self.env_desc_file).render()
        except Exception:
            return str(self.env_desc_file)

    def _format_params(self, params: np.ndarray, include_reward: Optional[float] = None) -> str:
        values = np.asarray(params).reshape(-1)
        line = "; ".join(f"params[{idx}]: {value:.8g}" for idx, value in enumerate(values))
        if include_reward is not None:
            line += f"; f(params): {include_reward:.2f}"
        return line

    def _format_numerical_history(self) -> str:
        return "\n".join(
            self._format_params(np.asarray(params), include_reward=float(reward))
            for params, reward in self.replay_buffer.buffer
        )

    def _parse_parameters(self, response: str) -> np.ndarray:
        first_line = response.splitlines()[0] if response.splitlines() else response
        matches = re.findall(
            r"params\[(\d+)\]\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            first_line,
        )
        parsed = {int(index): float(value) for index, value in matches}
        missing = [index for index in range(self.rank) if index not in parsed]
        if missing:
            raise ValueError(
                f"Expected params[0] through params[{self.rank - 1}]; missing {missing[:10]}"
            )
        return np.array([parsed[index] for index in range(self.rank)], dtype=float)

    def _sanitize_llm_proposal(self, proposal: np.ndarray) -> np.ndarray:
        """Apply the original BMPS parameter constraint to the LLM proposal."""
        return np.round(
            np.clip(
                np.asarray(proposal, dtype=float).reshape(-1),
                self.cma_lower_bound,
                self.cma_upper_bound,
            ),
            1,
        )

    @staticmethod
    def _safe_render(world: BaseWorld) -> Optional[np.ndarray]:
        try:
            frame = world.env.render()
            return frame if isinstance(frame, np.ndarray) else None
        except Exception:
            return None

    def _rollout_with_visuals(
        self,
        world: BaseWorld,
        logging_file,
        capture_visual: bool,
    ) -> tuple[float, list[np.ndarray], int, bool]:
        """Roll out one episode and collect periodically sampled visual frames."""
        state = np.expand_dims(world.reset(), axis=0)
        sampled_frames: list[np.ndarray] = []

        logging_file.write(
            f"{', '.join(str(x) for x in self.policy.get_parameters().reshape(-1))}\n"
        )
        logging_file.write("parameter ends\n\nstate | action | reward\n")

        if capture_visual:
            initial_frame = self._safe_render(world)
            if initial_frame is not None:
                sampled_frames.append(initial_frame)

        step_idx = 0
        done = False
        while not done and step_idx < self.max_traj_length:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.array([np.argmax(action)])

            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")

            next_step_idx = step_idx + 1
            if capture_visual and (
                next_step_idx % self.stack_frame_period == 0
                or done
                or next_step_idx >= self.max_traj_length
            ):
                frame = self._safe_render(world)
                if frame is not None:
                    sampled_frames.append(frame)

            step_idx = next_step_idx
            self.total_steps += 1
            state = np.expand_dims(next_state, axis=0)

        total_reward = float(world.get_accu_reward())
        terminated_early = step_idx < self.max_traj_length
        logging_file.write(f"Total reward: {total_reward}\n")
        self.total_episodes += 1
        frame_count = len(sampled_frames)
        if self.stack_trajectory_frames and sampled_frames:
            stacker = TemporalFrameStacker(**self.stack_kwargs)
            for frame in sampled_frames:
                stacker.add(frame)
            stacked_image = stacker.finalize()
            visual_images = [stacked_image] if stacked_image is not None else []
        else:
            visual_images = sampled_frames
        return total_reward, visual_images, frame_count, terminated_early

    def _evaluate_candidate(
        self,
        world: BaseWorld,
        params: np.ndarray,
        identifier: str,
        role: str,
        logdir: str,
        capture_visual: bool,
    ) -> EvaluatedCandidate:
        params = self.cma.repair(params)
        self.policy.update_policy(params)
        rewards = []
        visual_images: list[np.ndarray] = []
        num_frames = 0
        terminated_early = False

        for evaluation_idx in range(self.candidate_evaluation_episodes):
            log_path = os.path.join(
                logdir, f"{identifier}_rollout_{evaluation_idx}.txt"
            )
            with open(log_path, "w", encoding="utf-8") as logging_file:
                reward, images, frames, terminated = self._rollout_with_visuals(
                    world,
                    logging_file,
                    capture_visual=capture_visual and evaluation_idx == 0,
                )
            rewards.append(reward)
            if images:
                visual_images = images
                num_frames = frames
                terminated_early = terminated

        if visual_images:
            if self.stack_trajectory_frames:
                Image.fromarray(visual_images[0]).save(
                    os.path.join(logdir, f"{identifier}_temporal_stack.png")
                )
            else:
                for frame_index, frame in enumerate(visual_images):
                    Image.fromarray(frame).save(
                        os.path.join(
                            logdir, f"{identifier}_frame_{frame_index:03d}.png"
                        )
                    )

        return EvaluatedCandidate(
            identifier=identifier,
            role=role,
            params=params.copy(),
            reward=float(np.mean(rewards)),
            visual_images=visual_images,
            num_frames=num_frames,
            terminated_early=terminated_early,
        )

    def _select_comparison_targets(
        self,
        anchor: EvaluatedCandidate,
        candidates: list[EvaluatedCandidate],
    ) -> list[EvaluatedCandidate]:
        others = [candidate for candidate in candidates if candidate.identifier != anchor.identifier]
        if not others:
            return []

        selected: list[EvaluatedCandidate] = []

        def add(candidate: Optional[EvaluatedCandidate]) -> None:
            if candidate is not None and all(
                existing.identifier != candidate.identifier for existing in selected
            ):
                selected.append(candidate)

        add(next((candidate for candidate in others if candidate.role == "llm_proposal"), None))
        add(max(others, key=lambda candidate: candidate.reward))
        add(min(others, key=lambda candidate: candidate.reward))
        for candidate in sorted(others, key=lambda item: item.reward, reverse=True):
            add(candidate)
            if len(selected) >= self.pairwise_max_comparisons:
                break
        return selected[: self.pairwise_max_comparisons]

    def _build_pairwise_landscape(
        self,
        candidates: list[EvaluatedCandidate],
        logdir: str,
        lambda_t: float,
    ) -> Optional[str]:
        visual_candidates = [
            candidate for candidate in candidates if candidate.visual_images
        ]
        anchor = next(
            (candidate for candidate in visual_candidates if candidate.role == "cma_mean"),
            None,
        )
        if anchor is None:
            return None
        targets = self._select_comparison_targets(anchor, visual_candidates)
        if not targets:
            return None

        env_description = self._render_environment_description()

        def compare(target: EvaluatedCandidate) -> tuple[PairwisePreference, float]:
            label_a = f"{anchor.identifier} (CMA mean)"
            label_b = f"{target.identifier} ({target.role})"
            if self.stack_trajectory_frames:
                response, api_time = self.vlm_analyzer.analyze_stacked_pair(
                    anchor.visual_images[0],
                    target.visual_images[0],
                    env_description=env_description,
                    label_a=label_a,
                    label_b=label_b,
                )
            else:
                response, api_time = self.vlm_analyzer.analyze_frame_sequences_pair(
                    anchor.visual_images,
                    target.visual_images,
                    env_description=env_description,
                    label_a=label_a,
                    label_b=label_b,
                )
            return (
                parse_pairwise_response(
                    response,
                    left_id=anchor.identifier,
                    right_id=target.identifier,
                ),
                api_time,
            )

        comparisons: list[PairwisePreference] = []
        total_vlm_time = 0.0
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(compare, target) for target in targets]
            for future in as_completed(futures):
                comparison, api_time = future.result()
                comparisons.append(comparison)
                total_vlm_time += api_time
        self.vlm_api_time += total_vlm_time

        compared_candidates = [anchor, *targets]
        candidate_by_id = {
            candidate.identifier: candidate for candidate in compared_candidates
        }
        ranked_ids = self.preference_aggregator.rank(
            list(candidate_by_id), comparisons
        )

        lines = [
            f"## Pairwise Behavioral Landscape from generation {self.training_episodes}",
            "The visual preference order below has no exposed Bradley-Terry score.",
            "Policies are shown as complete parameter vectors in the existing BMPS format.",
            "",
            "### Visual preference order",
        ]
        for rank, identifier in enumerate(ranked_ids, start=1):
            candidate = candidate_by_id[identifier]
            lines.append(f"Rank {rank} — {identifier} ({candidate.role})")
            lines.append(self._format_params(candidate.params, candidate.reward))

        lines.append("\n### Pairwise visual comparisons")
        for comparison in comparisons:
            left = candidate_by_id[comparison.left_id]
            right = candidate_by_id[comparison.right_id]
            if comparison.winner == "tie":
                preferred = "TIE"
            else:
                preferred = comparison.winner
            lines.extend(
                [
                    "",
                    f"Policy A — {left.identifier} ({left.role})",
                    self._format_params(left.params, left.reward),
                    f"Policy B — {right.identifier} ({right.role})",
                    self._format_params(right.params, right.reward),
                    f"Preferred: {preferred}",
                    f"Confidence: {comparison.confidence}",
                    f"Evidence: {comparison.evidence}",
                ]
            )

        landscape = "\n".join(lines)
        with open(
            os.path.join(logdir, "pairwise_behavioral_landscape.txt"),
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(landscape)

        best = max(candidates, key=lambda candidate: candidate.reward)
        self.visual_analysis_history.append(
            {
                "iteration": self.training_episodes,
                "lambda": lambda_t,
                "reward": best.reward,
                "analysis": landscape,
                "num_frames": sum(candidate.num_frames for candidate in visual_candidates),
                "params": self._format_params(best.params),
            }
        )
        return landscape

    def _log_cma_state(
        self,
        logdir: str,
        projection,
        update_status: str,
        baseline_reward: float,
        proposal_reward: float,
        best_gaussian_reward: float,
        weighted_elite_reward: float,
    ) -> None:
        summary = self.cma.covariance_summary()
        lines = [
            f"generation: {self.training_episodes}",
            f"proposal_mahalanobis_distance: {projection.mahalanobis_distance:.8g}",
            f"proposal_applied_fraction: {projection.applied_fraction:.8g}",
            f"trust_region_radius: {self.trust_region_radius:.8g}",
            f"update_status: {update_status}",
            f"baseline_reward: {baseline_reward:.8g}",
            f"proposal_reward: {proposal_reward:.8g}",
            f"best_gaussian_reward: {best_gaussian_reward:.8g}",
            f"weighted_elite_reward: {weighted_elite_reward:.8g}",
            f"covariance_mode: {summary['mode']}",
            f"sigma: {summary['sigma']:.8g}",
            f"covariance_diag_min: {summary['diag_min']:.8g}",
            f"covariance_diag_max: {summary['diag_max']:.8g}",
            "mean:",
            self._format_params(self.cma.mean),
        ]
        with open(os.path.join(logdir, "cma_state.txt"), "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def train_policy(self, world: BaseWorld, logdir: str):
        """Run one LLM-guided CMA generation."""
        self._ensure_cma_initialized()

        lambda_t = 0.0
        use_vision = False
        if self.enable_vision:
            lambda_t = self.visual_guidance.get_lambda(self.training_episodes)
            use_vision = self.visual_guidance.should_invoke_vlm(
                self.training_episodes,
                random_state=np.random.RandomState(self.training_episodes),
            )

        print("\n[BMPS-CMA] Requesting a complete policy proposal from the LLM...")
        proposal, reasoning, llm_time = self.llm_brain.llm_update_parameters_num_optim_vision(
            self._format_numerical_history(),
            self._parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            neighborhood_analysis=self.pending_behavioral_landscape,
        )
        self.api_call_time += llm_time
        proposal = self._sanitize_llm_proposal(proposal)

        with open(
            os.path.join(logdir, "llm_proposal_reasoning.txt"), "w", encoding="utf-8"
        ) as file:
            file.write(reasoning)
        with open(os.path.join(logdir, "llm_proposal.txt"), "w", encoding="utf-8") as file:
            file.write(self._format_params(proposal))

        projection = self.cma.project_proposal(
            proposal,
            trust_region_radius=self.trust_region_radius,
            guidance_alpha=self.llm_guidance_alpha,
        )
        gaussian_params = self.cma.ask(projection.center)

        print(
            f"[BMPS-CMA] proposal distance={projection.mahalanobis_distance:.3f}, "
            f"applied fraction={projection.applied_fraction:.3f}, "
            f"capture period={self.stack_frame_period if use_vision else 'disabled'}, "
            f"visual mode={'stacked' if self.stack_trajectory_frames else 'separate'}"
        )

        results: list[EvaluatedCandidate] = []
        results.append(
            self._evaluate_candidate(
                world,
                self.cma.mean,
                "candidate_cma_mean",
                "cma_mean",
                logdir,
                use_vision,
            )
        )
        results.append(
            self._evaluate_candidate(
                world,
                proposal,
                "candidate_llm_proposal",
                "llm_proposal",
                logdir,
                use_vision,
            )
        )
        for index, params in enumerate(gaussian_params):
            results.append(
                self._evaluate_candidate(
                    world,
                    params,
                    f"candidate_gaussian_{index:03d}",
                    "gaussian",
                    logdir,
                    use_vision,
                )
            )

        baseline = results[0]
        raw_proposal = results[1]
        gaussian_results = results[2:]
        gaussian_rewards = np.array(
            [candidate.reward for candidate in gaussian_results], dtype=float
        )
        best_gaussian_reward = float(np.max(gaussian_rewards))
        elite_order = np.argsort(gaussian_rewards)[::-1][: self.cma.mu]
        weighted_elite_reward = float(
            np.sum(self.cma.weights * gaussian_rewards[elite_order])
        )

        if (
            projection.mahalanobis_distance >= self.cma_restart_distance
            and raw_proposal.reward
            > max(baseline.reward, best_gaussian_reward) + self.cma_restart_margin
        ):
            self.cma.restart(raw_proposal.params)
            update_status = "restart_at_reward_validated_remote_llm_proposal"
        elif weighted_elite_reward > baseline.reward + self.cma_acceptance_margin:
            self.cma.tell(gaussian_params, gaussian_rewards, projection.center)
            update_status = "accepted_reward_ranked_gaussian_update"
        elif raw_proposal.reward > baseline.reward + self.cma_acceptance_margin:
            self.cma.relocate(raw_proposal.params, reset_paths=True)
            update_status = "accepted_reward_validated_llm_proposal"
        else:
            update_status = "rejected_guidance_kept_previous_cma_state"

        # Keep the exact BMPS history contract: complete params and scalar reward.
        for candidate in results:
            self.replay_buffer.add(candidate.params.copy(), candidate.reward)

        if use_vision:
            landscape = self._build_pairwise_landscape(results, logdir, lambda_t)
            if landscape:
                self.pending_behavioral_landscape = landscape

        best_result = max(results, key=lambda candidate: candidate.reward)
        self.policy.update_policy(best_result.params)
        self.last_generation_results = results
        self._log_cma_state(
            logdir,
            projection,
            update_status,
            baseline.reward,
            raw_proposal.reward,
            best_gaussian_reward,
            weighted_elite_reward,
        )

        print(
            f"[BMPS-CMA] {update_status}; baseline={baseline.reward:.2f}, "
            f"proposal={raw_proposal.reward:.2f}, best Gaussian={best_gaussian_reward:.2f}"
        )

        self.training_episodes += 1
        if self.enable_vision:
            self.visual_guidance.increment_iteration()

        cpu_time = time.process_time() - self.start_time
        api_time = self.api_call_time + self.vlm_api_time
        return (
            cpu_time,
            api_time,
            self.total_episodes,
            self.total_steps,
            best_result.reward,
        )
