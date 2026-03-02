import time
from jinja2 import Template
from litellm import completion


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
        credit_assignment_template: Template = None,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_model_name = llm_model_name
        self.credit_assignment_template = credit_assignment_template
        self.llm_conversation = []

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text: str, role: str):
        """Always use OpenAI-style messages. LiteLLM handles conversion internally."""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}. Use 'user', 'assistant', or 'system'.")
        self.llm_conversation.append({"role": role, "content": text})

    # def query_llm(self, temperature=1.0):
    #     for attempt in range(5):
    #         try:
    #             response = completion(
    #                 model=self.llm_model_name,
    #                 messages=self.llm_conversation,
    #                 temperature=temperature,
    #                 timeout=60,
    #             )
    #             text = response["choices"][0]["message"]["content"]
    #             self.add_llm_conversation(text, "assistant")
    #             return text
    #         except Exception as e:
    #             print(f"[LLM ERROR] Attempt {attempt + 1}/5: {e}")
    #             if attempt == 4:
    #                 raise RuntimeError("Failed to get LLM response after 5 attempts") from e
    #             time.sleep(10)
    #     return ""  # unreachable
    #
    # def query_llm_multiple_response(self, num_responses: int, temperature=1.0):
    #     for attempt in range(3):
    #         try:
    #             response = completion(
    #                 model=self.llm_model_name,
    #                 messages=self.llm_conversation,
    #                 n=num_responses,
    #                 temperature=temperature,
    #                 timeout=60,
    #             )
    #             responses = [choice.message.content for choice in response.choices]
    #             if len(responses) == num_responses:
    #                 return responses
    #             else:
    #                 raise ValueError(f"Expected {num_responses} responses, got {len(responses)}")
    #         except Exception as e:
    #             print(f"[LLM MULTIPLE ERROR] Attempt {attempt + 1}/3: {e}")
    #             if attempt == 2:
    #                 raise RuntimeError("Failed to get multiple LLM responses after 3 attempts") from e
    #             time.sleep(5)
    #     return []

    def query_llm(self, temperature=1.0):
        for attempt in range(5):
            try:
                response = completion(
                    model="openai/gpt-oss-120b",  # ví dụ: "openai/DeepSeek-V3.2-Speciale"
                    messages=self.llm_conversation,
                    temperature=temperature,
                    timeout=60,
                    api_key="sk-uDXg03MCrYREykzUKG0g2kHPZFjDmIvAShRkL1q0dCdohnxf",
                    api_base="https://mkp-api.fptcloud.com/v1",
                    stream=False,  # vì bạn đang lấy full text
                )

                text = response["choices"][0]["message"]["content"]

                # Append assistant response to conversation history
                self.add_llm_conversation(text, "assistant")
                return text

            except Exception as e:
                print(f"[LLM ERROR] Attempt {attempt + 1}/5: {e}")
                if attempt == 4:
                    raise RuntimeError("Failed to get LLM response after 5 attempts") from e
                time.sleep(10)

        return ""  # unreachable

    def query_llm_multiple_response(self, num_responses: int, temperature=1.0):
        for attempt in range(3):
            try:
                response = completion(
                    model="openai/DeepSeek-R1",
                    messages=self.llm_conversation,
                    n=num_responses,
                    temperature=temperature,
                    timeout=60,
                    api_key="sk-uDXg03MCrYREykzUKG0g2kHPZFjDmIvAShRkL1q0dCdohnxf",
                    api_base="https://mkp-api.fptcloud.com/v1",
                    stream=False,
                )

                responses = [
                    choice["message"]["content"]
                    for choice in response["choices"]
                ]

                if len(responses) == num_responses:
                    return responses
                else:
                    raise ValueError(
                        f"Expected {num_responses} responses, got {len(responses)}"
                    )

            except Exception as e:
                print(f"[LLM MULTIPLE ERROR] Attempt {attempt + 1}/3: {e}")
                if attempt == 2:
                    raise RuntimeError(
                        "Failed to get multiple LLM responses after 3 attempts"
                    ) from e
                time.sleep(5)

        return []

    def parse_parameters(self, parameters_string: str):
        new_parameters_list = []
        for row in parameters_string.split("\n"):
            row = row.strip().strip(",")
            if row:
                try:
                    parameters_row = [float(x.strip().strip(",")) for x in row.split(",")]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(f"[PARSE ERROR] Row: {row} | Error: {e}")
        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render(
            {"replay_buffer_string": str(replay_buffer), "parameters_string": str(parameters)}
        )
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()

        self.add_llm_conversation(self.llm_output_conversion_template.render(), "user")
        raw_params = self.query_llm()

        parser = parse_parameters if parse_parameters is not None else self.parse_parameters
        parsed_params = parser(raw_params)
        return parsed_params, [reasoning, raw_params]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()

        self.add_llm_conversation(self.llm_output_conversion_template.render(), "user")
        raw_params = self.query_llm()

        parser = parse_parameters if parse_parameters is not None else self.parse_parameters
        parsed_params = parser(raw_params)
        return parsed_params, ["system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning, raw_params]

    def llm_update_parameters_num_optim(
        self, episode_reward_buffer, parse_parameters, step_number,
        rank=None, optimum=None, search_step_size=0.1, actions=None
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time

    def llm_update_parameters_num_optim_q_table(
        self, episode_reward_buffer, parse_parameters, step_number, actions, num_states, optimum
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "actions": actions,
            "rank": num_states,
            "optimum": str(optimum),
        })
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log

    def llm_update_parameters_num_optim_imitation(
        self, demonstrations_str, episode_reward_buffer, parse_parameters, step_number, search_std
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "expert_demonstration_string": demonstrations_str,
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "search_std": str(search_std),
        })
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log

    def llm_propose_parameters_num_optim_based_on_anchor(
        self, episode_reward_buffer, parse_parameters, step_number, search_std, anchor_parameters
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "search_std": str(search_std),
            "anchor_parameters": str(anchor_parameters),
        })
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log

    def llm_propose_multiple_parameters_num_optim_based_on_anchor(
        self, episode_reward_buffer, parse_parameters, step_number,
        search_std, anchor_parameters, num_candidates, temperature
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "search_std": str(search_std),
            "anchor_parameters": str(anchor_parameters),
        })
        self.add_llm_conversation(system_prompt, "user")
        reasoning_list = self.query_llm_multiple_response(num_candidates, temperature)
        param_list = [parse_parameters(r) for r in reasoning_list]
        return system_prompt, param_list, reasoning_list

    def llm_propose_parameters_num_optim_based_on_anchor_thread(
        self, new_candidates, new_idx, episode_reward_buffer, parse_parameters,
        step_number, search_std, anchor_parameters
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "step_number": str(step_number),
            "search_std": str(search_std),
            "anchor_parameters": str(anchor_parameters),
        })
        self.add_llm_conversation(system_prompt, "user")
        reasoning = self.query_llm()
        parsed = parse_parameters(reasoning)
        new_candidates[new_idx] = parsed
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed, log

    def llm_update_parameters_num_optim_semantics(
        self, episode_reward_buffer, parse_parameters, step_number, env_desc_file,
        rank=None, optimum=None, search_step_size=0.1, actions=None
    ):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "env_description": env_desc_file,
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time

    def llm_credit_assignment(
        self,
        policy_group: dict,
        env_desc_file: str,
        n_rollouts: int,
    ) -> tuple:
        """
        ProPS-V per-policy credit assignment step.

        Given one labeled policy's top-k and bottom-k rollout descriptions,
        identify which behavioral components are responsible for high vs low reward.

        Args:
            policy_group:   Dict with keys:
                              'label'         : str (e.g. 'current', 'best', 'worst')
                              'avg_reward'    : float
                              'best_rollouts' : list of {reward, description}
                              'worst_rollouts': list of {reward, description}
            env_desc_file:  Environment description string
            n_rollouts:     How many rollouts were run (for the template)

        Returns:
            Tuple of (credit_assignment_text, api_time)
        """
        env_str = env_desc_file if env_desc_file else "RL environment"

        if self.credit_assignment_template is not None:
            prompt = self.credit_assignment_template.render(
                env_description=env_str,
                label=policy_group.get('label', ''),
                avg_reward=policy_group.get('avg_reward', 0.0),
                best_rollouts=policy_group.get('best_rollouts', []),
                worst_rollouts=policy_group.get('worst_rollouts', []),
                n_rollouts=n_rollouts,
            )
        else:
            # Fallback inline prompt
            def _fmt(rollouts, tag):
                if not rollouts:
                    return f"({tag}: no rollouts available)"
                return "\n\n".join(
                    f"{tag} rollout {i+1} (reward={r['reward']:.2f}):\n{r['description']}"
                    for i, r in enumerate(rollouts)
                )
            label = policy_group.get('label', '').upper()
            avg = policy_group.get('avg_reward', 0.0)
            best_block  = _fmt(policy_group.get('best_rollouts',  []), f"{label} high-reward")
            worst_block = _fmt(policy_group.get('worst_rollouts', []), f"{label} low-reward")
            prompt = (
                f"You are analyzing rollouts of a reinforcement learning agent.\n"
                f"Environment: {env_str}\n\n"
                f"The {label} policy (average reward = {avg:.2f}) was rolled out {n_rollouts} times.\n\n"
                f"High-reward rollouts:\n{best_block}\n\n"
                f"Low-reward rollouts:\n{worst_block}\n\n"
                f"Compare high vs low reward rollouts. Identify which specific actions, decisions, "
                f"timing patterns, or behavioral components consistently appear in high-reward rollouts "
                f"and are absent or reversed in low-reward rollouts. Explain what each behavior is and "
                f"why it plausibly causes the reward difference. "
                f"Do not suggest parameter values. Write 4-6 concise sentences as a single paragraph."
            )

        self.reset_llm_conversation()
        self.add_llm_conversation(prompt, "user")
        api_start_time = time.time()
        text = self.query_llm()
        api_time = time.time() - api_start_time
        return text, api_time

    def llm_update_parameters_num_optim_vision(
        self, episode_reward_buffer, parse_parameters, step_number, env_desc_file,
        visual_analysis, lambda_t,
        rank=None, optimum=None, search_step_size=0.1, actions=None, visual_params=None,
        best_visual_analysis=None, best_visual_entry=None,
        worst_visual_analysis=None, worst_visual_entry=None,
        credit_assignment=None,
    ):
        """
        Update parameters using behavioral credit assignment feedback (ProPS-V).

        Implements Eq. (4): θ ← LLM(Γ, P, Ψ, λ_t)

        Args:
            episode_reward_buffer: String of past parameters and rewards (Γ)
            parse_parameters: Function to parse LLM output
            step_number: Current iteration number
            env_desc_file: Environment description
            visual_analysis: unused (kept for API compatibility)
            lambda_t: Current guidance coefficient
            rank: Number of parameters
            optimum: Expected optimal reward
            search_step_size: Step size for exploration
            actions: Action space description
            visual_params: unused (kept for API compatibility)
            best_visual_analysis: unused (kept for API compatibility)
            best_visual_entry: unused (kept for API compatibility)
            worst_visual_analysis: unused (kept for API compatibility)
            worst_visual_entry: unused (kept for API compatibility)
            credit_assignment: dict with keys 'current', 'best', 'worst' (text),
                               and 'current_reward', 'best_reward', 'worst_reward' (floats)

        Returns:
            Tuple of (parsed_params, log, api_time)
        """
        ca = credit_assignment or {}
        ca_current = ca.get('current') if isinstance(ca, dict) else None
        ca_best    = ca.get('best')    if isinstance(ca, dict) else None
        ca_worst   = ca.get('worst')   if isinstance(ca, dict) else None
        has_ca = bool(ca_current or ca_best or ca_worst)

        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "env_description": env_desc_file,
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
            "lambda_t": f"{lambda_t:.3f}",
            # Per-policy credit assignment outputs
            "has_credit_assignment": has_ca,
            "ca_current": ca_current or "",
            "ca_best":    ca_best    or "",
            "ca_worst":   ca_worst   or "",
            "ca_current_reward": ca.get('current_reward', 'N/A') if isinstance(ca, dict) else 'N/A',
            "ca_best_reward":    ca.get('best_reward',    'N/A') if isinstance(ca, dict) else 'N/A',
            "ca_worst_reward":   ca.get('worst_reward',   'N/A') if isinstance(ca, dict) else 'N/A',
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time

    # ------------------------------------------------------------------ #
    #  BL-ProPS helpers                                                    #
    # ------------------------------------------------------------------ #

    def llm_generate_behavioral_hypothesis(
        self,
        anchors: dict,
        history_str: str,
        step_number: int,
        env_description: str = None,
        rank: int = None,
        actions=None,
    ) -> tuple[str, str, float]:
        """
        BL-ProPS STEP 1: LLM → Behavioral Hypothesis b_hyp.

        Uses 'blprops_step1_hypothesis.j2'. The LLM is given the three
        behavioral anchor descriptions (worst / current / best) plus the
        full numerical history, and returns a structured b_hyp paragraph.

        Args:
            anchors: dict with keys 'worst', 'current', 'best', each a dict
                     with 'params', 'reward', 'description'
            history_str: Formatted numerical history string (θ_k, r_k)
            step_number: Current iteration index
            env_description: Jinja2 include path for env description (optional)
            rank: Number of parameters (optional, used by Q-table templates)
            actions: Valid action values (optional, used by Q-table templates)

        Returns:
            Tuple of (b_hyp_text, log_string, api_time)
        """
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "anchors": anchors,
            "history_str": history_str,
            "step_number": str(step_number),
            "env_description": env_description,
            "rank": rank,
            "actions": actions,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        b_hyp = self.query_llm()
        api_time = time.time() - api_start_time
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + b_hyp
        return b_hyp, log, api_time

    def llm_propose_parameters_blprops(
        self,
        b_post: str,
        anchors: dict,
        history_str: str,
        rank: int,
        step_number: int,
        parse_parameters,
        optimum: float = 1000.0,
        search_step_size: float = 0.1,
        env_description: str = None,
        actions=None,
    ) -> tuple:
        """
        BL-ProPS STEP 3: LLM → Parameter Proposal θ_{t+1}.

        Uses 'blprops_step3_proposal.j2'. Given the VLM behavioral posterior
        b_post, the three anchor policies, and the numerical history, the LLM
        performs inverse mapping (behavior → params) while respecting CLIFF /
        FLAT topology constraints.

        Args:
            b_post:            VLM behavioral posterior string (from Step 2)
            anchors:           Same anchor dict as in Step 1
            history_str:       Formatted numerical history string
            rank:              Number of policy parameters
            step_number:       Current iteration index
            parse_parameters:  Callable that extracts np.ndarray from LLM text
            optimum:           Expected optimal reward
            search_step_size:  Reference step size hint

        Returns:
            Tuple of (parsed_params, log_string, api_time)
        """
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "b_post": b_post,
            "anchors": anchors,
            "history_str": history_str,
            "rank": rank,
            "step_number": str(step_number),
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "env_description": env_description,
            "actions": actions,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time

    def llm_update_parameters_num_optim_q_table_vision(
        self, episode_reward_buffer, parse_parameters, step_number,
        env_desc_file, visual_analysis, lambda_t,
        actions, num_states, optimum
    ):
        """
        Update Q-table parameters using vision-guided feedback (ProPS-V for discrete states).

        Args:
            episode_reward_buffer: String of past Q-values and rewards
            parse_parameters: Function to parse LLM output
            step_number: Current iteration number
            env_desc_file: Environment description
            visual_analysis: VLM analysis (can be None)
            lambda_t: Current guidance coefficient
            actions: Action space description
            num_states: Number of states
            optimum: Expected optimal reward

        Returns:
            Tuple of (parsed_params, log, api_time)
        """
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "env_description": env_desc_file,
            "step_number": str(step_number),
            "actions": actions,
            "rank": num_states,
            "optimum": str(optimum),
            "visual_analysis": visual_analysis if visual_analysis else "No visual analysis available for this iteration.",
            "lambda_t": f"{lambda_t:.3f}",
            "has_visual": visual_analysis is not None,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time