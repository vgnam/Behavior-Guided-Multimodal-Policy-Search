import time
from jinja2 import Template
from litellm import completion


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_model_name = llm_model_name
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
    #             # Append assistant response to conversation history
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

    # --- Các phương thức cập nhật tham số (giữ nguyên logic, chỉ sửa phần LLM) ---

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

    # --- Các phương thức còn lại giữ nguyên logic, chỉ đảm bảo không dùng model_group ---
    # (Bạn có thể áp dụng cùng mẫu như trên cho các hàm còn lại)

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

    def llm_update_parameters_num_optim_vision(
        self, episode_reward_buffer, parse_parameters, step_number, env_desc_file,
        visual_analysis, lambda_t,
        rank=None, optimum=None, search_step_size=0.1, actions=None, visual_params=None,
        best_visual_analysis=None, best_visual_entry=None, neighborhood_analysis=None,
        poisson_lam=2.0, neighbor_step=0.1
    ):
        """
        Update parameters using vision-guided feedback (ProPS-V).
        
        Implements Eq. (4): θ ← LLM(Γ, P, Ψ, λ_t)
        
        Args:
            episode_reward_buffer: String of past parameters and rewards (Γ)
            parse_parameters: Function to parse LLM output
            step_number: Current iteration number
            env_desc_file: Environment description
            visual_analysis: VLM analysis ψ_t (can be None)
            lambda_t: Current guidance coefficient
            guidance_phase: Description of current phase
            rank: Number of parameters
            optimum: Expected optimal reward
            search_step_size: Step size for exploration
            actions: Action space description
            
        Returns:
            Tuple of (parsed_params, log, api_time)
        """
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render({
            "episode_reward_buffer_string": str(episode_reward_buffer),
            "env_description": env_desc_file,
            "step_number": str(step_number),
            "rank": rank,
            "optimum": str(optimum),
            "step_size": str(search_step_size),
            "actions": actions,
            "visual_analysis": visual_analysis if visual_analysis else "No visual analysis available for this iteration.",
            "lambda_t": f"{lambda_t:.3f}",
            "has_visual": visual_analysis is not None,
            "visual_params": visual_params,
            "best_visual_analysis": best_visual_analysis,
            "best_visual_iter": best_visual_entry['iteration'] if best_visual_entry else None,
            "best_visual_reward": f"{best_visual_entry['reward']:.2f}" if best_visual_entry else None,
            "best_visual_params": best_visual_entry['params'] if best_visual_entry else None,
            "has_best_visual": best_visual_analysis is not None,
            "neighborhood_analysis": neighborhood_analysis,
            "has_neighborhood": neighborhood_analysis is not None,
            "poisson_lam": poisson_lam,
            "neighbor_step": neighbor_step,
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
        actions, num_states, optimum,
        neighborhood_analysis=None, poisson_lam=2.0, neighbor_step=0.1
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
            guidance_phase: Visual guidance instruction
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
            "neighborhood_analysis": neighborhood_analysis,
            "has_neighborhood": neighborhood_analysis is not None,
            "poisson_lam": poisson_lam,
            "neighbor_step": neighbor_step,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time