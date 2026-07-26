import time
from jinja2 import Template
from litellm import completion
import litellm


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
        llm_api_key: str = None,
        llm_api_base: str = None,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_model_name = llm_model_name
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.llm_conversation = []

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text: str, role: str):
        """Always use OpenAI-style messages. LiteLLM handles conversion internally."""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}. Use 'user', 'assistant', or 'system'.")
        self.llm_conversation.append({"role": role, "content": text})

    def _build_completion_kwargs(self, temperature, extra_kwargs=None):
        kwargs = {
            "model": self.llm_model_name,
            "messages": self.llm_conversation,
            "temperature": temperature,
            "timeout": 60,
        }
        if self.llm_api_key is not None:
            kwargs["api_key"] = self.llm_api_key
        if self.llm_api_base is not None:
            kwargs["api_base"] = self.llm_api_base
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    def query_llm(self, temperature=1.0):
        for attempt in range(5):
            try:
                response = completion(**self._build_completion_kwargs(temperature))
                text = response["choices"][0]["message"]["content"]
                # Extract token usage
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, 'usage') and response.usage is not None:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                # Append assistant response to conversation history
                self.add_llm_conversation(text, "assistant")
                return text, prompt_tokens, completion_tokens
            except Exception as e:
                print(f"[LLM ERROR] Attempt {attempt + 1}/5: {e}")
                if attempt == 4:
                    raise RuntimeError("Failed to get LLM response after 5 attempts") from e
                time.sleep(10)
        return "", 0, 0  # unreachable
    
    def query_llm_multiple_response(self, num_responses: int, temperature=1.0):
        for attempt in range(3):
            try:
                response = completion(
                    **self._build_completion_kwargs(
                        temperature, extra_kwargs={"n": num_responses}
                    )
                )
                responses = [choice.message.content for choice in response.choices]
                # Extract token usage
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, 'usage') and response.usage is not None:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                if len(responses) == num_responses:
                    return responses, prompt_tokens, completion_tokens
                else:
                    raise ValueError(f"Expected {num_responses} responses, got {len(responses)}")
            except Exception as e:
                print(f"[LLM MULTIPLE ERROR] Attempt {attempt + 1}/3: {e}")
                if attempt == 2:
                    raise RuntimeError("Failed to get multiple LLM responses after 3 attempts") from e
                time.sleep(5)
        return [], 0, 0

    def parse_parameters(self, parameters_string: str):
        import re
        new_parameters_list = []
        # Try to find a bracketed list first: params[0:N] = [...] or just [...]
        bracket_match = re.search(r'\[([^\[\]]+)\]', parameters_string)
        if bracket_match:
            try:
                values = [float(x.strip()) for x in bracket_match.group(1).split(",") if x.strip()]
                if values:
                    return [values]
            except Exception as e:
                print(f"[PARSE ERROR] bracket list: {e}")
        # Fallback: original row-by-row comma parsing
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
        reasoning, pt1, ct1 = self.query_llm()

        self.add_llm_conversation(self.llm_output_conversion_template.render(), "user")
        raw_params, pt2, ct2 = self.query_llm()

        parser = parse_parameters if parse_parameters is not None else self.parse_parameters
        parsed_params = parser(raw_params)
        return parsed_params, [reasoning, raw_params]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()
        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )
        self.add_llm_conversation(system_prompt, "user")
        reasoning, pt1, ct1 = self.query_llm()

        self.add_llm_conversation(self.llm_output_conversion_template.render(), "user")
        raw_params, pt2, ct2 = self.query_llm()

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
        reasoning, llm_prompt_tokens, llm_completion_tokens = self.query_llm()
        api_time = time.time() - api_start_time

        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time, llm_prompt_tokens, llm_completion_tokens

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
        reasoning, llm_prompt_tokens, llm_completion_tokens = self.query_llm()
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, llm_prompt_tokens, llm_completion_tokens

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
        reasoning, _pt, _ct = self.query_llm()
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
        reasoning, _pt, _ct = self.query_llm()
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
        reasoning_list, _pt, _ct = self.query_llm_multiple_response(num_candidates, temperature)

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
        reasoning, _pt, _ct = self.query_llm()
        parsed = parse_parameters(reasoning)
        new_candidates[new_idx] = parsed
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed, log

    def llm_update_parameters_num_optim_semantics(
        self, episode_reward_buffer, parse_parameters, step_number, env_desc_file,
        rank=None, optimum=None, search_step_size=0.1, actions=None,
        latent_mode=False, full_parameter_dim=None, projection_scale=1.0,
        policy_type="linear",
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
            "latent_mode": latent_mode,
            "full_parameter_dim": full_parameter_dim,
            "projection_scale": projection_scale,
            "policy_type": policy_type,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning, llm_prompt_tokens, llm_completion_tokens = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time, llm_prompt_tokens, llm_completion_tokens

    def llm_update_parameters_num_optim_vision(
        self, episode_reward_buffer, parse_parameters, step_number, env_desc_file,
        rank=None, optimum=None, search_step_size=0.1, actions=None,
        neighborhood_analysis=None, latent_mode=False,
        full_parameter_dim=None, projection_scale=1.0,
        policy_type="linear",
    ):
        """
        Update parameters using vision-guided feedback (BMPS).
        
        Unified method for both linear policy and Q-table agents.
        The template distinguishes between them using the `actions` variable:
        - actions is not None → Q-table (integer params from actions[0])
        - actions is None → linear policy (float params [-6.0, 6.0])
        
        Args:
            episode_reward_buffer: String of past parameters and rewards
            parse_parameters: Function to parse LLM output
            step_number: Current iteration number
            env_desc_file: Environment description template path
            rank: Number of parameters
            optimum: Expected optimal reward
            search_step_size: Step size for exploration (linear policy only)
            actions: Action space description (Q-table only, None for linear)
            neighborhood_analysis: Formatted neighborhood landscape string (None if VLM not invoked)
            latent_mode: Whether params represent a low-dimensional update z
            full_parameter_dim: Dimension D of the decoded policy vector
            projection_scale: Alpha used by theta_new = theta + alpha * A @ z
            
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
            "neighborhood_analysis": neighborhood_analysis,
            "has_neighborhood": neighborhood_analysis is not None,
            "latent_mode": latent_mode,
            "full_parameter_dim": full_parameter_dim,
            "projection_scale": projection_scale,
            "policy_type": policy_type,
        })
        self.add_llm_conversation(system_prompt, "user")
        api_start_time = time.time()
        reasoning, llm_prompt_tokens, llm_completion_tokens = self.query_llm()
        api_time = time.time() - api_start_time
        parsed_params = parse_parameters(reasoning)
        log = "system:\n" + system_prompt + "\n\n\nLLM:\n" + reasoning
        return parsed_params, log, api_time, llm_prompt_tokens, llm_completion_tokens
