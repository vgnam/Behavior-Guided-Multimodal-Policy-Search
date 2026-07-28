import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from agent.llm_num_optim_linear_policy_vision import LLMNumOptimVisionAgent


class _TrackingAnalyzer:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def analyze_frames(self, frames, env, reward, terminated, current_params):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return current_params, 0.01, 2, 3


def _make_agent(parallel):
    agent = LLMNumOptimVisionAgent.__new__(LLMNumOptimVisionAgent)
    agent.parallel_vlm_calls = parallel
    agent.vlm_max_workers = 2
    agent.random_subspace = None
    agent.rank = 1
    agent.n_neighbors = 2
    agent.max_traj_length = 10
    agent.env_desc_file = "test environment"
    agent.policy = SimpleNamespace(update_policy=lambda params: None)
    agent._generate_neighbors = lambda params, count: [
        (np.array([1.0]), np.array([1.0])),
        (np.array([2.0]), np.array([2.0])),
    ]
    agent.rollout_episode = lambda world, log, record, capture_frames: (
        1.0,
        [{"frame": object()}],
        False,
    )
    agent.frame_sampler = SimpleNamespace(
        sample_frames=lambda trajectory, terminated, max_length: [0],
        get_frames=lambda trajectory, indices: trajectory,
    )
    agent.vlm_analyzer = _TrackingAnalyzer()
    agent.vlm_api_time = 0.0
    agent.api_call_time = 0.0
    agent.total_vlm_prompt_tokens = 0
    agent.total_vlm_completion_tokens = 0
    return agent


@pytest.mark.parametrize(
    ("parallel", "expected_concurrency"),
    [(False, 1), (True, 2)],
)
def test_anchor_and_neighbor_vlm_calls_are_configurable(
    tmp_path, parallel, expected_concurrency
):
    agent = _make_agent(parallel)

    anchor, neighbors = agent._rollout_neighbors(
        object(), np.array([0.0]), "current", str(tmp_path)
    )

    assert agent.vlm_analyzer.max_active == expected_concurrency
    assert anchor["analysis"] == "params[0]: 0"
    assert {item["analysis"] for item in neighbors} == {
        "params[0]: 1",
        "params[0]: 2",
    }
    assert agent.total_vlm_prompt_tokens == 6
    assert agent.total_vlm_completion_tokens == 9
