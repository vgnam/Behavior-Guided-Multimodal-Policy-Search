import numpy as np
from agent.policy.base_policy import Policy


class ValueBasedPolicy(Policy):
    """Tabular Q-value policy for discrete state/action spaces.

    Parameters are a flat vector of length n_states * n_actions,
    reshaped as a Q-table: theta[state, action].
    Policy: pi(s) = argmax_a theta[s, a].
    """

    def __init__(self, n_states, n_actions):
        # Keep signature compatible with Policy base.
        super().__init__(states=list(range(n_states)), actions=list(range(n_actions)))
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.theta = np.zeros(self.n_states * self.n_actions, dtype=np.float32)

    def initialize_policy(self):
        self.theta = np.random.normal(0.0, 0.5, size=self.n_states * self.n_actions).astype(np.float32)

    def get_action(self, state):
        """state: int state index (or scalar array)."""
        idx = int(np.asarray(state).reshape(-1)[0])
        idx = max(0, min(idx, self.n_states - 1))
        start = idx * self.n_actions
        q_vals = self.theta[start : start + self.n_actions]
        return int(np.argmax(q_vals))

    def update_policy(self, params_flat):
        if params_flat is None:
            return
        self.theta = np.array(params_flat, dtype=np.float32).reshape(-1)
        if self.theta.size != self.n_states * self.n_actions:
            raise ValueError(
                f"ValueBasedPolicy expects {self.n_states * self.n_actions} params, "
                f"got {self.theta.size}"
            )

    def get_parameters(self):
        return self.theta

    def __str__(self):
        lines = ["State | Action | Q-values"]
        for s in range(self.n_states):
            start = s * self.n_actions
            q_vals = self.theta[start : start + self.n_actions]
            best_a = int(np.argmax(q_vals))
            q_str = ", ".join(f"{v:.3f}" for v in q_vals)
            lines.append(f"{s} | {best_a} | [{q_str}]")
        return "\n".join(lines)
