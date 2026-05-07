import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions

        self.weight = np.random.rand(self.dim_states, self.dim_actions)

    def initialize_policy(self):
        self.weight = np.round(np.random.normal(0., 3., size=(self.dim_states, self.dim_actions)), 1)

    def get_action(self, state):
        state = state.T
        return np.matmul(state, self.weight)

    def __str__(self):
        output = "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        return output

    def update_policy(self, weight_list):
        if weight_list is None:
            return

        self.weight = np.array(weight_list).reshape(self.dim_states, self.dim_actions)
    
    def get_parameters(self):
        return self.weight
