from typing import Any


class Policy:
    def __init__(self, states: Any, actions: Any):
        self.states = states
        self.actions = actions
        self.mapping = {}

    def get_action(self, state):
        return self.mapping[state]

    def __str__(self):
        table = ["States\t\tAction"]
        for key, val in self.mapping.items():
            table.append(f"{key}\t\t{val}")
        return "\n".join(table)

    def initialize_policy(self):
        pass

    # Keep the old misspelled name for compatibility with older callers.
    def initlize_policy(self):
        self.initialize_policy()

    def update_policy(self, *args, **kwargs):
        pass
