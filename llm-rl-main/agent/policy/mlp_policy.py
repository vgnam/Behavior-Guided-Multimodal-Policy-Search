"""Small NumPy MLP policy for black-box policy parameter search."""

import numpy as np

from agent.policy.base_policy import Policy


class MLPPolicy(Policy):
    """Fully-connected policy with a flat parameter-vector interface."""

    SUPPORTED_ACTIVATIONS = {"identity", "relu", "tanh"}

    def __init__(
        self,
        dim_states,
        dim_actions,
        hidden_sizes=(32, 32),
        hidden_activation="tanh",
        output_activation="tanh",
        bias=True,
    ):
        super().__init__(dim_states, dim_actions)
        self.dim_states = int(dim_states)
        self.dim_actions = int(dim_actions)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("All hidden_sizes must be positive")
        self.hidden_activation = str(hidden_activation).lower()
        self.output_activation = str(output_activation).lower()
        for activation in (self.hidden_activation, self.output_activation):
            if activation not in self.SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"Unsupported activation '{activation}'. "
                    f"Choose from {sorted(self.SUPPORTED_ACTIVATIONS)}"
                )
        self.use_bias = bool(bias)
        self.layer_sizes = (
            self.dim_states,
            *self.hidden_sizes,
            self.dim_actions,
        )
        self.weights = []
        self.biases = []
        self.initialize_policy()

    @property
    def parameter_count(self):
        weight_count = sum(
            fan_in * fan_out
            for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        )
        bias_count = sum(self.layer_sizes[1:]) if self.use_bias else 0
        return weight_count + bias_count

    @staticmethod
    def _activate(values, activation):
        if activation == "tanh":
            return np.tanh(values)
        if activation == "relu":
            return np.maximum(values, 0.0)
        return values

    def initialize_policy(self):
        """Use Xavier-uniform initialization to avoid immediately saturated tanh."""
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(
                np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            )
            if self.use_bias:
                self.biases.append(np.zeros((1, fan_out), dtype=float))

    def get_action(self, state):
        values = np.asarray(state, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim != 2:
            raise ValueError(f"Expected a 1-D or 2-D state, got shape {values.shape}")
        if values.shape[-1] != self.dim_states:
            if values.shape[0] == self.dim_states:
                values = values.T
            else:
                raise ValueError(
                    f"Expected state dimension {self.dim_states}, got {values.shape}"
                )

        for layer_idx, weight in enumerate(self.weights):
            values = values @ weight
            if self.use_bias:
                values = values + self.biases[layer_idx]
            activation = (
                self.output_activation
                if layer_idx == len(self.weights) - 1
                else self.hidden_activation
            )
            values = self._activate(values, activation)
        return values

    def get_parameters(self):
        flat_parts = []
        for layer_idx, weight in enumerate(self.weights):
            flat_parts.append(weight.reshape(-1))
            if self.use_bias:
                flat_parts.append(self.biases[layer_idx].reshape(-1))
        return np.concatenate(flat_parts)

    def update_policy(self, parameters):
        if parameters is None:
            return
        parameters = np.asarray(parameters, dtype=float).reshape(-1)
        if parameters.size != self.parameter_count:
            raise ValueError(
                f"Expected {self.parameter_count} MLP parameters, "
                f"got {parameters.size}"
            )

        offset = 0
        new_weights = []
        new_biases = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight_count = fan_in * fan_out
            new_weights.append(
                parameters[offset:offset + weight_count].reshape(fan_in, fan_out)
            )
            offset += weight_count
            if self.use_bias:
                new_biases.append(
                    parameters[offset:offset + fan_out].reshape(1, fan_out)
                )
                offset += fan_out
        self.weights = new_weights
        self.biases = new_biases

    def __str__(self):
        lines = [
            f"MLPPolicy(layer_sizes={self.layer_sizes}, "
            f"hidden_activation={self.hidden_activation}, "
            f"output_activation={self.output_activation})"
        ]
        for layer_idx, weight in enumerate(self.weights):
            lines.append(f"Layer {layer_idx} weights:\n{weight}")
            if self.use_bias:
                lines.append(f"Layer {layer_idx} bias:\n{self.biases[layer_idx]}")
        return "\n".join(lines)
