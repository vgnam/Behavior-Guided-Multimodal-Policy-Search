import numpy as np
from agent.policy.base_policy import Policy


class MLPPolicy(Policy):
    """
    Multi-layer perceptron policy parameterized as a flat numpy array.

    Architecture: input_dim -[tanh]-> hidden[0] -[tanh]-> ... -> output_dim (linear)

    Parameters layout (flat):
        Layer 0: W0 (in0 * h0) then b0 (h0)
        Layer 1: W1 (h0 * h1)  then b1 (h1)
        ...

    Optional obs_projection_dim: before the MLP, the input observation is projected
    via a fixed random matrix from dim_states → obs_projection_dim.  Useful when
    the raw observation is large (e.g. Atari pixel frames).
    """

    def __init__(
        self,
        dim_states: int,
        dim_actions: int,
        hidden_dims: list,
        obs_projection_dim: int = None,
        obs_proj_seed: int = 42,
    ):
        super().__init__(dim_states, dim_actions)
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.hidden_dims = list(hidden_dims)

        # Fixed random projection on observations (optional)
        if obs_projection_dim:
            rng = np.random.RandomState(obs_proj_seed)
            raw = rng.randn(dim_states, obs_projection_dim)
            # Orthonormalize columns so the projection preserves scale
            self.obs_proj_mat, _ = np.linalg.qr(raw)
            self.obs_proj_mat = self.obs_proj_mat[:, :obs_projection_dim]
            input_dim = obs_projection_dim
        else:
            self.obs_proj_mat = None
            input_dim = dim_states

        all_dims = [input_dim] + self.hidden_dims + [dim_actions]
        self.layer_shapes = [
            (all_dims[i], all_dims[i + 1]) for i in range(len(all_dims) - 1)
        ]
        self.param_count = sum(r * c + c for r, c in self.layer_shapes)

        self.layers = []  # list of (W, b)
        self.initialize_policy()

    def initialize_policy(self):
        params = np.random.normal(0.0, 0.3, self.param_count)
        self.update_policy(params)

    def update_policy(self, params_flat):
        params_flat = np.asarray(params_flat, dtype=float).reshape(-1)
        self.layers = []
        idx = 0
        for r, c in self.layer_shapes:
            W = params_flat[idx: idx + r * c].reshape(r, c)
            idx += r * c
            b = params_flat[idx: idx + c]
            idx += c
            self.layers.append((W.copy(), b.copy()))

    def get_parameters(self) -> np.ndarray:
        parts = []
        for W, b in self.layers:
            parts.append(W.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def get_action(self, state) -> np.ndarray:
        """
        state: shape (obs_dim, 1) or (1, obs_dim) or (obs_dim,)
        returns: shape (1, dim_actions)
        """
        x = np.asarray(state, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.T          # (obs_dim, 1) → (1, obs_dim)
        else:
            x = x.reshape(1, -1)

        if self.obs_proj_mat is not None:
            x = x @ self.obs_proj_mat   # (1, obs_projection_dim)

        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)    # relu on all hidden layers
        return x                        # linear output, shape (1, dim_actions)

    def __str__(self) -> str:
        out = ""
        for i, (W, b) in enumerate(self.layers):
            out += f"Layer {i} weights ({W.shape[0]}x{W.shape[1]}):\n"
            for row in W:
                out += ", ".join(f"{v:.5g}" for v in row) + "\n"
            out += f"Layer {i} bias:\n"
            out += ", ".join(f"{v:.5g}" for v in b) + "\n"
        return out
