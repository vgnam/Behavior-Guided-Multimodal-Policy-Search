import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x)).
    This mirrors the ranking behavior used in OpenAI's ES starter.
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=np.int32)
    ranks[x.argsort()] = np.arange(len(x), dtype=np.int32)
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    if x.size > 1:
        y /= (x.size - 1)
    y -= 0.5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = None
    num_items_summed = 0

    for batch_weights, batch_vecs in zip(
        itergroups(weights, batch_size),
        itergroups(vecs, batch_size),
    ):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        batch_dot = np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32),
        )
        if total is None:
            total = batch_dot
        else:
            total += batch_dot
        num_items_summed += len(batch_weights)

    if total is None:
        total = np.array([], dtype=np.float32)

    return total, num_items_summed


class OptimizerFlat:
    def __init__(self, dim):
        self.dim = int(dim)
        self.t = 0

    def update(self, theta, globalg):
        self.t += 1
        step = self._compute_step(np.asarray(globalg, dtype=np.float32))
        denom = max(float(np.linalg.norm(theta)), 1e-8)
        ratio = float(np.linalg.norm(step) / denom)
        return np.asarray(theta, dtype=np.float32) + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGDFlat(OptimizerFlat):
    def __init__(self, dim, stepsize, momentum=0.9):
        super().__init__(dim)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize = float(stepsize)
        self.momentum = float(momentum)

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class AdamFlat(OptimizerFlat):
    def __init__(self, dim, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(dim)
        self.stepsize = float(stepsize)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
