import numpy as np


def print_reward_summary(results):
    rewards = [float(result) for result in results]
    arr = np.asarray(rewards, dtype=float)

    if arr.size == 0:
        raise ValueError("results must contain at least one reward")

    mean_reward = float(arr.mean())
    std_reward = float(arr.std())

    print(f"Results: {rewards}")
    print(f"mean reward = {mean_reward:.3f} +/- {std_reward:.3f}")
    return mean_reward, std_reward
