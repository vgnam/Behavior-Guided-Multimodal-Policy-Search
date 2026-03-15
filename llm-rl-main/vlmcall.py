import gymnasium as gym

for env in gym.envs.registry:
    print(env)