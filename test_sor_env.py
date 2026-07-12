# test_sor_env.py

import gymnasium as gym
from sor_env import SORGymEnv

env = SORGymEnv("2026-07-11")

obs, info = env.reset()
print("Initial obs:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Action:", action, "Reward:", reward, "Obs[0:5]:", obs[:5])
    if terminated or truncated:
        break

env.close()
