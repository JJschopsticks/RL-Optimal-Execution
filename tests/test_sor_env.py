import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym
from sor_env import SORGymEnv

env = SORGymEnv("2026-07-11")
obs, info = env.reset()
assert obs.shape[0] == env.observation_space.shape[0], "Observation vector length mismatch"
assert env.action_space.n == len(env.engine.action_fractions), "Action space size mismatch"
assert env.action_space.n > 1, "Action space should contain sell fractions"
print("Initial obs length:", obs.shape[0])
print("Available actions:", list(range(env.action_space.n)))

for step_index in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    remaining_inventory = obs[env.feature_names.index("remaining_inventory")]
    time_fraction = obs[env.feature_names.index("time_fraction")]
    print(
        f"Step {step_index}: action={action}, reward={reward:.4f}, "
        f"remaining_inv={remaining_inventory:.6f}, time_frac={time_fraction:.3f}"
    )
    if terminated or truncated:
        print("Episode ended: terminated=", terminated, "truncated=", truncated)
        break

env.close()
