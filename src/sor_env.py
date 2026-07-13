# sor_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from replay_engine import SmartOrderRouterReplay


class SORGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, date_str="2026-07-11"):
        super().__init__()

        self.engine = SmartOrderRouterReplay(date_str=date_str)

        self.action_space = spaces.Discrete(3)

        # 22 features:
        # 9 base + 3 returns + 2 vol + 3 OFI + 1 depth + 4 micro/queue/depth/spread_ratio
        low = np.array([0.0] * 22, dtype=np.float32)
        high = np.array([1e9] * 22, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _obs_to_vector(self, obs):
        return np.array([
            obs.get("best_bid_price", 0.0),
            obs.get("best_bid_qty", 0.0),
            obs.get("best_ask_price", 0.0),
            obs.get("best_ask_qty", 0.0),
            obs.get("midprice", 0.0),
            obs.get("spread", 0.0),
            obs.get("cash", 0.0),
            obs.get("inventory", 0.0),
            obs.get("portfolio_value", 0.0),
            obs.get("return_1", 0.0),
            obs.get("return_5", 0.0),
            obs.get("return_10", 0.0),
            obs.get("vol_5", 0.0),
            obs.get("vol_10", 0.0),
            obs.get("ofi_price", 0.0),
            obs.get("ofi_qty", 0.0),
            obs.get("ofi_combined", 0.0),
            obs.get("depth_imbalance", 0.0),
            obs.get("microprice", 0.0),
            obs.get("queue_imbalance", 0.0),
            obs.get("total_depth", 0.0),
            obs.get("spread_ratio", 0.0),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.engine.reset()
        return self._obs_to_vector(obs_dict), {}

    def step(self, action):
        if action == 0:
            act = "hold"
        elif action == 1:
            act = "market_buy"
        else:
            act = "market_sell"

        obs_dict, reward, done, info = self.engine.step(act)
        obs = self._obs_to_vector(obs_dict)

        return obs, reward, done, False, info

    def close(self):
        self.engine.close()


def main():
    env = SORGymEnv("2026-07-11")
    obs, info = env.reset()
    print("Initial obs:", obs)

    for _ in range(10):
        a = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(a)
        print("Action:", a,
              "| Reward:", reward,
              "| Returns:", obs[9:12],
              "| Vol:", obs[12:14],
              "| OFI:", obs[14:17],
              "| Depth:", obs[17],
              "| Micro/Queue:", obs[18:20],
              "| TotalDepth/SpreadRatio:", obs[20:22])
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
