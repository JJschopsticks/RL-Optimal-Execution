# sor_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from replay_engine import SmartOrderRouterReplay


class SORGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, date_str="2026-07-11", **engine_kwargs):
        super().__init__()

        self.engine = SmartOrderRouterReplay(date_str=date_str, **engine_kwargs)

        self.action_space = spaces.Discrete(len(self.engine.action_fractions))

        self.feature_names = [
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "midprice",
            "spread",
            "cash",
            "inventory",
            "portfolio_value",
            "return_1",
            "return_5",
            "return_10",
            "vol_5",
            "vol_10",
            "ofi_price",
            "ofi_qty",
            "ofi_combined",
            "depth_imbalance",
            "microprice",
            "queue_imbalance",
            "total_depth",
            "spread_ratio",
            "time_fraction",
            "twap_target",
            "twap_diff",
            "vwap_price",
            "remaining_inventory",
            "remaining_inventory_ratio",
            "remaining_time",
            "execution_cost",
            "market_impact",
            "slippage",
            "perm_impact",
            "bid_depth_5",
            "ask_depth_5",
            "book_imbalance_5",
            "impact_bps_small",
            "impact_bps_large",
        ]

        low = np.full(len(self.feature_names), -1e9, dtype=np.float32)
        high = np.full(len(self.feature_names), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _obs_to_vector(self, obs):
        return np.array([obs.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.engine.seed(seed)
        obs_dict = self.engine.reset()
        return self._obs_to_vector(obs_dict), {}

    def step(self, action):
        obs_dict, reward, done, info = self.engine.step(action)
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
