# sor_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from replay_engine import SmartOrderRouterReplay  # or replay_engine_v2_sor if that's your file name


class SORGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, date_str: str = "2026-07-11"):
        super().__init__()

        # Underlying replay engine
        self.engine = SmartOrderRouterReplay(date_str=date_str)

        # Actions: 0 = hold, 1 = market_buy, 2 = market_sell
        self.action_space = spaces.Discrete(3)

        # Observation: vectorized version of obs dict
        # [best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
        #  midprice, spread, cash, inventory, portfolio_value]
        low = np.array([0.0] * 9, dtype=np.float32)
        high = np.array([1e9] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _obs_to_vector(self, obs_dict):
        return np.array([
            obs_dict.get("best_bid_price", 0.0),
            obs_dict.get("best_bid_qty", 0.0),
            obs_dict.get("best_ask_price", 0.0),
            obs_dict.get("best_ask_qty", 0.0),
            obs_dict.get("midprice", 0.0),
            obs_dict.get("spread", 0.0),
            obs_dict.get("cash", 0.0),
            obs_dict.get("inventory", 0.0),
            obs_dict.get("portfolio_value", 0.0),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.engine.reset()
        obs = self._obs_to_vector(obs_dict)
        info = {}
        return obs, info

    def step(self, action):
        # Map integer action to string for replay engine
        if action == 0:
            act_str = "hold"
        elif action == 1:
            act_str = "market_buy"
        elif action == 2:
            act_str = "market_sell"
        else:
            act_str = "hold"

        obs_dict, reward, done, info = self.engine.step(action=act_str)
        obs = self._obs_to_vector(obs_dict)

        terminated = done
        truncated = False  # you can add time limits later

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Optional: print current state
        pass

    def close(self):
        self.engine.close()
