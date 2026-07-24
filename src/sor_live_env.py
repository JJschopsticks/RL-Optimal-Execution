# sor_live_env.py
#
# Live counterpart of sor_env.py: turns SmartOrderRouterLive's obs dict into
# the same fixed-order float vector the trained PPO model expects. Exposes
# .action_space/.engine/.feature_names identically to SORGymEnv so the
# existing rule-based policy functions in eval_policies.py (dump_policy,
# baseline_twap_policy, twap_policy, no_trade_policy) run against it
# unmodified. Not a gymnasium.Env subclass -- the async engine underneath
# doesn't fit gym's synchronous step()/reset() API.

import numpy as np
from gymnasium import spaces

from feature_spec import FEATURE_NAMES
from live_engine import SmartOrderRouterLive
from live_order_book import LiveOrderBook


class SORLiveEnv:
    def __init__(self, order_book: LiveOrderBook, **engine_kwargs):
        self.engine = SmartOrderRouterLive(order_book, **engine_kwargs)
        self.feature_names = FEATURE_NAMES
        self.action_space = spaces.Discrete(len(self.engine.action_fractions))

    def _obs_to_vector(self, obs_dict):
        return np.array([obs_dict.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    async def reset(self, tick=None):
        obs_dict = await self.engine.reset(tick=tick)
        return self._obs_to_vector(obs_dict)

    async def step(self, action, tick=None):
        obs_dict, reward, done, info = await self.engine.astep(action, tick=tick)
        return self._obs_to_vector(obs_dict), reward, done, info

    def close(self):
        pass
