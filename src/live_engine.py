# live_engine.py
#
# The live counterpart of SmartOrderRouterReplay (replay_engine.py): drives
# ExecutionCore's shared reward/impact/feature logic from a live order book
# instead of a historical file. Async because the next tick isn't already in
# memory -- it has to be awaited from an open websocket.
#
# Unlike replay, there's no file to clamp max_steps against -- the stream is
# open-ended -- so max_steps stays exactly horizon_steps, the same 300-tick
# default training used, so the model's implicit sense of "time remaining"
# (time_fraction, twap_target, remaining_time) isn't fed a distribution it
# never trained on.
#
# Both reset() and astep() accept an optional pre-fetched `tick`. When
# running a single policy live, omitting it lets the engine pull its own tick
# from order_book. When running several policies against the same moment in
# the market (the multi-policy paper-trading comparison), the caller fetches
# one tick from the shared LiveOrderBook and passes it to every engine's
# astep() for that round, so every policy sees identical market data and only
# diverges through its own accumulated state (inventory, perm_impact).

from typing import Optional

import websockets

from execution_core import ExecutionCore, ReplayState
from live_order_book import LiveOrderBook


class SmartOrderRouterLive(ExecutionCore):
    def __init__(self, order_book: LiveOrderBook, **core_kwargs):
        super().__init__(**core_kwargs)
        self.order_book = order_book
        self.max_steps = int(self.horizon_steps)

    async def reset(self, tick: Optional[ReplayState] = None):
        self.reset_episode_state()
        obs, _, _, _ = await self.astep(action=None, tick=tick)
        return obs

    async def astep(self, action: Optional[int] = None, tick: Optional[ReplayState] = None):
        if tick is None:
            try:
                tick = await self.order_book.next_tick()
            except websockets.ConnectionClosed:
                return self.terminal("stream_stalled")
        return self.process_tick(tick, action)
