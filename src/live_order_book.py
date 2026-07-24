# live_order_book.py
#
# Maintains a live, in-memory L2 order book from Binance's diffDepth
# websocket stream, using the exact same reconstruction algorithm
# (l2_book_state.apply_diff/top_of_book) as the batch pipeline (clean_l2.py)
# so live and historical reconstruction can never silently diverge.
#
# There is no REST snapshot (same reasoning as the batch pipeline): a fresh
# connection's local book starts empty and is only as complete as the levels
# touched so far, so every (re)connect needs the same WARMUP_EVENTS skip
# before the top-of-book can be trusted -- carried over unchanged from
# clean_l2.py's own warmup constant. This endpoint (no "@100ms" suffix)
# pushes once per second, matching the tick cadence cleaned_l2/ and
# horizon_steps=300 were already built around.

import json
from decimal import Decimal
from typing import Dict, Optional

import websockets

from download_binance_data import get_timestamp
from execution_core import ReplayState
from l2_book_state import apply_diff, top_of_book

DEPTH_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@depth"
DEPTH_LEVELS = 10
WARMUP_EVENTS = 100


class LiveOrderBook:
    """status: connecting -> warming_up -> live, or disconnected on failure.

    A dropped connection has no way to resume mid-diff-sequence (no snapshot
    exists), so reconnecting always re-warms from an empty book rather than
    trying to splice state back together -- callers should treat a
    ConnectionClosed from next_tick() as "this session's episode ends here",
    not "retry the same tick."
    """

    def __init__(self, depth_levels: int = DEPTH_LEVELS, warmup_events: int = WARMUP_EVENTS):
        self.depth_levels = depth_levels
        self.warmup_events = warmup_events

        self.bids: Dict[Decimal, Decimal] = {}
        self.asks: Dict[Decimal, Decimal] = {}
        self.n_events = 0
        self.prev_u: Optional[int] = None
        self.gaps = 0

        self.status = "connecting"
        self._ws = None

    async def connect(self):
        self.status = "connecting"
        self._ws = await websockets.connect(DEPTH_STREAM)
        self.status = "warming_up"

    async def wait_until_live(self):
        """Consume ticks (discarding them) until the book clears warmup and
        produces a real top-of-book, so callers never prime an episode's
        arrival price from an untrustworthy book."""
        while self.status != "live":
            await self.next_tick()

    async def next_tick(self) -> ReplayState:
        """Blocks until the next trustworthy tick is available -- silently
        skips warmup ticks and crossed/incomplete-book ticks. Raises
        websockets.ConnectionClosed if the stream drops."""
        if self._ws is None:
            raise RuntimeError("LiveOrderBook.connect() must be awaited before next_tick()")

        while True:
            try:
                message = await self._ws.recv()
            except websockets.ConnectionClosed:
                self.status = "disconnected"
                raise

            update = json.loads(message)
            U, u = update.get("U"), update.get("u")
            if self.prev_u is not None and U is not None and U != self.prev_u + 1:
                self.gaps += 1
            prev_u = u if u is not None else self.prev_u
            self.prev_u = prev_u

            apply_diff(self.bids, self.asks, update)
            self.n_events += 1

            if self.n_events <= self.warmup_events:
                continue

            top_bids, top_asks, _crossed = top_of_book(self.bids, self.asks, self.depth_levels)
            if not top_bids or not top_asks:
                # Crossed, or (rarely, past warmup) still one-sided -- keep waiting.
                continue

            self.status = "live"
            midprice = (top_bids[0][0] + top_asks[0][0]) / 2
            return ReplayState(
                timestamp=get_timestamp(),
                bids=[[str(p), str(q)] for p, q in top_bids],
                asks=[[str(p), str(q)] for p, q in top_asks],
                midprice=str(midprice),
            )

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
        self.status = "disconnected"
