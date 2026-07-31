# live_order_book.py
#
# Maintains a live, in-memory L2 order book from Binance's diffDepth
# websocket stream, using the exact same reconstruction algorithm
# (l2_book_state.apply_diff/top_of_book) as the batch pipeline (clean_l2.py)
# so live and historical reconstruction can never silently diverge.
#
# The book is seeded from a REST depth snapshot using Binance's documented
# sync procedure (buffer stream events -> fetch snapshot -> discard events
# already contained in it -> replay the rest). That replaces the old
# ~100-event warmup entirely: previously a fresh connection started from an
# empty book and could only learn levels that happened to be *touched* by a
# diff, so it took ~100s of wall clock before top-of-book was trustworthy and
# the deep book stayed permanently incomplete. A snapshot gives the whole
# ladder at tick zero, which matters a lot for large orders -- selling 100
# BTC genuinely walks ~400 levels.
#
# The stream endpoint (no "@100ms" suffix) pushes once per second, matching
# the tick cadence cleaned_l2/ and horizon_steps were built around.

import asyncio
import json
import urllib.request
from decimal import Decimal
from typing import Dict, List, Optional

import websockets

from download_binance_data import get_timestamp
from execution_core import ReplayState
from l2_book_state import apply_diff, top_of_book, DEPTH_MAX_LEVELS

DEPTH_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@depth"
SNAPSHOT_URL = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000"
SNAPSHOT_TIMEOUT_SEC = 20


def _fetch_snapshot() -> dict:
    """Blocking REST fetch; callers run it via asyncio.to_thread so a slow
    response can't stall the event loop serving dashboard clients."""
    with urllib.request.urlopen(SNAPSHOT_URL, timeout=SNAPSHOT_TIMEOUT_SEC) as r:
        return json.load(r)


class LiveOrderBook:
    """status: connecting -> syncing -> live, or disconnected on failure.

    A dropped connection can't resume mid-diff-sequence, but unlike the old
    warmup-based version it no longer needs to: reconnecting just re-runs the
    snapshot sync and is immediately trustworthy again. Callers still treat a
    ConnectionClosed from next_tick() as "this episode ends here" rather than
    splicing a fresh book into an in-flight episode, since the agent's own
    accumulated impact state wouldn't match the new book.
    """

    def __init__(self, depth_levels: int = DEPTH_MAX_LEVELS):
        self.depth_levels = depth_levels

        self.bids: Dict[Decimal, Decimal] = {}
        self.asks: Dict[Decimal, Decimal] = {}
        self.n_events = 0
        self.prev_u: Optional[int] = None
        self.gaps = 0
        self.last_update_id: Optional[int] = None

        self.status = "connecting"
        self._ws = None

    async def connect(self):
        self.status = "connecting"
        self._ws = await websockets.connect(DEPTH_STREAM)
        self.status = "syncing"
        await self._sync_from_snapshot()
        self.status = "live"

    async def _sync_from_snapshot(self):
        """Binance's documented local-book bootstrap: buffer stream events
        while the snapshot is in flight, then drop everything the snapshot
        already accounts for and replay the remainder on top of it."""
        snapshot_task = asyncio.create_task(asyncio.to_thread(_fetch_snapshot))
        buffered: List[dict] = []

        # Keep draining the socket while the REST call is in flight, so no
        # update between the snapshot's state and "now" is lost.
        while not snapshot_task.done():
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                snapshot_task.cancel()
                self.status = "disconnected"
                raise
            buffered.append(json.loads(message))

        snapshot = await snapshot_task
        self.last_update_id = snapshot["lastUpdateId"]

        self.bids = {Decimal(p): Decimal(q) for p, q in snapshot["bids"] if Decimal(q) != 0}
        self.asks = {Decimal(p): Decimal(q) for p, q in snapshot["asks"] if Decimal(q) != 0}

        # Discard buffered events already reflected in the snapshot; the first
        # kept event must straddle lastUpdateId+1 or the book has a hole.
        for update in buffered:
            u = update.get("u")
            if u is not None and u <= self.last_update_id:
                continue
            apply_diff(self.bids, self.asks, update)
            self.prev_u = u if u is not None else self.prev_u
            self.n_events += 1

    async def wait_until_live(self):
        """Kept for API compatibility. With snapshot bootstrapping the book is
        already live once connect() returns, so this is normally a no-op."""
        while self.status not in ("live", "disconnected"):
            await asyncio.sleep(0.05)

    async def next_tick(self) -> ReplayState:
        """Blocks until the next trustworthy tick is available -- silently
        skips crossed/incomplete-book ticks. Raises
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
            self.prev_u = u if u is not None else self.prev_u

            apply_diff(self.bids, self.asks, update)
            self.n_events += 1

            top_bids, top_asks, _crossed = top_of_book(self.bids, self.asks, self.depth_levels)
            if not top_bids or not top_asks:
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
