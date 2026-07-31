# l2_book_state.py
#
# Shared order-book maintenance primitives used by both the batch
# reconstruction pipeline (clean_l2.py) and the live streaming order book
# (live_order_book.py), so the two can never silently diverge in how they
# apply Binance diffDepth updates or decide what depth to keep.

from decimal import Decimal
from typing import Dict, List, Tuple

# Depth-capture policy. The original pipeline kept only 10 levels, which
# turned out to overstate large-order slippage by ~10x: selling 100 BTC
# genuinely walks ~400 levels, so 96% of that fill was being priced by a
# fabricated "worst visible level * 0.995" fallback rather than real depth
# (measured against a live REST snapshot: real 4.7 bps vs simulated 47 bps).
#
# Keeping every level is too expensive (5000 levels/side is ~1.2ms just to
# parse per tick, and most of the tail is dust), so:
#   * the first EXACT_LEVELS are kept verbatim -- near-touch fidelity is what
#     small fills and the depth features actually depend on, and the true
#     best bid/ask must be preserved even when it's a tiny order, since it
#     defines the spread and midprice;
#   * beyond that, levels below MIN_QTY are dropped. On a real BTCUSDT book
#     this removes ~75% of levels while retaining ~99% of the liquidity;
#   * everything is capped to a price band around the touch, since an order
#     that never reaches 100 bps deep can't be affected by what's past it.
# min_qty 0.05 rather than 0.01: measured against a live snapshot it halves
# the level count (653 -> 335) while still retaining ~98% of the liquidity,
# which roughly halves both file size and per-tick parse cost. Those matter
# here because replay_engine loads whole days into memory for training.
DEPTH_EXACT_LEVELS = 20
DEPTH_MIN_QTY = Decimal("0.05")
DEPTH_BAND_BPS = Decimal("100")
DEPTH_MAX_LEVELS = 400


def apply_diff(bids: Dict[Decimal, Decimal], asks: Dict[Decimal, Decimal], update: dict) -> None:
    """Apply one Binance diffDepth update in place: qty == 0 removes a price
    level, otherwise it's an upsert. Standard incremental book-maintenance."""
    for price_str, qty_str in update.get("b", []):
        price = Decimal(price_str)
        qty = Decimal(qty_str)
        if qty == 0:
            bids.pop(price, None)
        else:
            bids[price] = qty

    for price_str, qty_str in update.get("a", []):
        price = Decimal(price_str)
        qty = Decimal(qty_str)
        if qty == 0:
            asks.pop(price, None)
        else:
            asks[price] = qty


def _slice_side(
    levels: List[Tuple[Decimal, Decimal]],
    best: Decimal,
    is_bid: bool,
    min_qty: Decimal,
    band_bps: Decimal,
    exact_levels: int,
    max_levels: int,
) -> List[Tuple[Decimal, Decimal]]:
    band = best * (band_bps / Decimal("10000"))
    limit = best - band if is_bid else best + band

    out: List[Tuple[Decimal, Decimal]] = []
    for i, (price, qty) in enumerate(levels):
        if i >= exact_levels:
            if qty < min_qty:
                continue
            if (is_bid and price < limit) or (not is_bid and price > limit):
                break
        out.append((price, qty))
        if len(out) >= max_levels:
            break
    return out


def top_of_book(
    bids: Dict[Decimal, Decimal],
    asks: Dict[Decimal, Decimal],
    depth_levels: int = DEPTH_MAX_LEVELS,
    min_qty: Decimal = DEPTH_MIN_QTY,
    band_bps: Decimal = DEPTH_BAND_BPS,
    exact_levels: int = DEPTH_EXACT_LEVELS,
) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]], bool]:
    """Returns (top_bids, top_asks, crossed).

    top_bids/top_asks are empty if either side of the book hasn't been
    populated yet, or if the book is crossed (best_bid >= best_ask) -- a
    transient state from partial reconstruction noise (a level not yet
    touched on one side), which callers should skip rather than emit a
    nonsensical spread. `crossed` is only True in that second case, so
    callers can distinguish "not ready yet" from "saw a crossed book."

    Depth beyond the first `exact_levels` is filtered per the module-level
    policy above; pass depth_levels to cap the total returned.
    """
    if not bids or not asks:
        return [], [], False

    best_bid_p = max(bids)
    best_ask_p = min(asks)
    if best_bid_p >= best_ask_p:
        return [], [], True

    sorted_bids = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)
    sorted_asks = sorted(asks.items(), key=lambda kv: kv[0])

    top_bids = _slice_side(
        sorted_bids, best_bid_p, True, min_qty, band_bps, exact_levels, depth_levels
    )
    top_asks = _slice_side(
        sorted_asks, best_ask_p, False, min_qty, band_bps, exact_levels, depth_levels
    )
    return top_bids, top_asks, False
