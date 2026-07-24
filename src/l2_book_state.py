# l2_book_state.py
#
# Shared order-book maintenance primitives used by both the batch
# reconstruction pipeline (clean_l2.py) and the live streaming order book
# (live_order_book.py), so the two can never silently diverge in how they
# apply Binance diffDepth updates or decide what counts as "top of book."

from decimal import Decimal
from typing import Dict, List, Tuple


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


def top_of_book(
    bids: Dict[Decimal, Decimal], asks: Dict[Decimal, Decimal], depth_levels: int = 10
) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]], bool]:
    """Returns (top_bids, top_asks, crossed).

    top_bids/top_asks are empty if either side of the book hasn't been
    populated yet, or if the book is crossed (best_bid >= best_ask) -- a
    transient state from partial reconstruction noise (a level not yet
    touched on one side), which callers should skip rather than emit a
    nonsensical spread. `crossed` is only True in that second case, so
    callers can distinguish "not ready yet" from "saw a crossed book."
    """
    if not bids or not asks:
        return [], [], False

    best_bid_p = max(bids)
    best_ask_p = min(asks)
    if best_bid_p >= best_ask_p:
        return [], [], True

    top_bids = sorted(bids.items(), key=lambda kv: kv[0], reverse=True)[:depth_levels]
    top_asks = sorted(asks.items(), key=lambda kv: kv[0])[:depth_levels]
    return top_bids, top_asks, False
