# clean_l2.py
#
# Reconstructs a multi-level (L2) order book from Binance diffDepth events and
# writes one JSONL record per event to cleaned_l2/{date}.jsonl. Each raw file
# under data/{date}/depth/ holds a chunk of the day's diffDepth stream; U/u
# update-ID continuity across files (checked below) confirms these are
# genuine sequential diffs, so the standard incremental book-maintenance
# algorithm applies: qty == "0" means the price level was removed, otherwise
# it's an upsert. There is no initial REST snapshot, so the book starts empty
# and is only as complete as the levels touched so far -- top-of-book settles
# within the first handful of events since it changes on nearly every tick,
# but a short warmup is skipped to let a few levels of depth populate too.

import json
import sys
from pathlib import Path

from l2_book_state import apply_diff, top_of_book, DEPTH_MAX_LEVELS

RAW_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "cleaned_l2"

# See l2_book_state for why this is no longer 10 levels: the old truncation
# priced ~96% of a large fill with a fabricated fallback rather than real
# depth. The filtering policy lives there so live and batch stay identical.
DEPTH_LEVELS = DEPTH_MAX_LEVELS
WARMUP_EVENTS = 100


def _fmt(d) -> str:
    """Trim trailing zeros ("64355.99000000" -> "64355.99"). Deep books make
    these files ~20x larger, and Binance pads everything to 8 decimals;
    Decimal() parses the trimmed form back identically."""
    s = format(d, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def reconstruct_day(date_str: str, depth_levels: int = DEPTH_LEVELS, warmup: int = WARMUP_EVENTS) -> int:
    depth_dir = RAW_DIR / date_str / "depth"
    files = sorted(depth_dir.glob("depth_*.json"))
    if not files:
        raise FileNotFoundError(f"No depth files found for {date_str} in {depth_dir}")

    bids: dict = {}
    asks: dict = {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{date_str}.jsonl"

    n_events = 0
    n_written = 0
    n_crossed = 0
    prev_u = None
    gaps = 0

    with out_path.open("w", encoding="utf-8") as out_fh:
        for fp in files:
            with fp.open("r", encoding="utf-8") as fh:
                events = json.load(fh)

            for ev in events:
                update = ev["update"]
                U, u = update.get("U"), update.get("u")
                if prev_u is not None and U is not None and U != prev_u + 1:
                    gaps += 1
                prev_u = u if u is not None else prev_u

                apply_diff(bids, asks, update)

                n_events += 1
                if n_events <= warmup:
                    continue

                top_bids, top_asks, crossed = top_of_book(bids, asks, depth_levels)
                if not top_bids or not top_asks:
                    if crossed:
                        n_crossed += 1
                    continue

                best_bid_p, best_ask_p = top_bids[0][0], top_asks[0][0]
                midprice = (best_bid_p + best_ask_p) / 2

                record = {
                    "timestamp": ev["timestamp"],
                    "bids": [[_fmt(p), _fmt(q)] for p, q in top_bids],
                    "asks": [[_fmt(p), _fmt(q)] for p, q in top_asks],
                    "best_bid": [_fmt(top_bids[0][0]), _fmt(top_bids[0][1])],
                    "best_ask": [_fmt(top_asks[0][0]), _fmt(top_asks[0][1])],
                    "midprice": _fmt(midprice),
                }
                out_fh.write(json.dumps(record) + "\n")
                n_written += 1

    print(
        f"{date_str}: events={n_events} written={n_written} "
        f"skipped_crossed={n_crossed} update_id_gaps={gaps} -> {out_path}"
    )
    return n_written


def main():
    dates = sys.argv[1:] or ["2026-07-11", "2026-07-12"]
    for date_str in dates:
        reconstruct_day(date_str)


if __name__ == "__main__":
    main()
