# replay_engine.py

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from execution_core import ExecutionCore, ReplayState


class SmartOrderRouterReplay(ExecutionCore):
    """Replays a reconstructed order book (a historical file) and simulates
    liquidating a position, using the shared reward/impact/feature logic in
    ExecutionCore. Owns everything specific to "the tick source is a
    pre-loaded historical file": which day(s), which random window, and the
    legacy L1/L2 file-format fallback.

    Falls back to the legacy single-level (L1) cleaned/{date}.jsonl format if
    no cleaned_l2/{date}.jsonl file exists, in which case the "book" is just
    the best bid/ask and walking it degenerates to a single-level fill.

    total_target_qty_range / horizon_steps_range (both default None, i.e. no
    randomization -- backward compatible with every existing eval/live call
    site) let training draw a fresh target size and/or window length each
    episode instead of always the fixed default. Without this, a model only
    ever experiences one (qty, horizon) point and doesn't generalize to any
    other -- confirmed the hard way: a model trained exclusively at 25 BTC /
    300 ticks lost badly to a rule-based baseline the first time it was asked
    to liquidate 100 BTC (real absolute trade size scales with
    total_target_qty, but real order-book depth doesn't, so the same action
    that was safe at 25 BTC walks much deeper into the same depth at 100).
    """

    def __init__(
        self,
        date_str,
        total_target_qty: float = 25.0,
        total_target_qty_range: Optional[Tuple[float, float]] = None,
        min_trade_size: float = 0.001,
        horizon_steps: int = 300,
        horizon_steps_range: Optional[Tuple[int, int]] = None,
        randomize_start: bool = True,
        perm_impact_coefficient: float = 0.02,
        perm_impact_decay: float = 0.97,
        schedule_penalty_factor: float = 0.1,
        leftover_penalty_factor: float = 0.5,
    ):
        super().__init__(
            total_target_qty=total_target_qty,
            min_trade_size=min_trade_size,
            horizon_steps=horizon_steps,
            perm_impact_coefficient=perm_impact_coefficient,
            perm_impact_decay=perm_impact_decay,
            schedule_penalty_factor=schedule_penalty_factor,
            leftover_penalty_factor=leftover_penalty_factor,
        )
        self.total_target_qty_range = total_target_qty_range
        self.horizon_steps_range = horizon_steps_range

        # date_str may be a single "YYYY-MM-DD" or a list of them. Each
        # episode (see reset()) picks one date at random and a random start
        # window within it, so training can draw from several days instead
        # of overfitting to one day's idiosyncrasies.
        self.dates: List[str] = [date_str] if isinstance(date_str, str) else list(date_str)
        if not self.dates:
            raise ValueError("At least one date_str must be provided")
        self.date_str = ",".join(self.dates)

        project_root = Path(__file__).resolve().parent.parent
        self.day_lines: Dict[str, List[str]] = {}
        day_is_l2: Dict[str, bool] = {}
        for d in self.dates:
            l2_path = project_root / "cleaned_l2" / f"{d}.jsonl"
            l1_path = project_root / "cleaned" / f"{d}.jsonl"
            if l2_path.exists():
                path, is_l2 = l2_path, True
            elif l1_path.exists():
                path, is_l2 = l1_path, False
            else:
                raise FileNotFoundError(
                    f"No cleaned data found for {d} (looked in {l2_path} and {l1_path})"
                )
            # Files are a few MB at most, so load once and index in memory.
            # This lets us pick a random start window per episode without
            # reparsing.
            with path.open("r", encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                raise ValueError(f"Cleaned file is empty: {path}")
            self.day_lines[d] = lines
            day_is_l2[d] = is_l2

        self.is_l2 = all(day_is_l2.values())

        # Placeholders; reset() picks the day and populates these per episode.
        self.current_date: Optional[str] = None
        self.current_lines: List[str] = self.day_lines[self.dates[0]]
        self.num_steps = len(self.current_lines)

        self.randomize_start = bool(randomize_start)
        # An episode runs for at most horizon_steps ticks, and never past EOF.
        self.max_steps = max(min(self.horizon_steps, self.num_steps), 1)

        self.start_index = 0

    @staticmethod
    def _parse_record(data: Dict[str, Any]):
        if "bids" in data and "asks" in data:
            return data["timestamp"], data["bids"], data["asks"], data.get("midprice")
        # Legacy L1 fallback: treat best_bid/best_ask as a single-level book.
        best_bid = data.get("best_bid")
        best_ask = data.get("best_ask")
        bids = [best_bid] if best_bid else []
        asks = [best_ask] if best_ask else []
        return data["timestamp"], bids, asks, data.get("midprice")

    def reset(self):
        # Resample target size / horizon *before* reset_episode_state(), so
        # inventory is initialized against the freshly-sampled quantity in
        # one pass rather than needing a second correction afterward.
        if self.total_target_qty_range is not None:
            lo, hi = self.total_target_qty_range
            sampled_qty = self._rng.uniform(lo, hi)
            self.total_target_qty = Decimal(str(round(sampled_qty, 3)))
        if self.horizon_steps_range is not None:
            lo_h, hi_h = self.horizon_steps_range
            self.horizon_steps = self._rng.randint(lo_h, hi_h)

        self.reset_episode_state()

        # Pick the day (if multiple) and the execution window's starting tick.
        if self.randomize_start and len(self.dates) > 1:
            self.current_date = self._rng.choice(self.dates)
        else:
            self.current_date = self.dates[0]
        self.current_lines = self.day_lines[self.current_date]
        self.num_steps = len(self.current_lines)
        self.max_steps = max(min(self.horizon_steps, self.num_steps), 1)

        max_start = max(self.num_steps - self.max_steps, 0)
        if self.randomize_start and max_start > 0:
            self.start_index = self._rng.randint(0, max_start)
        else:
            self.start_index = 0

        obs, _, _, _ = self.step(action=None)
        return obs

    def step(self, action: Optional[int] = None):
        line_cursor = self.start_index + self.step_index
        if line_cursor >= self.num_steps:
            # Ran off the end of the data before the window closed.
            return self.terminal("end_of_data")

        data = json.loads(self.current_lines[line_cursor])
        timestamp, bids, asks, midprice = self._parse_record(data)
        tick = ReplayState(timestamp=timestamp, bids=bids, asks=asks, midprice=midprice)
        return self.process_tick(tick, action)


def main():
    engine = SmartOrderRouterReplay("2026-07-11", randomize_start=False)
    obs = engine.reset()
    print("First obs keys:", sorted(obs.keys()))
    print("Using L2 depth:", engine.is_l2)

    actions = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]
    for a in actions:
        obs, reward, done, info = engine.step(a)
        print(
            f"Action: {a} | Reward(bps): {reward:.4f} | "
            f"filled: {info.get('filled')} | qty: {info.get('trade_qty')} | "
            f"slip: {obs['slippage']:.6e} | perm_impact: {obs['perm_impact']:.4f} | "
            f"bid_depth_5: {obs['bid_depth_5']:.4f} | impact_small/large(bps): "
            f"{obs['impact_bps_small']:.3f}/{obs['impact_bps_large']:.3f} | "
            f"time: {obs['time_fraction']:.3f} | rem_inv: {obs['remaining_inventory']:.6f}"
        )
        if done:
            break

    engine.close()


if __name__ == "__main__":
    main()
