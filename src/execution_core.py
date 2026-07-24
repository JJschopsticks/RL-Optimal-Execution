# execution_core.py
#
# Shared reward/impact/feature logic for liquidating a position against a
# reconstructed L2 order book. This is deliberately data-source-agnostic:
# ExecutionCore doesn't know or care whether the next tick comes from a
# pre-loaded historical file (SmartOrderRouterReplay, in replay_engine.py) or
# a live websocket-fed order book (SmartOrderRouterLive, in live_engine.py) --
# both drive the exact same process_tick()/_execute_action()/_compute_reward()
# code, so backtest and live can never silently diverge in their math.

import math
import random
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


class ReplayState:
    """A single reconstructed order-book tick.

    Accepts multi-level (L2) bids/asks when available (list of [price, qty]
    string pairs, sorted best-first) and falls back to a single-level book
    when only best_bid/best_ask are present (legacy L1 cleaned files).
    """

    def __init__(self, timestamp: str, bids: List[List[str]], asks: List[List[str]],
                 midprice: Optional[str]):
        self.timestamp = timestamp
        self.bids: List[Tuple[Decimal, Decimal]] = [(Decimal(p), Decimal(q)) for p, q in bids]
        self.asks: List[Tuple[Decimal, Decimal]] = [(Decimal(p), Decimal(q)) for p, q in asks]
        self.midprice = Decimal(midprice) if midprice else None

        self.best_bid_price = self.bids[0][0] if self.bids else None
        self.best_bid_qty = self.bids[0][1] if self.bids else None
        self.best_ask_price = self.asks[0][0] if self.asks else None
        self.best_ask_qty = self.asks[0][1] if self.asks else None

    def to_features(self) -> Dict[str, Any]:
        spread = None
        if self.best_bid_price is not None and self.best_ask_price is not None:
            spread = self.best_ask_price - self.best_bid_price

        return {
            "timestamp": self.timestamp,
            "best_bid_price": float(self.best_bid_price) if self.best_bid_price is not None else None,
            "best_bid_qty": float(self.best_bid_qty) if self.best_bid_qty is not None else None,
            "best_ask_price": float(self.best_ask_price) if self.best_ask_price is not None else None,
            "best_ask_qty": float(self.best_ask_qty) if self.best_ask_qty is not None else None,
            "midprice": float(self.midprice) if self.midprice is not None else None,
            "spread": float(spread) if spread is not None else None,
        }


class ExecutionCore:
    """Reward design (all in basis points of arrival notional so PPO sees O(1-1000)
    values instead of raw dollars):

      * Per fill: implementation shortfall vs the arrival midprice.
        Selling below arrival = a cost (negative reward); selling above = a gain.
      * A small per-step schedule penalty (a *rate*, so its sum over the episode
        stays bounded) nudges the agent to keep pace with a TWAP line.
      * A terminal leftover penalty models a punitive forced liquidation of any
        inventory left when the execution window closes.

    Market reaction to the agent's own sells is modelled two ways:

      * Temporary impact: each fill walks the real reconstructed bid ladder,
        so cost is a genuine volume-weighted average price against displayed
        depth rather than a parametric formula -- smaller child orders are
        cheaper because they don't need to walk as deep.
      * Permanent impact: each sell shifts every level of the ladder down by a
        decaying amount (book resilience) -> spreading a big sell over time is
        cheaper than dumping it in one shot.

    Subclasses own the tick source (a historical file for replay, a live
    order book for paper trading) and call process_tick()/terminal() with a
    ReplayState they've obtained however is appropriate for that source.
    """

    def __init__(
        self,
        total_target_qty: float = 25.0,
        min_trade_size: float = 0.001,
        horizon_steps: int = 300,
        perm_impact_coefficient: float = 0.02,
        perm_impact_decay: float = 0.97,
        schedule_penalty_factor: float = 0.1,
        leftover_penalty_factor: float = 0.5,
    ):
        self.current_state: Optional[ReplayState] = None

        self.total_target_qty = Decimal(str(total_target_qty))
        self.min_trade_size = Decimal(str(min_trade_size))
        # Fractions of the total target sold per fire. Small enough that
        # liquidating the position takes many child orders (pacing matters).
        self.action_fractions = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

        # Execution window (in ticks). The whole liquidation happens inside this
        # short burst, not over the entire day. Subclasses that know a hard
        # upper bound on available ticks (a finite file) may clamp max_steps
        # further; a live, open-ended stream leaves it as horizon_steps.
        self.horizon_steps = int(horizon_steps)
        self.max_steps = int(horizon_steps)

        self.perm_impact_coefficient = float(perm_impact_coefficient)
        self.perm_impact_decay = float(perm_impact_decay)
        self.schedule_penalty_factor = float(schedule_penalty_factor)
        self.leftover_penalty_factor = float(leftover_penalty_factor)

        self._rng = random.Random()

        self.cash = Decimal("0")
        self.inventory = Decimal(str(total_target_qty))
        self.remaining_inventory = Decimal(str(total_target_qty))

        self.midprice_history: List[float] = []
        self.return_history: List[float] = []

        self.prev_bid_price: Optional[Decimal] = None
        self.prev_ask_price: Optional[Decimal] = None
        self.prev_bid_qty: Optional[Decimal] = None
        self.prev_ask_qty: Optional[Decimal] = None
        self.prev_midprice: Optional[Decimal] = None

        self.step_index = 0
        self.cum_traded_qty = Decimal("0")

        # Arrival benchmark and the agent's own accumulated (decaying) impact.
        self.arrival_mid: Optional[float] = None
        self.perm_impact = Decimal("0")

    def seed(self, seed: Optional[int] = None):
        self._rng.seed(seed)

    def portfolio_value(self) -> Decimal:
        if self.current_state and self.current_state.midprice is not None:
            return self.cash + self.inventory * self.current_state.midprice
        return self.cash

    def _compute_midprice_returns(self, mid: float):
        self.midprice_history.append(mid)

        if len(self.midprice_history) < 2:
            r1 = r5 = r10 = 0.0
        else:
            prev_1 = self.midprice_history[-2]
            r1 = (mid - prev_1) / prev_1 if prev_1 != 0 else 0.0

            if len(self.midprice_history) >= 6:
                prev_5 = self.midprice_history[-6]
                r5 = (mid - prev_5) / prev_5 if prev_5 != 0 else 0.0
            else:
                r5 = 0.0

            if len(self.midprice_history) >= 11:
                prev_10 = self.midprice_history[-11]
                r10 = (mid - prev_10) / prev_10 if prev_10 != 0 else 0.0
            else:
                r10 = 0.0

        self.return_history.append(r1)
        return float(r1), float(r5), float(r10)

    def _compute_volatility(self):
        if len(self.return_history) >= 5:
            window5 = self.return_history[-5:]
            mean5 = sum(window5) / 5
            var5 = sum((x - mean5) ** 2 for x in window5) / 5
            vol_5 = math.sqrt(var5)
        else:
            vol_5 = 0.0

        if len(self.return_history) >= 10:
            window10 = self.return_history[-10:]
            mean10 = sum(window10) / 10
            var10 = sum((x - mean10) ** 2 for x in window10) / 10
            vol_10 = math.sqrt(var10)
        else:
            vol_10 = 0.0

        return float(vol_5), float(vol_10)

    def _compute_ofi(self):
        if self.current_state is None:
            return 0.0, 0.0, 0.0

        bid_p = self.current_state.best_bid_price
        ask_p = self.current_state.best_ask_price
        bid_q = self.current_state.best_bid_qty
        ask_q = self.current_state.best_ask_qty

        if self.prev_bid_price is None or self.prev_ask_price is None:
            self.prev_bid_price = bid_p
            self.prev_ask_price = ask_p
            self.prev_bid_qty = bid_q
            self.prev_ask_qty = ask_q
            return 0.0, 0.0, 0.0

        dp_bid = float(bid_p - self.prev_bid_price) if bid_p and self.prev_bid_price else 0.0
        dp_ask = float(ask_p - self.prev_ask_price) if ask_p and self.prev_ask_price else 0.0
        ofi_price = dp_bid - dp_ask

        dq_bid = float(bid_q - self.prev_bid_qty) if bid_q and self.prev_bid_qty else 0.0
        dq_ask = float(ask_q - self.prev_ask_qty) if ask_q and self.prev_ask_qty else 0.0
        ofi_qty = dq_bid - dq_ask

        ofi_combined = ofi_price + ofi_qty

        self.prev_bid_price = bid_p
        self.prev_ask_price = ask_p
        self.prev_bid_qty = bid_q
        self.prev_ask_qty = ask_q

        return float(ofi_price), float(ofi_qty), float(ofi_combined)

    def _compute_depth_imbalance(self) -> float:
        if self.current_state is None:
            return 0.0
        bid_q = self.current_state.best_bid_qty
        ask_q = self.current_state.best_ask_qty
        if bid_q is None or ask_q is None:
            return 0.0
        total = float(bid_q + ask_q)
        if total == 0.0:
            return 0.0
        return (float(bid_q) - float(ask_q)) / total

    def _compute_microprice_and_queue(self):
        if self.current_state is None:
            return 0.0, 0.0, 0.0, 0.0

        bid_p = self.current_state.best_bid_price
        ask_p = self.current_state.best_ask_price
        bid_q = self.current_state.best_bid_qty
        ask_q = self.current_state.best_ask_qty
        mid = self.current_state.midprice

        if bid_p is None or ask_p is None or bid_q is None or ask_q is None:
            return 0.0, 0.0, 0.0, 0.0

        total_q = float(bid_q + ask_q)
        if total_q == 0.0:
            micro = float(mid) if mid is not None else 0.0
            return micro, 0.0, 0.0, 0.0

        micro = (float(bid_p) * float(ask_q) + float(ask_p) * float(bid_q)) / total_q
        qi = float(bid_q) / total_q
        spread_ratio = 0.0
        if mid is not None and mid != 0:
            spread_ratio = float(ask_p - bid_p) / float(mid)

        return float(micro), float(qi), total_q, float(spread_ratio)

    def _compute_market_impact(self, mid: Optional[Decimal]) -> float:
        if mid is None or self.prev_midprice is None:
            impact = 0.0
        else:
            impact = float(mid - self.prev_midprice)
        self.prev_midprice = mid
        return impact

    def _compute_twap_vwap(self):
        time_frac = min(self.step_index / self.max_steps, 1.0)
        twap_target = float(self.total_target_qty) * time_frac
        twap_diff = float(self.cum_traded_qty - self.total_target_qty * Decimal(str(time_frac)))
        vwap_price = sum(self.midprice_history) / len(self.midprice_history) if self.midprice_history else 0.0
        return float(time_frac), float(twap_target), float(twap_diff), float(vwap_price)

    def _compute_book_depth(self):
        """Multi-level depth/imbalance over the top 5 levels each side."""
        if self.current_state is None:
            return 0.0, 0.0, 0.0
        bid_depth = float(sum(q for _, q in self.current_state.bids[:5]))
        ask_depth = float(sum(q for _, q in self.current_state.asks[:5]))
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0.0 else 0.0
        return bid_depth, ask_depth, imbalance

    def _walk_book_sell(
        self, levels: List[Tuple[Decimal, Decimal]], qty: Decimal, perm_impact: Decimal
    ) -> Tuple[Optional[Decimal], Decimal, bool]:
        """Volume-weighted average fill price for selling `qty` against bid
        `levels` (sorted best-first), each shifted down by `perm_impact`.

        Returns (avg_price, filled_qty, exhausted_depth). If the order is
        larger than the visible depth, the remainder fills at the worst
        visible level with an extra cushion (real depth beyond what we
        captured would be thinner still).
        """
        remaining = qty
        total_cost = Decimal("0")
        filled = Decimal("0")
        last_eff_price: Optional[Decimal] = None

        for price, level_qty in levels:
            eff_price = price - perm_impact
            if eff_price < 0:
                eff_price = Decimal("0")
            last_eff_price = eff_price

            take = min(remaining, level_qty)
            if take <= 0:
                continue
            total_cost += take * eff_price
            filled += take
            remaining -= take
            if remaining <= 0:
                break

        exhausted = remaining > 0
        if exhausted and last_eff_price is not None:
            extra_price = max(Decimal("0"), last_eff_price * Decimal("0.995"))
            total_cost += remaining * extra_price
            filled += remaining
            remaining = Decimal("0")

        if filled <= 0:
            return None, Decimal("0"), exhausted

        return total_cost / filled, filled, exhausted

    def _estimate_impact_bps(self, fraction: float) -> float:
        """Hypothetical cost (bps vs top-of-book) of executing `fraction` of
        the target right now, without actually trading. Lets the agent see
        how expensive firing would be before deciding to fire."""
        if self.current_state is None or not self.current_state.bids:
            return 0.0
        qty = max(self.min_trade_size, self.total_target_qty * Decimal(str(fraction)))
        avg_price, filled_qty, _ = self._walk_book_sell(self.current_state.bids, qty, self.perm_impact)
        best_bid = self.current_state.best_bid_price
        if avg_price is None or filled_qty <= 0 or not best_bid:
            return 0.0
        cost = (best_bid - avg_price) / best_bid
        return float(cost) * 1e4

    def _arrival_notional(self) -> float:
        """Notional used to express reward in basis points."""
        ref = self.arrival_mid
        if ref is None and self.current_state and self.current_state.midprice is not None:
            ref = float(self.current_state.midprice)
        return max(float(self.total_target_qty) * (ref or 0.0), 1e-9)

    def reset_episode_state(self):
        """Resets portfolio/episode state. Subclasses call this from their own
        reset(), then handle picking a tick source (a file window, a live
        book) and driving the priming tick themselves."""
        self.current_state = None
        self.cash = Decimal("0")
        self.inventory = Decimal(str(self.total_target_qty))
        self.remaining_inventory = Decimal(str(self.total_target_qty))
        self.midprice_history = []
        self.return_history = []

        self.prev_bid_price = None
        self.prev_ask_price = None
        self.prev_bid_qty = None
        self.prev_ask_qty = None
        self.prev_midprice = None

        self.step_index = 0
        self.cum_traded_qty = Decimal("0")
        self.arrival_mid = None
        self.perm_impact = Decimal("0")

    def _execute_action(self, action: Optional[int]) -> Dict[str, Any]:
        info = {
            "filled": False,
            "fill_price": None,
            "slippage": 0.0,
            "trade_qty": 0.0,
            "impact_cost": 0.0,
            "exhausted_depth": False,
        }

        if action is None or self.current_state is None:
            return info

        try:
            action = int(action)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid action {action} for sell execution")

        if action < 0 or action >= len(self.action_fractions):
            raise ValueError(f"Invalid action {action} for sell execution")

        fraction = self.action_fractions[action]
        if fraction == 0.0 or self.inventory <= 0:
            return info

        if not self.current_state.bids:
            return info

        target_qty = self.total_target_qty * Decimal(str(fraction))
        trade_qty = min(self.inventory, max(self.min_trade_size, target_qty))
        if trade_qty <= 0:
            return info

        avg_price, filled_qty, exhausted = self._walk_book_sell(
            self.current_state.bids, trade_qty, self.perm_impact
        )
        if avg_price is None or filled_qty <= 0:
            return info

        best_bid_price = self.current_state.best_bid_price
        proceeds = avg_price * filled_qty

        self.cash += proceeds
        self.inventory -= filled_qty
        self.remaining_inventory = self.inventory
        self.cum_traded_qty += filled_qty

        # This sell adds to permanent impact, which decays each step (see process_tick()).
        added_impact = self.perm_impact_coefficient * float(best_bid_price) * (
            float(filled_qty) / float(self.total_target_qty)
        )
        self.perm_impact += Decimal(str(added_impact))

        mid = float(self.current_state.midprice) if self.current_state.midprice is not None else None
        fill_price_f = float(avg_price)
        slippage = mid - fill_price_f if mid is not None else 0.0
        impact_cost = float(best_bid_price) - fill_price_f

        info["filled"] = True
        info["fill_price"] = fill_price_f
        info["trade_qty"] = float(filled_qty)
        info["impact_cost"] = impact_cost
        info["slippage"] = slippage
        info["exhausted_depth"] = exhausted
        return info

    def _terminal_obs(self) -> Dict[str, Any]:
        obs = self.current_state.to_features() if self.current_state else {}
        obs["cash"] = float(self.cash)
        obs["inventory"] = float(self.inventory)
        obs["portfolio_value"] = float(self.portfolio_value())
        obs["return_1"] = 0.0
        obs["return_5"] = 0.0
        obs["return_10"] = 0.0
        obs["vol_5"] = 0.0
        obs["vol_10"] = 0.0
        obs["ofi_price"] = 0.0
        obs["ofi_qty"] = 0.0
        obs["ofi_combined"] = 0.0
        obs["depth_imbalance"] = 0.0
        obs["microprice"] = 0.0
        obs["queue_imbalance"] = 0.0
        obs["total_depth"] = 0.0
        obs["spread_ratio"] = 0.0
        obs["time_fraction"] = 1.0
        obs["twap_target"] = float(self.total_target_qty)
        obs["twap_diff"] = float(self.cum_traded_qty - self.total_target_qty)
        obs["vwap_price"] = self._compute_twap_vwap()[3]
        obs["remaining_inventory"] = float(self.inventory)
        obs["remaining_inventory_ratio"] = float(self.inventory / self.total_target_qty)
        obs["remaining_time"] = 0.0
        obs["execution_cost"] = 0.0
        obs["market_impact"] = 0.0
        obs["slippage"] = 0.0
        obs["perm_impact"] = float(self.perm_impact)
        obs["bid_depth_5"] = 0.0
        obs["ask_depth_5"] = 0.0
        obs["book_imbalance_5"] = 0.0
        obs["impact_bps_small"] = 0.0
        obs["impact_bps_large"] = 0.0
        return obs

    def terminal(self, reason: str):
        """A non-horizon episode end (e.g. replay ran off the end of its file,
        or a live stream stalled). Applies the same leftover penalty as
        reaching the horizon with inventory still held."""
        obs = self._terminal_obs()
        reward = self._leftover_penalty()
        return obs, reward, True, {reason: True}

    def process_tick(self, tick: ReplayState, action: Optional[int] = None):
        """Advance one tick given an already-constructed ReplayState, whatever
        its source. This is the shared core: identical for replay and live."""
        # Permanent impact recovers a little every tick (book resilience).
        self.perm_impact *= Decimal(str(self.perm_impact_decay))

        self.current_state = tick

        if self.arrival_mid is None and self.current_state.midprice is not None:
            self.arrival_mid = float(self.current_state.midprice)

        self.step_index += 1
        info = self._execute_action(action)

        obs = self.current_state.to_features()
        obs["cash"] = float(self.cash)
        obs["inventory"] = float(self.inventory)
        obs["portfolio_value"] = float(self.portfolio_value())

        mid = self.current_state.midprice
        if mid is not None:
            mid_f = float(mid)
            r1, r5, r10 = self._compute_midprice_returns(mid_f)
        else:
            r1 = r5 = r10 = 0.0

        obs["return_1"] = r1
        obs["return_5"] = r5
        obs["return_10"] = r10

        vol_5, vol_10 = self._compute_volatility()
        obs["vol_5"] = vol_5
        obs["vol_10"] = vol_10

        ofi_price, ofi_qty, ofi_combined = self._compute_ofi()
        obs["ofi_price"] = ofi_price
        obs["ofi_qty"] = ofi_qty
        obs["ofi_combined"] = ofi_combined

        depth_imbalance = self._compute_depth_imbalance()
        obs["depth_imbalance"] = depth_imbalance

        microprice, queue_imbalance, total_depth, spread_ratio = self._compute_microprice_and_queue()
        obs["microprice"] = microprice
        obs["queue_imbalance"] = queue_imbalance
        obs["total_depth"] = total_depth
        obs["spread_ratio"] = spread_ratio

        bid_depth_5, ask_depth_5, book_imbalance_5 = self._compute_book_depth()
        obs["bid_depth_5"] = bid_depth_5
        obs["ask_depth_5"] = ask_depth_5
        obs["book_imbalance_5"] = book_imbalance_5
        obs["impact_bps_small"] = self._estimate_impact_bps(self.action_fractions[1])
        obs["impact_bps_large"] = self._estimate_impact_bps(self.action_fractions[-1])

        impact = self._compute_market_impact(mid)
        obs["market_impact"] = impact

        slippage = info.get("slippage", 0.0)
        obs["slippage"] = slippage
        obs["execution_cost"] = abs(slippage) + abs(impact) + info.get("impact_cost", 0.0)

        time_frac, twap_target, twap_diff, vwap_price = self._compute_twap_vwap()
        obs["time_fraction"] = time_frac
        obs["twap_target"] = twap_target
        obs["twap_diff"] = twap_diff
        obs["vwap_price"] = vwap_price
        obs["remaining_inventory"] = float(self.inventory)
        obs["remaining_inventory_ratio"] = float(self.inventory / self.total_target_qty)
        obs["remaining_time"] = max(0.0, 1.0 - time_frac)
        obs["perm_impact"] = float(self.perm_impact)

        reward = self._compute_reward(info, twap_target)

        done = self.inventory <= 0 or self.step_index >= self.max_steps
        if done:
            reward += self._leftover_penalty()

        return obs, reward, done, info

    def _compute_reward(self, info: Dict[str, Any], twap_target: float) -> float:
        """Reward in basis points of arrival notional."""
        notional = self._arrival_notional()

        # Implementation shortfall of this fill vs the arrival midprice.
        executed_reward = 0.0
        if info.get("filled") and self.arrival_mid is not None:
            shortfall = (self.arrival_mid - info["fill_price"]) * info["trade_qty"]
            executed_reward = -(shortfall / notional) * 1e4  # bps

        # Small pacing nudge. Divided by max_steps so the sum over the episode
        # stays a bounded fraction of notional rather than exploding per-step.
        behind = max(0.0, twap_target - float(self.cum_traded_qty))
        schedule_penalty = (
            self.schedule_penalty_factor
            * (behind / float(self.total_target_qty))
            * 1e4
            / self.max_steps
        )

        return executed_reward - schedule_penalty

    def _leftover_penalty(self) -> float:
        """Punitive terminal cost for inventory not liquidated in time (bps)."""
        if self.inventory <= 0:
            return 0.0
        leftover_ratio = float(self.inventory) / float(self.total_target_qty)
        return -self.leftover_penalty_factor * leftover_ratio * 1e4

    def close(self):
        pass
