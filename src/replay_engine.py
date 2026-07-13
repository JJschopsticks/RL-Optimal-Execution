# replay_engine.py

import os
import json
from decimal import Decimal
from typing import Dict, Any, Optional, List
import math

CLEANED_DIR = "../cleaned"   # adjust if needed


class ReplayState:
    def __init__(self, timestamp: str, best_bid: Optional[List[str]],
                 best_ask: Optional[List[str]], midprice: Optional[str]):
        self.timestamp = timestamp
        self.best_bid_price = Decimal(best_bid[0]) if best_bid else None
        self.best_bid_qty = Decimal(best_bid[1]) if best_bid else None
        self.best_ask_price = Decimal(best_ask[0]) if best_ask else None
        self.best_ask_qty = Decimal(best_ask[1]) if best_ask else None
        self.midprice = Decimal(midprice) if midprice else None

    def to_features(self) -> Dict[str, Any]:
        spread = None
        if self.best_bid_price is not None and self.best_ask_price is not None:
            spread = self.best_ask_price - self.best_bid_price

        return {
            "timestamp": self.timestamp,
            "best_bid_price": float(self.best_bid_price) if self.best_bid_price else None,
            "best_bid_qty": float(self.best_bid_qty) if self.best_bid_qty else None,
            "best_ask_price": float(self.best_ask_price) if self.best_ask_price else None,
            "best_ask_qty": float(self.best_ask_qty) if self.best_ask_qty else None,
            "midprice": float(self.midprice) if self.midprice else None,
            "spread": float(spread) if spread is not None else None,
        }


class SmartOrderRouterReplay:
    def __init__(self, date_str: str, initial_cash: float = 10000.0, trade_size: float = 0.001):
        self.date_str = date_str
        self.file_path = os.path.join(CLEANED_DIR, f"{date_str}.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Cleaned file not found: {self.file_path}")

        self.file = open(self.file_path, "r")
        self.current_state: Optional[ReplayState] = None

        self.initial_cash = Decimal(str(initial_cash))
        self.cash = Decimal(str(initial_cash))
        self.inventory = Decimal("0")
        self.trade_size = Decimal(str(trade_size))

        self.last_portfolio_value = self.portfolio_value()

        self.midprice_history: List[float] = []
        self.return_history: List[float] = []

        self.prev_bid_price: Optional[Decimal] = None
        self.prev_ask_price: Optional[Decimal] = None
        self.prev_bid_qty: Optional[Decimal] = None
        self.prev_ask_qty: Optional[Decimal] = None

        self.prev_midprice: Optional[Decimal] = None

        # Phase 2: execution metrics
        self.step_index = 0
        self.max_steps = 10000  # rough horizon for time-normalization
        self.cum_traded_qty = Decimal("0")
        self.total_target_qty = Decimal("1.0")  # pretend we want to execute 1 BTC over the day

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

    def _compute_ofi(self) -> (float, float, float):
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
        di = (float(bid_q) - float(ask_q)) / total
        return float(di)

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
            qi = 0.0
            total_depth = 0.0
            spread_ratio = 0.0
            return micro, qi, total_depth, spread_ratio

        micro = (float(bid_p) * float(ask_q) + float(ask_p) * float(bid_q)) / total_q
        qi = float(bid_q) / total_q
        total_depth = total_q

        if mid is not None and mid != 0:
            spread = float(ask_p - bid_p)
            spread_ratio = spread / float(mid)
        else:
            spread_ratio = 0.0

        return float(micro), float(qi), float(total_depth), float(spread_ratio)

    def _compute_market_impact(self, mid: Optional[Decimal]) -> float:
        if mid is None or self.prev_midprice is None:
            impact = 0.0
        else:
            impact = float(mid - self.prev_midprice)
        self.prev_midprice = mid
        return impact

    def _compute_twap_vwap(self) -> (float, float, float):
        # time fraction in [0,1]
        time_frac = min(self.step_index / self.max_steps, 1.0)

        # TWAP target: linear schedule of total_target_qty
        twap_target = float(self.total_target_qty) * time_frac

        # TWAP deviation: how much we've executed vs target
        twap_diff = float(self.cum_traded_qty - self.total_target_qty * Decimal(str(time_frac)))

        # VWAP baseline: simple average of midprices so far
        if len(self.midprice_history) > 0:
            vwap_price = sum(self.midprice_history) / len(self.midprice_history)
        else:
            vwap_price = 0.0

        return float(time_frac), float(twap_target), float(twap_diff), float(vwap_price)

    def reset(self):
        self.file.seek(0)
        self.current_state = None
        self.cash = self.initial_cash
        self.inventory = Decimal("0")
        self.last_portfolio_value = self.portfolio_value()
        self.midprice_history = []
        self.return_history = []

        self.prev_bid_price = None
        self.prev_ask_price = None
        self.prev_bid_qty = None
        self.prev_ask_qty = None
        self.prev_midprice = None

        self.step_index = 0
        self.cum_traded_qty = Decimal("0")

        obs, _, done, _ = self.step(action=None)
        return obs

    def _execute_action(self, action: Optional[str]):
        info = {"filled": False, "fill_price": None, "slippage": 0.0}

        if action is None or self.current_state is None:
            return info

        if action == "hold":
            return info

        mid = float(self.current_state.midprice) if self.current_state.midprice is not None else None

        if action == "market_buy":
            if self.current_state.best_ask_price is None:
                return info
            cost = self.trade_size * self.current_state.best_ask_price
            if cost > self.cash:
                return info
            self.cash -= cost
            self.inventory += self.trade_size
            self.cum_traded_qty += self.trade_size
            fill_price = float(self.current_state.best_ask_price)
            info["filled"] = True
            info["fill_price"] = fill_price
            if mid is not None:
                info["slippage"] = fill_price - mid

        elif action == "market_sell":
            if self.current_state.best_bid_price is None:
                return info
            if self.inventory < self.trade_size:
                return info
            proceeds = self.trade_size * self.current_state.best_bid_price
            self.cash += proceeds
            self.inventory -= self.trade_size
            self.cum_traded_qty += self.trade_size
            fill_price = float(self.current_state.best_bid_price)
            info["filled"] = True
            info["fill_price"] = fill_price
            if mid is not None:
                info["slippage"] = mid - fill_price

        return info

    def step(self, action: Optional[str] = None):
        line = self.file.readline()
        if not line:
            obs = self.current_state.to_features() if self.current_state else {}
            return obs, 0.0, True, {"end_of_day": True}

        data = json.loads(line)
        self.current_state = ReplayState(
            timestamp=data["timestamp"],
            best_bid=data["best_bid"],
            best_ask=data["best_ask"],
            midprice=data["midprice"]
        )

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

        # Phase 2: execution metrics
        impact = self._compute_market_impact(mid)
        obs["market_impact"] = impact

        slippage = info.get("slippage", 0.0)
        obs["slippage"] = slippage

        # execution cost: absolute slippage + absolute impact
        exec_cost = abs(slippage) + abs(impact)
        obs["execution_cost"] = exec_cost

        time_frac, twap_target, twap_diff, vwap_price = self._compute_twap_vwap()
        obs["time_fraction"] = time_frac
        obs["twap_target"] = twap_target
        obs["twap_diff"] = twap_diff
        obs["vwap_price"] = vwap_price

        current_value = self.portfolio_value()
        # reward: portfolio change minus execution cost
        reward = float(current_value - self.last_portfolio_value) - exec_cost
        self.last_portfolio_value = current_value

        return obs, reward, False, info

    def close(self):
        self.file.close()


def main():
        engine = SmartOrderRouterReplay("2026-07-11")
        obs = engine.reset()
        print("First obs:", obs)

        actions = ["hold", "market_buy", "hold", "market_sell"] * 3

        for a in actions:
            obs, reward, done, info = engine.step(a)
            print(
                f"Action: {a} | Reward: {reward:.6f} | "
                f"slip: {obs['slippage']:.6e} | impact: {obs['market_impact']:.6e} | "
                f"cost: {obs['execution_cost']:.6e} | time: {obs['time_fraction']:.3f} | "
                f"twap_diff: {obs['twap_diff']:.6e}"
            )
            if done:
                break

        engine.close()


if __name__ == "__main__":
    main()
