# replay_engine_v2_sor.py

import os
import json
from decimal import Decimal
from typing import Dict, Any, Optional, List

CLEANED_DIR = "cleaned"


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
        """
        date_str: e.g. '2026-07-11'
        initial_cash: starting USDT
        trade_size: BTC size per market order
        """
        self.date_str = date_str
        self.file_path = os.path.join(CLEANED_DIR, f"{date_str}.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Cleaned file not found: {self.file_path}")

        self.file = open(self.file_path, "r")
        self.current_state: Optional[ReplayState] = None

        # Execution / portfolio state
        self.initial_cash = Decimal(str(initial_cash))
        self.cash = Decimal(str(initial_cash))
        self.inventory = Decimal("0")  # BTC
        self.trade_size = Decimal(str(trade_size))

        # For simple reward: PnL change per step
        self.last_portfolio_value = self.portfolio_value()

    def portfolio_value(self) -> Decimal:
        if self.current_state and self.current_state.midprice is not None:
            return self.cash + self.inventory * self.current_state.midprice
        return self.cash

    def reset(self) -> Dict[str, Any]:
        self.file.seek(0)
        self.current_state = None
        self.cash = self.initial_cash
        self.inventory = Decimal("0")
        self.last_portfolio_value = self.portfolio_value()
        obs, _, done, _ = self.step(action=None)
        if done:
            return {}
        return obs

    def _execute_action(self, action: Optional[str]) -> Dict[str, Any]:
        """
        action: 'hold', 'market_buy', 'market_sell' or None
        """
        info: Dict[str, Any] = {"filled": False, "fill_price": None}

        if action is None or self.current_state is None:
            return info

        if action == "hold":
            return info

        # Simple single-venue execution model:
        # - market_buy: fill at best_ask_price
        # - market_sell: fill at best_bid_price
        if action == "market_buy":
            if self.current_state.best_ask_price is None:
                return info
            cost = self.trade_size * self.current_state.best_ask_price
            if cost > self.cash:
                return info  # not enough cash
            self.cash -= cost
            self.inventory += self.trade_size
            info["filled"] = True
            info["fill_price"] = float(self.current_state.best_ask_price)

        elif action == "market_sell":
            if self.current_state.best_bid_price is None:
                return info
            if self.inventory < self.trade_size:
                return info  # not enough BTC
            proceeds = self.trade_size * self.current_state.best_bid_price
            self.cash += proceeds
            self.inventory -= self.trade_size
            info["filled"] = True
            info["fill_price"] = float(self.current_state.best_bid_price)

        return info

    def step(self, action: Optional[str] = None) -> (Dict[str, Any], float, bool, Dict[str, Any]):
        """
        Advance one tick.

        action: 'hold', 'market_buy', 'market_sell' or None

        Returns: (observation, reward, done, info)
        """
        line = self.file.readline()
        if not line:
            # end of file / end of day
            obs = self.current_state.to_features() if self.current_state else {}
            reward = 0.0
            done = True
            info: Dict[str, Any] = {"end_of_day": True}
            return obs, reward, done, info

        data = json.loads(line)
        self.current_state = ReplayState(
            timestamp=data["timestamp"],
            best_bid=data["best_bid"],
            best_ask=data["best_ask"],
            midprice=data["midprice"]
        )

        # Execute action at this tick
        info = self._execute_action(action)

        # Build observation
        obs = self.current_state.to_features()
        obs["cash"] = float(self.cash)
        obs["inventory"] = float(self.inventory)
        obs["portfolio_value"] = float(self.portfolio_value())

        # Reward: change in portfolio value since last step
        current_value = self.portfolio_value()
        reward = float(current_value - self.last_portfolio_value)
        self.last_portfolio_value = current_value

        done = False
        return obs, reward, done, info

    def close(self):
        self.file.close()


def main():
    engine = SmartOrderRouterReplay("2026-07-11")

    obs = engine.reset()
    print("First obs:", obs)

    # Example: naive strategy that alternates buy/sell
    actions = ["hold", "market_buy", "hold", "market_sell"] * 5

    for a in actions:
        obs, reward, done, info = engine.step(action=a)
        print(f"Action: {a} | Reward: {reward:.6f} | Obs midprice: {obs.get('midprice')} | PV: {obs.get('portfolio_value')}")
        if done:
            break

    engine.close()


if __name__ == "__main__":
    main()
