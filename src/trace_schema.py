# trace_schema.py
#
# The tick-by-tick trace record shared between the offline backtest exporter
# (export_dashboard_data.py) and the live paper-trading session logger, so a
# live session and a backtest trace are always structurally identical and can
# be rendered by the same chart components.

from typing import Dict


def build_record(engine, step: int, action: int, reward: float, cum_reward: float, info: Dict) -> Dict:
    cs = engine.current_state
    # cum_reward is already implementation shortfall in bps of arrival
    # notional (see ExecutionCore._compute_reward); converting it to a raw
    # dollar figure is just that same number expressed in the other unit --
    # no new simulation logic, just "how much this policy has made/lost."
    notional = engine._arrival_notional()
    return {
        "step": step,
        "timestamp": cs.timestamp if cs else None,
        "midprice": float(cs.midprice) if cs and cs.midprice is not None else None,
        "best_bid": float(cs.best_bid_price) if cs and cs.best_bid_price is not None else None,
        "best_ask": float(cs.best_ask_price) if cs and cs.best_ask_price is not None else None,
        "action": int(action),
        "filled": bool(info.get("filled")),
        "trade_qty": float(info.get("trade_qty", 0.0)),
        "fill_price": info.get("fill_price"),
        "reward": float(reward),
        "cum_reward": float(cum_reward),
        "cum_pnl_usd": float(cum_reward) * notional / 1e4,
        "remaining_inventory_ratio": float(engine.inventory / engine.total_target_qty),
        "perm_impact": float(engine.perm_impact),
    }
