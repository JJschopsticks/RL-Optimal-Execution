# feature_spec.py
#
# The fixed-order observation feature list the trained PPO model expects.
# Shared by every gym wrapper (backtest, live) so the observation vector can
# never silently drift between the two -- a divergence here would feed the
# model an input it was never trained on without raising any error.

FEATURE_NAMES = [
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "midprice",
    "spread",
    "cash",
    "inventory",
    "portfolio_value",
    "return_1",
    "return_5",
    "return_10",
    "vol_5",
    "vol_10",
    "ofi_price",
    "ofi_qty",
    "ofi_combined",
    "depth_imbalance",
    "microprice",
    "queue_imbalance",
    "total_depth",
    "spread_ratio",
    "time_fraction",
    "twap_target",
    "twap_diff",
    "vwap_price",
    "remaining_inventory",
    "remaining_inventory_ratio",
    "remaining_time",
    "execution_cost",
    "market_impact",
    "slippage",
    "perm_impact",
    "bid_depth_5",
    "ask_depth_5",
    "book_imbalance_5",
    "impact_bps_small",
    "impact_bps_large",
]
