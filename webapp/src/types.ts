// types.ts
//
// Mirrors src/trace_schema.py's build_record() and src/paper_trading_session.py's
// to_summary()/to_traces()/meta.json shapes exactly, so a live session and a
// finished session (read back from disk or from dashboard_data.json) are the
// same shape everywhere in this app.

export interface TickRecord {
  step: number;
  timestamp: string | null;
  midprice: number | null;
  best_bid: number | null;
  best_ask: number | null;
  action: number;
  filled: boolean;
  trade_qty: number;
  fill_price: number | null;
  reward: number;
  cum_reward: number;
  cum_pnl_usd: number;
  remaining_inventory_ratio: number;
  perm_impact: number;
}

export interface PolicyTrace {
  name: string;
  trace: TickRecord[];
  total_reward: number;
}

export interface SessionDetail {
  session_id: string;
  status: SessionStatus;
  traces: PolicyTrace[];
}

export type SessionStatus = "starting" | "warming_up" | "running" | "completed" | "stopped" | "error";

export interface PolicySummary {
  steps: number;
  total_reward: number;
}

export interface SessionSummary {
  session_id: string;
  status: SessionStatus;
  start_time: string | null;
  end_time: string | null;
  total_target_qty: number;
  horizon_steps: number;
  policies: Record<string, PolicySummary>;
}

export interface WarmupProgress {
  n_events: number;
  warmup_events: number;
}

export interface HealthResponse {
  status: SessionStatus | "idle";
  order_book_status: string | null;
  warmup_progress: WarmupProgress | null;
}

// WebSocket message shapes (see api/server.py's stream_session)
export type StreamMessage =
  | { type: "tick"; policy: string; data: TickRecord }
  | { type: "status"; status: SessionStatus }
  | { type: "error"; detail: string };

export const POLICY_NAMES = [
  "Dump Everything",
  "Baseline TWAP",
  "Catch-up TWAP",
  "No Trade",
  "Trained PPO",
] as const;

export type PolicyName = (typeof POLICY_NAMES)[number];

export const POLICY_COLOR_VAR: Record<string, string> = {
  "Trained PPO": "--series-1",
  "Baseline TWAP": "--series-2",
  "Catch-up TWAP": "--series-3",
  "Dump Everything": "--series-4",
  "No Trade": "--series-5",
};

// The current model was retrained with domain randomization across these
// ranges (total_target_qty and horizon_steps both resampled per training
// episode -- see src/train_rl.py), specifically to fix an earlier model that
// only ever saw 25 BTC / 300 ticks and fell apart outside that single point
// (confirmed live: -131 bps vs TWAP's -24 bps at 100 BTC). Validated via a
// 30-window eval sweep across both axes before being deployed. The Start
// form bounds its inputs to these ranges, and sessions from before this
// retrain (fixed at exactly 300) are flagged in History as not comparable.
export const TRAINED_QTY_RANGE: [number, number] = [5, 100];
export const TRAINED_HORIZON_RANGE: [number, number] = [150, 450];
export const TRAINED_HORIZON = 300; // the single point every pre-retrain session used
