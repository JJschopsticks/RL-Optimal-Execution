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

// The model was trained assuming a 300-tick pacing schedule (time_fraction,
// twap_target, etc. all scale off horizon_steps) -- a session run with a
// different horizon isn't a fair comparison of model quality, it's a
// different problem. The live Start form no longer lets this be overridden
// (see src/api/server.py's StartSessionRequest); this constant remains only
// for flagging any older/short-horizon sessions still in history.
export const TRAINED_HORIZON = 300;
