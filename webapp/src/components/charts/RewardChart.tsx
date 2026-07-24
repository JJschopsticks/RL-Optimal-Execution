// RewardChart.tsx
//
// Cumulative reward (bps of arrival notional) for four policies -- No Trade
// is excluded here: its leftover penalty (-5,502 bps, applied once at window
// close) would flatten every other line to the axis, same reasoning as
// frontend/index.html's backtest dashboard.

import type { PolicyTrace } from "../../types";
import { MultiLineChart, type LineSeriesSpec } from "./MultiLineChart";
import { colorVar } from "./Legend";

export const REWARD_CHART_POLICIES = ["Trained PPO", "Baseline TWAP", "Catch-up TWAP", "Dump Everything"];

export function RewardChart({ traces }: { traces: PolicyTrace[] }) {
  const included = traces.filter((t) => REWARD_CHART_POLICIES.includes(t.name));
  const all = included.flatMap((t) => t.trace.map((r) => r.cum_reward));
  const lo = all.length ? Math.min(...all) : -1;
  const hi = Math.max(0, all.length ? Math.max(...all) : 1);
  const pad = (hi - lo) * 0.08 || 1;

  const series: LineSeriesSpec[] = included.map((t) => ({
    name: t.name,
    color: colorVar(t.name),
    trace: t.trace,
    dataKey: "cum_reward",
    fmt: (v: number) => `${v.toFixed(1)} bps`,
  }));

  return (
    <MultiLineChart
      series={series}
      yDomain={[lo - pad, hi + pad]}
      yFormat={(v) => v.toFixed(1)}
      yTickCount={4}
      showZeroLine
    />
  );
}
