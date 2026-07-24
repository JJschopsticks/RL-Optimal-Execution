// PnlChart.tsx
//
// The same cumulative-reward data as RewardChart, just expressed in dollars
// (cum_pnl_usd) rather than bps -- "how much money they've made/lost," the
// literal framing the live multi-policy comparison asked for. No Trade
// excluded for the same reason as RewardChart (its leftover penalty in
// dollar terms would dwarf everything else).

import type { PolicyTrace } from "../../types";
import { MultiLineChart, type LineSeriesSpec } from "./MultiLineChart";
import { colorVar } from "./Legend";
import { REWARD_CHART_POLICIES } from "./RewardChart";

export function PnlChart({ traces }: { traces: PolicyTrace[] }) {
  const included = traces.filter((t) => REWARD_CHART_POLICIES.includes(t.name));
  const all = included.flatMap((t) => t.trace.map((r) => r.cum_pnl_usd));
  const lo = all.length ? Math.min(...all) : -1;
  const hi = Math.max(0, all.length ? Math.max(...all) : 1);
  const pad = (hi - lo) * 0.08 || 1;

  const fmtUsd = (v: number) => `${v < 0 ? "-" : ""}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  const series: LineSeriesSpec[] = included.map((t) => ({
    name: t.name,
    color: colorVar(t.name),
    trace: t.trace,
    dataKey: "cum_pnl_usd",
    fmt: fmtUsd,
  }));

  return <MultiLineChart series={series} yDomain={[lo - pad, hi + pad]} yFormat={fmtUsd} yTickCount={4} showZeroLine />;
}
