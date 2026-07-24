// InventoryChart.tsx
//
// Remaining inventory (share of target) for all five policies -- the core
// "splitting into pieces" visual. Bounded 0-1, so unlike the reward/PnL
// charts, No Trade's flat line at 1.0 doesn't distort the scale and stays in.

import type { PolicyTrace } from "../../types";
import { MultiLineChart, type LineSeriesSpec } from "./MultiLineChart";
import { colorVar } from "./Legend";

export function InventoryChart({ traces }: { traces: PolicyTrace[] }) {
  const series: LineSeriesSpec[] = traces.map((t) => ({
    name: t.name,
    color: colorVar(t.name),
    trace: t.trace,
    dataKey: "remaining_inventory_ratio",
    fmt: (v: number) => `${(v * 100).toFixed(1)}%`,
  }));

  return <MultiLineChart series={series} yDomain={[0, 1]} yFormat={(v) => v.toFixed(1)} yTickCount={4} />;
}
