// PriceChart.tsx
//
// Trained PPO's midprice + its own fills (circle radius ~ sqrt(trade_qty),
// area-proportional). Single series, so no legend box is needed -- the panel
// title already says what's plotted (dataviz convention: a legend restates
// the title and costs space when there's only one color).

import type { TickRecord } from "../../types";
import { MARGIN, VB_W, VB_H, scale, padSeries, pathFor } from "./scale";
import { Axes } from "./Axes";
import { useChartGroup } from "./ChartGroup";
import { useCrosshairTooltip } from "./useCrosshairTooltip";
import { Tooltip, type TooltipRow } from "./Tooltip";
import { colorVar } from "./Legend";

interface PriceChartProps {
  trace: TickRecord[]; // Trained PPO's trace
}

export function PriceChart({ trace }: PriceChartProps) {
  const { xScale, xMax } = useChartGroup();
  const { hoverStep, mousePos, handleMove, handleLeave } = useCrosshairTooltip();

  const mids = trace.map((r) => r.midprice).filter((v): v is number => v !== null);
  const lo = mids.length ? Math.min(...mids) : 0;
  const hi = mids.length ? Math.max(...mids) : 1;
  const pad = (hi - lo) * 0.12 || 1;
  const yDomain: [number, number] = [lo - pad, hi + pad];
  const yScale = scale(yDomain, [VB_H - MARGIN.bottom, MARGIN.top]);

  const padded = padSeries(trace, xMax, "midprice");
  const ppoColor = colorVar("Trained PPO");

  const rows: TooltipRow[] = [];
  let timestamp: string | null = null;
  if (hoverStep !== null && trace.length > 0) {
    const idx = Math.min(hoverStep, trace.length - 1);
    const row = trace[idx];
    timestamp = row.timestamp;
    rows.push({ name: "Midprice", color: ppoColor, value: row.midprice != null ? Math.round(row.midprice).toLocaleString() : "—" });
    rows.push({ name: "PPO fill size (BTC)", color: ppoColor, value: row.filled ? row.trade_qty.toFixed(3) : "—" });
  }

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${VB_W} ${VB_H}`} onPointerMove={handleMove} onPointerLeave={handleLeave}>
        <Axes
          xScale={xScale}
          yScale={yScale}
          yDomain={yDomain}
          yFormat={(v) => Math.round(v).toLocaleString()}
          yTickCount={4}
          xMax={xMax}
        />
        <path
          d={pathFor(xScale, yScale, padded, true)}
          fill="none"
          stroke={ppoColor}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {trace.map((row, i) =>
          row.filled && row.midprice != null ? (
            <circle
              key={i}
              cx={xScale(i)}
              cy={yScale(row.midprice)}
              r={Math.max(4, Math.sqrt(row.trade_qty) * 3.2)}
              fill={ppoColor}
              stroke="var(--panel)"
              strokeWidth={2}
              opacity={0.85}
            />
          ) : null,
        )}
        {hoverStep !== null && (
          <line
            className="crosshair"
            x1={xScale(hoverStep)}
            x2={xScale(hoverStep)}
            y1={MARGIN.top}
            y2={VB_H - MARGIN.bottom}
          />
        )}
      </svg>
      {hoverStep !== null && mousePos && <Tooltip step={hoverStep} timestamp={timestamp} rows={rows} x={mousePos.x} y={mousePos.y} />}
    </div>
  );
}
