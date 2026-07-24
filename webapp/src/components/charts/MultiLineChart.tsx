// MultiLineChart.tsx
//
// Generic multi-series line chart used by InventoryChart, RewardChart, and
// PnlChart. Reused, not duplicated per chart, because they only differ in
// which field they plot and its y-domain/format. Each series is padded (held
// flat, dashed) to the group's shared xMax exactly like frontend/index.html's
// backtest dashboard -- a policy that finishes early just holds its last
// value, whether that's because its backtest episode ended or because it's
// paused live while others keep trading.

import type { TickRecord } from "../../types";
import { MARGIN, VB_W, VB_H, scale, padSeries, pathFor, pathForHeld } from "./scale";
import { Axes } from "./Axes";
import { useChartGroup } from "./ChartGroup";
import { useCrosshairTooltip } from "./useCrosshairTooltip";
import { Tooltip, type TooltipRow } from "./Tooltip";

export interface LineSeriesSpec {
  name: string;
  color: string;
  trace: TickRecord[];
  dataKey: keyof TickRecord;
  fmt: (v: number) => string;
}

interface MultiLineChartProps {
  series: LineSeriesSpec[];
  yDomain: [number, number];
  yFormat: (v: number) => string;
  yTickCount?: number;
  showZeroLine?: boolean;
}

export function MultiLineChart({ series, yDomain, yFormat, yTickCount = 4, showZeroLine = false }: MultiLineChartProps) {
  const { xScale, xMax } = useChartGroup();
  const { hoverStep, mousePos, handleMove, handleLeave } = useCrosshairTooltip();
  const yScale = scale(yDomain, [VB_H - MARGIN.bottom, MARGIN.top]);

  const padded = series.map((s) => ({ ...s, points: padSeries(s.trace, xMax, s.dataKey) }));

  const headTrace = padded[0]?.trace ?? [];
  const timestamp =
    hoverStep !== null && headTrace.length > 0 ? headTrace[Math.min(hoverStep, headTrace.length - 1)].timestamp : null;

  const rows: TooltipRow[] =
    hoverStep !== null
      ? padded.map((s) => {
          const point = s.points[hoverStep];
          return { name: s.name, color: s.color, value: s.fmt(point.v) + (point.real ? "" : " (done)") };
        })
      : [];

  return (
    <div className="chart-wrap">
      <svg className="chart" viewBox={`0 0 ${VB_W} ${VB_H}`} onPointerMove={handleMove} onPointerLeave={handleLeave}>
        <Axes xScale={xScale} yScale={yScale} yDomain={yDomain} yFormat={yFormat} yTickCount={yTickCount} xMax={xMax} />
        {showZeroLine && (
          <line
            className="axis-line"
            x1={MARGIN.left}
            x2={VB_W - MARGIN.right}
            y1={yScale(0)}
            y2={yScale(0)}
            opacity={0.5}
          />
        )}
        {padded.map((s) => {
          const heldD = pathForHeld(xScale, yScale, s.points);
          return (
            <g key={s.name}>
              <path
                d={pathFor(xScale, yScale, s.points, true)}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {heldD && (
                <path d={heldD} fill="none" stroke={s.color} strokeWidth={1.4} strokeDasharray="3 4" opacity={0.55} />
              )}
            </g>
          );
        })}
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
