// Axes.tsx
import { MARGIN, VB_W, VB_H, niceTicks, type ScaleFn } from "./scale";

interface AxesProps {
  xScale: ScaleFn;
  yScale: ScaleFn;
  yDomain: [number, number];
  yFormat: (v: number) => string;
  yTickCount: number;
  xMax: number;
}

export function Axes({ xScale, yScale, yDomain, yFormat, yTickCount, xMax }: AxesProps) {
  const yTicks = niceTicks(yDomain, yTickCount);
  const xTicks = [0, Math.round(xMax * 0.25), Math.round(xMax * 0.5), Math.round(xMax * 0.75), xMax];
  return (
    <>
      {yTicks.map((t, i) => {
        const y = yScale(t);
        return (
          <g key={i}>
            <line className="grid-line" x1={MARGIN.left} x2={VB_W - MARGIN.right} y1={y} y2={y} />
            <text className="axis-label" x={MARGIN.left - 8} y={y + 3} textAnchor="end">
              {yFormat(t)}
            </text>
          </g>
        );
      })}
      <line className="axis-line" x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={VB_H - MARGIN.bottom} />
      {xTicks.map((t, i) => (
        <text key={i} className="axis-label" x={xScale(t)} y={VB_H - 4} textAnchor="middle">
          {t}
        </text>
      ))}
    </>
  );
}
