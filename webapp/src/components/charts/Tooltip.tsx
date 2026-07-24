// Tooltip.tsx
//
// One tooltip, every series at that step -- the reader never has to land on
// a specific line to get a value (see the dataviz interaction convention
// this whole dashboard already follows in frontend/index.html).

export interface TooltipRow {
  name: string;
  color: string;
  value: string;
}

interface TooltipProps {
  step: number;
  timestamp: string | null;
  rows: TooltipRow[];
  x: number;
  y: number;
}

export function Tooltip({ step, timestamp, rows, x, y }: TooltipProps) {
  const timeLabel = timestamp ? `${timestamp.slice(9, 11)}:${timestamp.slice(11, 13)}:${timestamp.slice(13, 15)}` : "";
  return (
    <div className="tooltip show" style={{ left: x + 16, top: y - 10 }}>
      <div className="t-head">
        step {step}
        {timeLabel ? ` · ${timeLabel}` : ""}
      </div>
      {rows.map((r) => (
        <div className="row" key={r.name}>
          <span className="key" style={{ background: r.color }} />
          <span className="name">{r.name}</span>
          <span className="val">{r.value}</span>
        </div>
      ))}
    </div>
  );
}
