// ChartGroup.tsx
//
// Shares one x-scale/xMax and one "hovered step" across sibling charts in a
// view (e.g. SessionDetailView's price/inventory/reward panels), so the
// crosshair lines up vertically across panels even though each draws its own
// tooltip. xMax is fixed per-view (the finished episode's longest trace for
// SessionDetailView, or the session's horizon_steps for LiveSessionView) --
// not recomputed per chart -- exactly like frontend/index.html's single
// shared `xMax` used by every panel.

import { createContext, useContext, useState, type ReactNode } from "react";
import { scale, MARGIN, VB_W, type ScaleFn } from "./scale";

interface ChartGroupValue {
  xScale: ScaleFn;
  xMax: number;
  hoverStep: number | null;
  setHoverStep: (step: number | null) => void;
}

const ChartGroupContext = createContext<ChartGroupValue | null>(null);

export function ChartGroupProvider({ xMax, children }: { xMax: number; children: ReactNode }) {
  const [hoverStep, setHoverStep] = useState<number | null>(null);
  const xScale = scale([0, Math.max(xMax, 1)], [MARGIN.left, VB_W - MARGIN.right]);
  return (
    <ChartGroupContext.Provider value={{ xScale, xMax, hoverStep, setHoverStep }}>
      {children}
    </ChartGroupContext.Provider>
  );
}

export function useChartGroup(): ChartGroupValue {
  const ctx = useContext(ChartGroupContext);
  if (!ctx) throw new Error("useChartGroup must be used within a ChartGroupProvider");
  return ctx;
}

export function stepFromPointerEvent(evt: React.PointerEvent<SVGSVGElement>, xMax: number): number {
  const svg = evt.currentTarget;
  const rect = svg.getBoundingClientRect();
  const frac = (evt.clientX - rect.left) / rect.width;
  const xUser = frac * VB_W;
  const step = Math.round(((xUser - MARGIN.left) / (VB_W - MARGIN.left - MARGIN.right)) * xMax);
  return Math.max(0, Math.min(xMax, step));
}
