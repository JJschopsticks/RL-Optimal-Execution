// useCrosshairTooltip.ts
//
// Shared pointer-interaction logic for every chart in a ChartGroup: reports
// the hovered step (synced across sibling panels via context) and the local
// mouse position (for that chart's own tooltip placement within its own
// .chart-wrap).

import { useCallback, useState } from "react";
import { useChartGroup, stepFromPointerEvent } from "./ChartGroup";

export function useCrosshairTooltip() {
  const { xMax, hoverStep, setHoverStep } = useChartGroup();
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  const handleMove = useCallback(
    (evt: React.PointerEvent<SVGSVGElement>) => {
      setHoverStep(stepFromPointerEvent(evt, xMax));
      const rect = evt.currentTarget.getBoundingClientRect();
      setMousePos({ x: evt.clientX - rect.left, y: evt.clientY - rect.top });
    },
    [xMax, setHoverStep],
  );

  const handleLeave = useCallback(() => {
    setHoverStep(null);
    setMousePos(null);
  }, [setHoverStep]);

  return { hoverStep, mousePos, handleMove, handleLeave };
}
