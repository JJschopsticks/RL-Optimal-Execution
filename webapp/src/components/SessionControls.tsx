// SessionControls.tsx
import { useState } from "react";
import { TRAINED_HORIZON } from "../types";

interface SessionControlsProps {
  onStart: (targetQty: number) => void;
  onStop: () => void;
  canStart: boolean;
  canStop: boolean;
  starting: boolean;
}

export function SessionControls({ onStart, onStop, canStart, canStop, starting }: SessionControlsProps) {
  const [targetQty, setTargetQty] = useState(25);

  return (
    <div className="controls-row">
      <div className="field">
        <label htmlFor="target-qty">Target (BTC)</label>
        <input
          id="target-qty"
          type="number"
          min={0.1}
          step={0.1}
          value={targetQty}
          onChange={(e) => setTargetQty(Number(e.target.value))}
          disabled={!canStart}
        />
      </div>
      <div className="field">
        <label htmlFor="horizon-steps">Horizon (ticks)</label>
        {/* Fixed, not editable: the model was trained on a 300-tick pacing
            schedule, so a different horizon isn't a fair comparison -- it's
            an out-of-distribution observation, not a "faster" execution. */}
        <input id="horizon-steps" type="text" value={`${TRAINED_HORIZON} (fixed)`} disabled readOnly />
      </div>
      <button className="btn primary" disabled={!canStart || starting} onClick={() => onStart(targetQty)}>
        {starting ? "Starting…" : "Start session"}
      </button>
      <button className="btn critical" disabled={!canStop} onClick={onStop}>
        Stop
      </button>
    </div>
  );
}
