// SessionControls.tsx
import { useState } from "react";
import { TRAINED_QTY_RANGE, TRAINED_HORIZON_RANGE } from "../types";

interface SessionControlsProps {
  onStart: (targetQty: number, horizonSteps: number) => void;
  onStop: () => void;
  canStart: boolean;
  canStop: boolean;
  starting: boolean;
}

const [QTY_MIN, QTY_MAX] = TRAINED_QTY_RANGE;
const [HORIZON_MIN, HORIZON_MAX] = TRAINED_HORIZON_RANGE;

export function SessionControls({ onStart, onStop, canStart, canStop, starting }: SessionControlsProps) {
  const [targetQty, setTargetQty] = useState(25);
  const [horizonSteps, setHorizonSteps] = useState(300);

  const clampedQty = Math.min(QTY_MAX, Math.max(QTY_MIN, targetQty));
  const clampedHorizon = Math.min(HORIZON_MAX, Math.max(HORIZON_MIN, Math.round(horizonSteps)));

  return (
    <div className="controls-row">
      <div className="field">
        <label htmlFor="target-qty">
          Target (BTC, {QTY_MIN}–{QTY_MAX})
        </label>
        <input
          id="target-qty"
          type="number"
          min={QTY_MIN}
          max={QTY_MAX}
          step={0.1}
          value={targetQty}
          onChange={(e) => setTargetQty(Number(e.target.value))}
          disabled={!canStart}
        />
      </div>
      <div className="field">
        <label htmlFor="horizon-steps">
          Horizon (ticks, {HORIZON_MIN}–{HORIZON_MAX})
        </label>
        {/* Bounded, not free-form: this is the range the current model was
            actually validated on (30-window sweep across both axes before
            deployment). A value outside it wouldn't be a "faster" or
            "slower" execution, it'd be an out-of-distribution observation
            the model hasn't been checked against. */}
        <input
          id="horizon-steps"
          type="number"
          min={HORIZON_MIN}
          max={HORIZON_MAX}
          step={10}
          value={horizonSteps}
          onChange={(e) => setHorizonSteps(Number(e.target.value))}
          disabled={!canStart}
        />
      </div>
      <button
        className="btn primary"
        disabled={!canStart || starting}
        onClick={() => onStart(clampedQty, clampedHorizon)}
      >
        {starting ? "Starting…" : "Start session"}
      </button>
      <button className="btn critical" disabled={!canStop} onClick={onStop}>
        Stop
      </button>
    </div>
  );
}
