// ConnectionStatusDot.tsx
import { useEffect, useState } from "react";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../types";

export function ConnectionStatusDot() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [reachable, setReachable] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      getHealth()
        .then((h) => {
          if (!cancelled) {
            setHealth(h);
            setReachable(true);
          }
        })
        .catch(() => {
          if (!cancelled) setReachable(false);
        });
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const dotClass = !reachable ? "disconnected" : health?.status === "idle" ? "connecting" : "";
  const label = !reachable ? "backend unreachable" : health ? `backend: ${health.status}` : "connecting…";

  return (
    <div className="eyebrow">
      <span className={`dot ${dotClass}`} />
      {label}
    </div>
  );
}
