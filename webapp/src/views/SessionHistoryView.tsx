// SessionHistoryView.tsx
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listSessions } from "../api/client";
import { TRAINED_QTY_RANGE, TRAINED_HORIZON_RANGE } from "../types";
import type { SessionSummary } from "../types";

const [QTY_MIN, QTY_MAX] = TRAINED_QTY_RANGE;
const [HORIZON_MIN, HORIZON_MAX] = TRAINED_HORIZON_RANGE;

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
}

function outOfRange(value: number | undefined, min: number, max: number): boolean {
  return value === undefined || value < min || value > max;
}

export function SessionHistoryView() {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listSessions()
      .then((rows) => {
        if (!cancelled) setSessions(rows);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e.message ?? e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="card">
      <div className="card-head">
        <h2>Session history</h2>
        <span className="note">every paper-trading run, live or finished</span>
      </div>
      <p className="desc">
        The current model is validated for target sizes {QTY_MIN}-{QTY_MAX} BTC and horizons {HORIZON_MIN}-
        {HORIZON_MAX} ticks (trained with both randomized episode-to-episode). Rows outside that range predate this
        model or were run against it anyway -- they aren't comparable to the rest, since the observation the model
        sees is out-of-distribution rather than a "harder" version of the same problem.
      </p>

      {error && <p className="footnote">Couldn't reach the backend: {error}</p>}
      {!error && sessions === null && <div className="empty-state">Loading…</div>}
      {!error && sessions !== null && sessions.length === 0 && (
        <div className="empty-state">No sessions yet — start one from the Live tab.</div>
      )}

      {sessions && sessions.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table className="session-table">
            <thead>
              <tr>
                <th>Session</th>
                <th>Status</th>
                <th>Started</th>
                <th>Target</th>
                <th>Horizon</th>
                <th>Trained PPO</th>
                <th>Baseline TWAP</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => {
                const qtyFlagged = outOfRange(s.total_target_qty, QTY_MIN, QTY_MAX);
                const horizonFlagged = outOfRange(s.horizon_steps, HORIZON_MIN, HORIZON_MAX);
                return (
                  <tr key={s.session_id} className="clickable">
                    <td className="mono">
                      <Link to={`/history/${s.session_id}`}>{s.session_id}</Link>
                    </td>
                    <td>
                      <span className={`status-pill ${s.status}`}>
                        <span className="dot" /> {s.status}
                      </span>
                    </td>
                    <td className="mono">{fmtTime(s.start_time)}</td>
                    <td
                      className="mono"
                      style={qtyFlagged ? { color: "var(--faint)" } : undefined}
                      title={qtyFlagged ? `Outside the validated range (${QTY_MIN}-${QTY_MAX} BTC)` : undefined}
                    >
                      {s.total_target_qty ?? "—"}
                      {qtyFlagged ? " *" : ""}
                    </td>
                    <td
                      className="mono"
                      style={horizonFlagged ? { color: "var(--faint)" } : undefined}
                      title={horizonFlagged ? `Outside the validated range (${HORIZON_MIN}-${HORIZON_MAX} ticks)` : undefined}
                    >
                      {s.horizon_steps ?? "—"}
                      {horizonFlagged ? " *" : ""}
                    </td>
                    <td className="mono">
                      {s.policies["Trained PPO"] ? `${s.policies["Trained PPO"].total_reward.toFixed(2)} bps` : "—"}
                    </td>
                    <td className="mono">
                      {s.policies["Baseline TWAP"] ? `${s.policies["Baseline TWAP"].total_reward.toFixed(2)} bps` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
