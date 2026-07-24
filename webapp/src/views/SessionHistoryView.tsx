// SessionHistoryView.tsx
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listSessions } from "../api/client";
import { TRAINED_HORIZON } from "../types";
import type { SessionSummary } from "../types";

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19) + " UTC";
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
        Rows with a horizon other than {TRAINED_HORIZON} ticks aren't comparable to the rest -- the model was
        trained assuming a {TRAINED_HORIZON}-tick pacing schedule, so a shorter window gives it the wrong sense of
        urgency rather than reflecting worse model quality.
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
                <th>Horizon</th>
                <th>Trained PPO</th>
                <th>Baseline TWAP</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => {
                const nonStandardHorizon = s.horizon_steps !== TRAINED_HORIZON;
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
                      style={nonStandardHorizon ? { color: "var(--faint)" } : undefined}
                      title={nonStandardHorizon ? `Not the trained horizon (${TRAINED_HORIZON} ticks) -- not comparable` : undefined}
                    >
                      {s.horizon_steps ?? "—"}
                      {nonStandardHorizon ? " *" : ""}
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
