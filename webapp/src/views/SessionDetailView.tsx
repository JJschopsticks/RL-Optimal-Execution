// SessionDetailView.tsx
//
// Renders one session's full traces with the same chart components as
// LiveSessionView -- the only difference is this view fetches a finished (or
// in-progress) session once over REST rather than subscribing to a socket.

import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getSession } from "../api/client";
import type { SessionDetail } from "../types";
import { ChartGroupProvider } from "../components/charts/ChartGroup";
import { PriceChart } from "../components/charts/PriceChart";
import { InventoryChart } from "../components/charts/InventoryChart";
import { RewardChart } from "../components/charts/RewardChart";
import { PnlChart } from "../components/charts/PnlChart";
import { Legend } from "../components/charts/Legend";
import { KpiRow, type KpiTile } from "../components/KpiRow";

const ALL_POLICIES = ["Trained PPO", "Baseline TWAP", "Catch-up TWAP", "Dump Everything", "No Trade"];
const REWARD_POLICIES = ["Trained PPO", "Baseline TWAP", "Catch-up TWAP", "Dump Everything"];

export function SessionDetailView() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    setDetail(null);
    setError(null);
    getSession(sessionId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e.message ?? e));
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  if (error) {
    return (
      <section className="card">
        <p className="footnote">Couldn't load session {sessionId}: {error}</p>
        <Link to="/history">&larr; back to history</Link>
      </section>
    );
  }
  if (!detail) {
    return (
      <section className="card">
        <div className="empty-state">Loading session {sessionId}…</div>
      </section>
    );
  }

  const traces = detail.traces;
  const ppoTrace = traces.find((t) => t.name === "Trained PPO")?.trace ?? [];
  const baselineTrace = traces.find((t) => t.name === "Baseline TWAP");
  const ppoSummary = traces.find((t) => t.name === "Trained PPO");
  const xMax = Math.max(0, ...traces.map((t) => t.trace.length - 1));

  const tiles: KpiTile[] = [
    {
      label: "Trained PPO",
      value: ppoSummary ? `${ppoSummary.total_reward.toFixed(2)} bps` : "—",
      hero: true,
      sub: `${ppoTrace.length} ticks`,
    },
    {
      label: "Baseline TWAP",
      value: baselineTrace ? `${baselineTrace.total_reward.toFixed(2)} bps` : "—",
    },
    {
      label: "Status",
      value: detail.status,
    },
    {
      label: "Session",
      value: sessionId ?? "",
    },
  ];

  return (
    <>
      <KpiRow tiles={tiles} />

      <section className="card">
        <div className="card-head">
          <h2>Session {sessionId}</h2>
          <span className="note">{detail.status}</span>
        </div>

        <ChartGroupProvider xMax={xMax}>
          <div className="trace-panel">
            <div className="panel-title">Midprice &amp; the trained agent's fills</div>
            <PriceChart trace={ppoTrace} />
          </div>

          <div className="trace-panel">
            <div className="panel-title">Remaining inventory (share of target)</div>
            <Legend names={ALL_POLICIES} />
            <InventoryChart traces={traces} />
          </div>

          <div className="trace-panel">
            <div className="panel-title">Cumulative reward (bps of arrival notional)</div>
            <Legend names={REWARD_POLICIES} />
            <RewardChart traces={traces} />
            <p className="footnote" style={{ marginTop: 10 }}>
              No Trade omitted -- its leftover penalty would flatten every other line to the axis.
            </p>
          </div>

          <div className="trace-panel">
            <div className="panel-title">Cumulative PnL (USD)</div>
            <Legend names={REWARD_POLICIES} />
            <PnlChart traces={traces} />
          </div>
        </ChartGroupProvider>
      </section>

      <Link to="/history">&larr; back to history</Link>
    </>
  );
}
