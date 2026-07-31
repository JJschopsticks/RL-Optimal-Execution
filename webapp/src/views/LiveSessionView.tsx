// LiveSessionView.tsx
//
// Starts/stops a paper-trading session and renders it live via
// useSessionSocket, feeding the same chart components SessionDetailView
// uses. Unlike the detail view's fixed (final) xMax, this uses the *current*
// live progress as xMax -- not the eventual horizon_steps -- so a policy
// that simply hasn't reached tick N yet (because real time hasn't gotten
// there) isn't drawn with the "finished early, holding flat" dashed
// treatment; only policies genuinely behind the live pack get that.

import { useCallback, useEffect, useState } from "react";
import { listSessions, startSession, stopSession } from "../api/client";
import { useSessionSocket } from "../hooks/useSessionSocket";
import { POLICY_NAMES } from "../types";
import type { PolicyTrace } from "../types";
import { ChartGroupProvider } from "../components/charts/ChartGroup";
import { PriceChart } from "../components/charts/PriceChart";
import { InventoryChart } from "../components/charts/InventoryChart";
import { RewardChart, REWARD_CHART_POLICIES } from "../components/charts/RewardChart";
import { PnlChart } from "../components/charts/PnlChart";
import { Legend } from "../components/charts/Legend";
import { KpiRow, type KpiTile } from "../components/KpiRow";
import { SessionControls } from "../components/SessionControls";

const ACTIVE_STATUSES = ["starting", "warming_up", "running"];

export function LiveSessionView() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const { recordsByPolicy, status, connected } = useSessionSocket(sessionId);

  // Reattach to an already-active session on mount (e.g. after a page refresh).
  useEffect(() => {
    listSessions()
      .then((rows) => {
        const active = rows.find((r) => ACTIVE_STATUSES.includes(r.status));
        if (active) setSessionId(active.session_id);
      })
      .catch(() => {});
  }, []);

  const handleStart = useCallback(async (targetQty: number, horizonSteps: number) => {
    setStarting(true);
    setErrorMsg(null);
    try {
      const res = await startSession({ total_target_qty: targetQty, horizon_steps: horizonSteps });
      setSessionId(res.session_id);
    } catch (e) {
      setErrorMsg(String((e as Error).message ?? e));
    } finally {
      setStarting(false);
    }
  }, []);

  const handleStop = useCallback(() => {
    if (sessionId) stopSession(sessionId).catch((e) => setErrorMsg(String(e.message ?? e)));
  }, [sessionId]);

  const isActive = status === null ? !!sessionId : ACTIVE_STATUSES.includes(status);
  const canStart = !sessionId || !isActive;
  const canStop = !!sessionId && isActive;

  const traces: PolicyTrace[] = POLICY_NAMES.map((name) => {
    const trace = recordsByPolicy[name] ?? [];
    return { name, trace, total_reward: trace.length ? trace[trace.length - 1].cum_reward : 0 };
  });
  const ppoTrace = traces.find((t) => t.name === "Trained PPO")?.trace ?? [];
  const ppoSummary = traces.find((t) => t.name === "Trained PPO");
  const baselineSummary = traces.find((t) => t.name === "Baseline TWAP");

  // Current live progress, not the eventual horizon -- see file header.
  const xMax = Math.max(0, ...traces.map((t) => t.trace.length - 1));
  // The book is seeded from a REST snapshot now (~3s), so this state is
  // brief. It previously needed a polled n_events/100 progress counter
  // because bootstrapping replayed ~100 diff events over ~100s.
  const isWaitingForWarmup = !!sessionId && xMax === 0 && ppoTrace.length === 0;

  const tiles: KpiTile[] = [
    {
      label: "Trained PPO",
      value: ppoTrace.length ? `${(ppoSummary?.total_reward ?? 0).toFixed(2)} bps` : "—",
      hero: true,
      sub: `${ppoTrace.length} ticks`,
    },
    {
      label: "Baseline TWAP",
      value: baselineSummary && baselineSummary.trace.length ? `${baselineSummary.total_reward.toFixed(2)} bps` : "—",
    },
    { label: "Status", value: status ?? (sessionId ? "connecting…" : "idle") },
    { label: "Session", value: sessionId ?? "none" },
  ];

  return (
    <>
      <section className="card">
        <div className="card-head">
          <h2>Start a paper-trading session</h2>
          <span className="note">
            {connected ? "stream connected" : sessionId ? "connecting…" : "no active session"}
          </span>
        </div>
        <p className="desc">
          Runs all five policies concurrently against the live order book, fully simulated. The book is seeded
          from a REST depth snapshot, so trading starts within a few seconds.
        </p>
        <SessionControls onStart={handleStart} onStop={handleStop} canStart={canStart} canStop={canStop} starting={starting} />
        {errorMsg && (
          <p className="footnote" style={{ marginTop: 10, color: "var(--critical)" }}>
            {errorMsg}
          </p>
        )}
      </section>

      {sessionId && (
        <>
          <KpiRow tiles={tiles} />

          <section className="card">
            <div className="card-head">
              <h2>Live execution</h2>
              <span className="note">{sessionId}</span>
            </div>

            {isWaitingForWarmup ? (
              <div className="empty-state">Syncing the order book snapshot…</div>
            ) : (
              <ChartGroupProvider xMax={xMax}>
                <div className="trace-panel">
                  <div className="panel-title">Midprice &amp; the trained agent's fills</div>
                  <PriceChart trace={ppoTrace} />
                </div>

                <div className="trace-panel">
                  <div className="panel-title">Remaining inventory (share of target)</div>
                  <Legend names={[...POLICY_NAMES]} />
                  <InventoryChart traces={traces} />
                </div>

                <div className="trace-panel">
                  <div className="panel-title">Cumulative reward (bps of arrival notional)</div>
                  <Legend names={REWARD_CHART_POLICIES} />
                  <RewardChart traces={traces} />
                </div>

                <div className="trace-panel">
                  <div className="panel-title">Cumulative PnL (USD)</div>
                  <Legend names={REWARD_CHART_POLICIES} />
                  <PnlChart traces={traces} />
                </div>
              </ChartGroupProvider>
            )}
          </section>
        </>
      )}
    </>
  );
}
