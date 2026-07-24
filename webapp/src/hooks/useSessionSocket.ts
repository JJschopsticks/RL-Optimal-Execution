// useSessionSocket.ts
//
// Subscribes to a session's WS stream and accumulates tick records per
// policy as they arrive. Records grow monotonically per policy; a policy
// that finishes early simply stops receiving new records (MultiLineChart's
// padSeries/pathForHeld handles drawing its held-flat dashed tail from
// whatever xMax the caller gives it -- see LiveSessionView for why that xMax
// must be the current live progress, not the eventual horizon).
//
// Reconnects with exponential backoff on an unexpected close (dev-server
// HMR reload, network blip, a backgrounded tab's browser suspending an idle
// socket) so the UI doesn't freeze on a stale status forever -- previously
// the only thing this file did on close was set connected=false and give up,
// which is exactly what made a warmup that outlived a dropped connection
// look like a permanent hang. Since the server always resends the full
// backlog on a fresh connection, records are reset at the start of every
// (re)connect attempt, not just the first, so a reconnect can't double-count
// ticks already received.

import { useEffect, useRef, useState } from "react";
import { sessionStreamUrl } from "../api/client";
import type { SessionStatus, StreamMessage, TickRecord } from "../types";

export interface SessionSocketState {
  recordsByPolicy: Record<string, TickRecord[]>;
  status: SessionStatus | null;
  connected: boolean;
}

const TERMINAL_STATUSES: SessionStatus[] = ["completed", "stopped", "error"];
const MAX_RECONNECT_DELAY_MS = 10_000;

export function useSessionSocket(sessionId: string | null): SessionSocketState {
  const [recordsByPolicy, setRecordsByPolicy] = useState<Record<string, TickRecord[]>>({});
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [connected, setConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const terminalRef = useRef(false);

  useEffect(() => {
    setRecordsByPolicy({});
    setStatus(null);
    setConnected(false);

    if (!sessionId) return;

    let cancelled = false;
    terminalRef.current = false;
    reconnectAttemptRef.current = 0;

    const connect = () => {
      if (cancelled) return;

      // Reset per-connection state: the server always resends the full
      // backlog from scratch on a new socket, so anything accumulated from a
      // prior connection would otherwise be duplicated.
      setRecordsByPolicy({});
      setStatus(null);

      const ws = new WebSocket(sessionStreamUrl(sessionId));
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectAttemptRef.current = 0;
      };

      ws.onmessage = (evt) => {
        const msg: StreamMessage = JSON.parse(evt.data);
        if (msg.type === "tick") {
          setRecordsByPolicy((prev) => {
            const existing = prev[msg.policy] ?? [];
            return { ...prev, [msg.policy]: [...existing, msg.data] };
          });
        } else if (msg.type === "status") {
          setStatus(msg.status);
          if (TERMINAL_STATUSES.includes(msg.status)) {
            terminalRef.current = true;
          }
        }
      };

      ws.onerror = () => {
        setConnected(false);
      };

      ws.onclose = () => {
        setConnected(false);
        if (cancelled || terminalRef.current) return;

        const attempt = reconnectAttemptRef.current + 1;
        reconnectAttemptRef.current = attempt;
        const delay = Math.min(1000 * 2 ** (attempt - 1), MAX_RECONNECT_DELAY_MS);
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  return { recordsByPolicy, status, connected };
}
