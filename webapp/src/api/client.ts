// client.ts
//
// Thin wrappers over the FastAPI backend (src/api/server.py). No client-side
// state here -- each view fetches what it needs and useSessionSocket handles
// the one stateful piece (the live WS stream).

import type { HealthResponse, SessionDetail, SessionSummary } from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface StartSessionParams {
  // horizon_steps is deliberately not a field here: the backend's
  // StartSessionRequest no longer accepts it (see api/server.py) -- the
  // trained 300-tick pacing schedule always applies.
  total_target_qty?: number;
}

export function startSession(params: StartSessionParams = {}): Promise<{ session_id: string; status: string }> {
  return request("/api/sessions", { method: "POST", body: JSON.stringify(params) });
}

export function stopSession(sessionId: string): Promise<{ session_id: string; status: string }> {
  return request(`/api/sessions/${sessionId}/stop`, { method: "POST" });
}

export function listSessions(): Promise<SessionSummary[]> {
  return request("/api/sessions");
}

export function getSession(sessionId: string): Promise<SessionDetail> {
  return request(`/api/sessions/${sessionId}`);
}

export function getHealth(): Promise<HealthResponse> {
  return request("/api/health");
}

export function sessionStreamUrl(sessionId: string): string {
  const wsBase = BASE_URL.replace(/^http/, "ws");
  return `${wsBase}/api/sessions/${sessionId}/stream`;
}
