# api/server.py
#
# FastAPI backend for the live paper-trading dashboard. Manages at most one
# PaperTradingSession at a time (one simulated position per live order book;
# each session already runs all five policies internally, see
# paper_trading_session.py), streams new ticks over a WebSocket, and serves
# completed sessions back from paper_sessions/ on disk.
#
# No order-placement code and no exchange API keys anywhere here -- every
# endpoint only starts/stops/reads a *simulated* run.
#
# Run from src/ with: python -m uvicorn api.server:app --reload

import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from paper_trading_session import PaperTradingSession, SESSIONS_DIR, POLICY_NAMES

app = FastAPI(title="Smart Order Router -- Paper Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only; the Vite dev server runs on a different port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level state is enough here: only one session runs at a time (see
# PaperTradingSession's docstring for why), and JSONL/meta files on disk are
# the durable store -- no database needed for a single-process local demo.
_active: Optional[PaperTradingSession] = None
_history: Dict[str, PaperTradingSession] = {}

# The current model was retrained with domain randomization across these
# exact ranges (see src/train_rl.py) -- validated via a 30-window eval sweep
# across both axes before deployment. Bounds are enforced here, not just in
# the UI, so a value outside the validated distribution can't be requested
# via any client (this is what fixed the earlier 100 BTC failure: the old
# model was only ever trained/tested at one fixed point, 25 BTC / 300 ticks).
QTY_MIN, QTY_MAX = 5.0, 100.0
HORIZON_MIN, HORIZON_MAX = 150, 450


class StartSessionRequest(BaseModel):
    total_target_qty: float = Field(default=25.0, ge=QTY_MIN, le=QTY_MAX)
    horizon_steps: int = Field(default=300, ge=HORIZON_MIN, le=HORIZON_MAX)


def _is_running(session: Optional[PaperTradingSession]) -> bool:
    return session is not None and session.status in ("starting", "warming_up", "running")


def _read_meta_files():
    if not SESSIONS_DIR.exists():
        return []
    metas = []
    for p in sorted(SESSIONS_DIR.glob("*.meta.json")):
        try:
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return metas


def _read_session_from_disk(session_id: str):
    """Returns (status, traces), or None if the session isn't on disk. Reads
    the real status from meta.json rather than assuming "completed" -- a
    stopped or errored session must still report itself as such when read
    back by a fresh server process with no memory of running it."""
    meta_path = SESSIONS_DIR / f"{session_id}.meta.json"
    session_dir = SESSIONS_DIR / session_id
    if not meta_path.exists() or not session_dir.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status = meta.get("status", "completed")
    except (json.JSONDecodeError, OSError):
        status = "completed"

    traces = []
    for name in POLICY_NAMES:
        safe_name = name.lower().replace(" ", "_")
        trace_path = session_dir / f"{safe_name}.jsonl"
        records = []
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        total_reward = records[-1]["cum_reward"] if records else 0.0
        traces.append({"name": name, "trace": records, "total_reward": total_reward})
    return status, traces


@app.post("/api/sessions")
async def start_session(body: StartSessionRequest = StartSessionRequest()):
    global _active
    if _is_running(_active):
        raise HTTPException(status_code=409, detail=f"Session {_active.session_id} is already {_active.status}")

    session = PaperTradingSession(total_target_qty=body.total_target_qty, horizon_steps=body.horizon_steps)
    session.start()
    _active = session
    _history[session.session_id] = session
    return {"session_id": session.session_id, "status": session.status}


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    session = _history.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id (or it finished and left memory)")
    session.request_stop()
    return {"session_id": session_id, "status": session.status}


@app.get("/api/sessions")
async def list_sessions():
    seen = {}
    for meta in _read_meta_files():
        seen[meta["session_id"]] = meta
    for session_id, session in _history.items():
        seen[session_id] = session.to_summary()  # the in-memory copy is fresher than any meta.json on disk
    return sorted(seen.values(), key=lambda m: m.get("start_time") or "", reverse=True)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = _history.get(session_id)
    if session is not None:
        return {"session_id": session_id, "status": session.status, "traces": session.to_traces()}

    result = _read_session_from_disk(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    status, traces = result
    return {"session_id": session_id, "status": status, "traces": traces}


@app.websocket("/api/sessions/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = _history.get(session_id)
    if session is None:
        await websocket.send_json({"type": "error", "detail": "Unknown session_id"})
        await websocket.close()
        return

    # Backlog: everything produced so far, so a client connecting mid-session
    # (or to one that already finished) isn't missing the start.
    for name in POLICY_NAMES:
        for record in session.records.get(name, []):
            await websocket.send_json({"type": "tick", "policy": name, "data": record})

    # Always send the current status immediately, whether active or not --
    # a client connecting mid-warmup or mid-run would otherwise not learn the
    # real status until the *next* transition (PaperTradingSession.run() only
    # publishes on status changes, which may have already happened before
    # this client subscribed).
    await websocket.send_json({"type": "status", "status": session.status})

    if not _is_running(session):
        await websocket.close()
        return

    queue = session.subscribe()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("type") == "status":
                break
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)


@app.get("/api/health")
async def health():
    if _active is None:
        return {"status": "idle", "order_book_status": None, "warmup_progress": None}

    ob = _active.order_book
    warmup_progress = None
    if ob.status in ("connecting", "warming_up"):
        warmup_progress = {"n_events": ob.n_events, "warmup_events": ob.warmup_events}

    return {"status": _active.status, "order_book_status": ob.status, "warmup_progress": warmup_progress}
