# paper_trading_session.py
#
# Orchestrates one live paper-trading run: five policies (four rule-based
# ones from eval_policies.py plus the trained PPO agent) sharing a single
# LiveOrderBook, each with its own independent inventory/cash/perm_impact
# state. This supersedes live_paper_trade_cli.py's inline script with a
# reusable, server-drivable class -- same multi-policy-sharing-one-tick
# design already verified there against real live Binance data (300-tick
# horizon: PPO -12.56 bps, Baseline TWAP -16.82 bps, matching backtest).

import asyncio
import json
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from live_order_book import LiveOrderBook
from sor_live_env import SORLiveEnv
from sor_env import SORGymEnv
from trace_schema import build_record
from eval_policies import (
    dump_policy,
    baseline_twap_policy,
    twap_policy,
    no_trade_policy,
    MODEL_DIR,
)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "paper_sessions"

BASELINES = [
    ("Dump Everything", dump_policy),
    ("Baseline TWAP", baseline_twap_policy),
    ("Catch-up TWAP", twap_policy),
    ("No Trade", no_trade_policy),
]
PPO_NAME = "Trained PPO"
POLICY_NAMES = [name for name, _ in BASELINES] + [PPO_NAME]


class PaperTradingSession:
    """status: starting -> warming_up -> running -> completed | stopped | error

    request_stop() only flips a flag checked between ticks -- the in-flight
    tick always finishes and is logged for every policy first, consistent
    with "simulated fills only, nothing torn down mid-fill."
    """

    def __init__(
        self,
        total_target_qty: float = 25.0,
        horizon_steps: int = 300,
        model_path: Path = MODEL_DIR / "ppo_sor_final.zip",
        vecnorm_path: Path = MODEL_DIR / "vecnormalize.pkl",
    ):
        self.session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.total_target_qty = total_target_qty
        self.horizon_steps = horizon_steps
        self.model_path = Path(model_path)
        self.vecnorm_path = Path(vecnorm_path)

        self.order_book = LiveOrderBook()
        self.envs: Dict[str, SORLiveEnv] = {}
        self.cum_reward: Dict[str, float] = {name: 0.0 for name in POLICY_NAMES}
        self.step: Dict[str, int] = {name: 0 for name in POLICY_NAMES}
        self.done: Dict[str, bool] = {name: False for name in POLICY_NAMES}
        self.records: Dict[str, List[dict]] = {name: [] for name in POLICY_NAMES}

        self.status = "starting"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self._stop_requested = False
        self._subscribers: List[asyncio.Queue] = []
        self._task: Optional[asyncio.Task] = None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _publish_tick(self, policy_name: str, record: dict):
        payload = {"type": "tick", "policy": policy_name, "data": record}
        for q in self._subscribers:
            q.put_nowait(payload)

    def _publish_status(self):
        payload = {"type": "status", "status": self.status}
        for q in self._subscribers:
            q.put_nowait(payload)

    def request_stop(self):
        self._stop_requested = True

    def start(self) -> asyncio.Task:
        """Launch run() as a background asyncio task; returns immediately."""
        self._task = asyncio.create_task(self.run())
        return self._task

    async def run(self):
        try:
            self.start_time = datetime.now(UTC).isoformat()
            await self.order_book.connect()
            self.status = "warming_up"
            self._publish_status()
            await self.order_book.wait_until_live()
            self.status = "running"
            self._publish_status()

            self.envs = {
                name: SORLiveEnv(self.order_book, total_target_qty=self.total_target_qty, horizon_steps=self.horizon_steps)
                for name, _ in BASELINES
            }
            self.envs[PPO_NAME] = SORLiveEnv(
                self.order_book, total_target_qty=self.total_target_qty, horizon_steps=self.horizon_steps
            )

            # VecNormalize is only used here for its fitted obs-normalization
            # stats; the date passed to this throwaway DummyVecEnv is unused.
            norm_wrapper = VecNormalize.load(
                str(self.vecnorm_path),
                DummyVecEnv([lambda: SORGymEnv("2026-07-11")]),
            )
            model = PPO.load(str(self.model_path))

            tick0 = await self.order_book.next_tick()
            obs = {name: await env.reset(tick=tick0) for name, env in self.envs.items()}

            while not all(self.done.values()) and not self._stop_requested:
                tick = await self.order_book.next_tick()

                for name, policy_fn in BASELINES:
                    if self.done[name]:
                        continue
                    action = policy_fn(obs[name], self.envs[name])
                    obs[name], reward, d, info = await self.envs[name].step(action, tick=tick)
                    self._record_tick(name, action, reward, info)
                    self.done[name] = d

                if not self.done[PPO_NAME]:
                    norm_obs = norm_wrapper.normalize_obs(obs[PPO_NAME].reshape(1, -1))
                    # model.predict() is a blocking torch/numpy call; run it off
                    # the event loop so it never stalls other sessions/clients.
                    action, _ = await asyncio.to_thread(model.predict, norm_obs, deterministic=True)
                    action = int(action[0])
                    obs[PPO_NAME], reward, d, info = await self.envs[PPO_NAME].step(action, tick=tick)
                    self._record_tick(PPO_NAME, action, reward, info)
                    self.done[PPO_NAME] = d

            self.status = "stopped" if self._stop_requested else "completed"
        except Exception:
            self.status = "error"
            raise
        finally:
            self.end_time = datetime.now(UTC).isoformat()
            self._write_logs()
            self._publish_status()
            await self.order_book.close()

    def _record_tick(self, name: str, action: int, reward: float, info: dict):
        self.cum_reward[name] += reward
        record = build_record(self.envs[name].engine, self.step[name], action, reward, self.cum_reward[name], info)
        self.records[name].append(record)
        self.step[name] += 1
        self._publish_tick(name, record)

    def _write_logs(self):
        session_dir = SESSIONS_DIR / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        for name, records in self.records.items():
            safe_name = name.lower().replace(" ", "_")
            out_path = session_dir / f"{safe_name}.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")

        meta = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "total_target_qty": self.total_target_qty,
            "horizon_steps": self.horizon_steps,
            "model_path": str(self.model_path),
            "vecnorm_path": str(self.vecnorm_path),
            "policies": {
                name: {"steps": self.step.get(name, 0), "total_reward": self.cum_reward.get(name, 0.0)}
                for name in POLICY_NAMES
            },
        }
        meta_path = SESSIONS_DIR / f"{self.session_id}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def to_summary(self) -> dict:
        """Lightweight summary for the GET /api/sessions listing. Includes
        horizon_steps/total_target_qty so the history view can distinguish a
        short dev/verification run from a genuine full-horizon session --
        without it, a mismatched-horizon run's much worse numbers read as an
        unexplained model regression rather than an apples-to-oranges episode."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_target_qty": self.total_target_qty,
            "horizon_steps": self.horizon_steps,
            "policies": {
                name: {"steps": self.step.get(name, 0), "total_reward": self.cum_reward.get(name, 0.0)}
                for name in POLICY_NAMES
            },
        }

    def to_traces(self) -> List[dict]:
        """Full detail in the {name, trace, total_reward} shape dashboard_data.json already uses."""
        return [
            {"name": name, "trace": self.records.get(name, []), "total_reward": self.cum_reward.get(name, 0.0)}
            for name in POLICY_NAMES
        ]
