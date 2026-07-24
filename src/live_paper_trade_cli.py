# live_paper_trade_cli.py
#
# Phase 2 manual verification: runs all five policies (the four rule-based
# ones from eval_policies.py plus the trained PPO agent) concurrently against
# ONE live order book. Each policy gets its own SORLiveEnv/SmartOrderRouterLive
# with its own inventory/cash/perm_impact state, but every round fetches a
# single tick from the shared LiveOrderBook and hands that same tick to every
# still-running policy -- the live counterpart of how export_dashboard_data.py
# runs every policy against the same seeded historical window, just
# concurrent instead of sequential. No server, no frontend -- just proving
# the multi-policy live loop behaves sensibly end-to-end.

import argparse
import asyncio
import json
from datetime import datetime, UTC
from pathlib import Path

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


async def main(total_target_qty: float, horizon_steps: int):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = "smoke_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    order_book = LiveOrderBook()
    print("connecting...")
    await order_book.connect()
    print(f"status: {order_book.status} -- warming up (~{order_book.warmup_events}s)...")
    await order_book.wait_until_live()
    print("status:", order_book.status)

    envs = {
        name: SORLiveEnv(order_book, total_target_qty=total_target_qty, horizon_steps=horizon_steps)
        for name, _ in BASELINES
    }
    envs[PPO_NAME] = SORLiveEnv(order_book, total_target_qty=total_target_qty, horizon_steps=horizon_steps)

    # VecNormalize is only used here for its fitted obs-normalization stats;
    # the date passed to this throwaway DummyVecEnv is irrelevant.
    norm_wrapper = VecNormalize.load(
        str(MODEL_DIR / "vecnormalize.pkl"),
        DummyVecEnv([lambda: SORGymEnv("2026-07-11")]),
    )
    model = PPO.load(str(MODEL_DIR / "ppo_sor_final.zip"))

    # Prime every engine on the same first (arrival) tick.
    tick0 = await order_book.next_tick()
    obs = {name: await env.reset(tick=tick0) for name, env in envs.items()}

    cum_reward = {name: 0.0 for name in envs}
    step = {name: 0 for name in envs}
    done = {name: False for name in envs}
    traces = {name: [] for name in envs}

    print(f"Running {len(envs)} policies for up to {horizon_steps} ticks...\n")
    round_idx = 0
    while not all(done.values()):
        tick = await order_book.next_tick()
        round_idx += 1

        for name, policy_fn in BASELINES:
            if done[name]:
                continue
            action = policy_fn(obs[name], envs[name])
            obs[name], reward, d, info = await envs[name].step(action, tick=tick)
            cum_reward[name] += reward
            traces[name].append(build_record(envs[name].engine, step[name], action, reward, cum_reward[name], info))
            step[name] += 1
            done[name] = d

        if not done[PPO_NAME]:
            norm_obs = norm_wrapper.normalize_obs(obs[PPO_NAME].reshape(1, -1))
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(action[0])
            obs[PPO_NAME], reward, d, info = await envs[PPO_NAME].step(action, tick=tick)
            cum_reward[PPO_NAME] += reward
            traces[PPO_NAME].append(build_record(envs[PPO_NAME].engine, step[PPO_NAME], action, reward, cum_reward[PPO_NAME], info))
            step[PPO_NAME] += 1
            done[PPO_NAME] = d

        if round_idx % 10 == 0 or all(done.values()):
            summary = " | ".join(
                f"{n}: inv={float(envs[n].engine.inventory / envs[n].engine.total_target_qty):.0%} "
                f"pnl={cum_reward[n]:.2f}bps"
                for n in envs
            )
            print(f"[{round_idx:>3}] {tick.timestamp}: {summary}")

    print("\nFinal results:")
    for name in envs:
        print(f"  {name:<20} steps={step[name]:<4} total_reward_bps={cum_reward[name]:.2f}")

    for name, trace in traces.items():
        safe_name = name.lower().replace(" ", "_")
        out_path = SESSIONS_DIR / f"{session_id}_{safe_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in trace:
                fh.write(json.dumps(rec) + "\n")
    print(f"\nWrote per-policy traces to {SESSIONS_DIR}/{session_id}_*.jsonl")

    await order_book.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-qty", type=float, default=25.0)
    p.add_argument("--horizon-steps", type=int, default=300)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.target_qty, args.horizon_steps))
