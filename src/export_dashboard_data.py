# export_dashboard_data.py
#
# Runs the trained PPO agent and rule-based baselines on the held-out day and
# writes a single JSON file for the visualization dashboard: a tick-by-tick
# trace of one representative episode per policy (shared seed, so they're
# directly comparable) plus a multi-window scoreboard (mean/std bps) matching
# eval_policies.py's methodology.

import json
from pathlib import Path
from typing import Dict, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sor_env import SORGymEnv
from trace_schema import build_record
from eval_policies import (
    dump_policy,
    baseline_twap_policy,
    twap_policy,
    no_trade_policy,
    _stats,
    MODEL_DIR,
)

OUT_PATH = Path(__file__).resolve().parent.parent / "dashboard_data.json"
HELD_OUT_DATE = "2026-07-21"
TRACE_SEED = 2000
SCOREBOARD_SEEDS = list(range(1000, 1030))

BASELINES = [
    ("Dump Everything", dump_policy),
    ("Baseline TWAP", baseline_twap_policy),
    ("Catch-up TWAP", twap_policy),
    ("No Trade", no_trade_policy),
]


def trace_baseline(name: str, policy_fn, seed: int) -> Dict:
    env = SORGymEnv(HELD_OUT_DATE, randomize_start=True)
    obs, _ = env.reset(seed=seed)
    trace: List[Dict] = []
    cum_reward = 0.0
    done = False
    step = 0
    while not done:
        action = policy_fn(obs, env)
        obs, reward, done, _, info = env.step(action)
        cum_reward += reward
        trace.append(build_record(env.engine, step, action, reward, cum_reward, info))
        step += 1
    env.close()
    return {"name": name, "trace": trace, "total_reward": cum_reward}


def trace_trained(seed: int) -> Dict:
    # We step raw_env directly rather than through venv.step(): DummyVecEnv
    # auto-resets the underlying env the instant done=True, so reading
    # raw_env.engine right after that call would see a freshly reset episode
    # (new random day/window, inventory back to 100%) instead of the true
    # final tick. Stepping raw_env ourselves avoids that entirely; the
    # VecNormalize wrapper is only used for its fitted normalization stats.
    raw_env = SORGymEnv(HELD_OUT_DATE, randomize_start=True)
    norm_wrapper = VecNormalize.load(
        str(MODEL_DIR / "vecnormalize.pkl"),
        DummyVecEnv([lambda: SORGymEnv(HELD_OUT_DATE, randomize_start=True)]),
    )

    model = PPO.load(str(MODEL_DIR / "ppo_sor_final.zip"))

    raw_env.engine.seed(seed)
    obs, _ = raw_env.reset()
    trace: List[Dict] = []
    cum_reward = 0.0
    done = False
    step = 0
    while not done:
        norm_obs = norm_wrapper.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, done, _, info = raw_env.step(int(action[0]))
        cum_reward += reward
        trace.append(build_record(raw_env.engine, step, int(action[0]), reward, cum_reward, info))
        step += 1
    raw_env.close()
    return {"name": "Trained PPO", "trace": trace, "total_reward": cum_reward}


def build_scoreboard() -> List[Dict]:
    from eval_policies import run_policy, run_trained, load_trained

    rows = []
    for name, policy_fn in BASELINES:
        env = SORGymEnv(HELD_OUT_DATE, randomize_start=True)
        rewards = [run_policy(env, policy_fn, seed=s)["total_reward"] for s in SCOREBOARD_SEEDS]
        env.close()
        rows.append({"name": name, **_stats(rewards)})

    model_path = MODEL_DIR / "ppo_sor_final.zip"
    vecnorm_path = MODEL_DIR / "vecnormalize.pkl"
    model, venv = load_trained(model_path, vecnorm_path, HELD_OUT_DATE)
    rewards = [run_trained(model, venv, s)["total_reward"] for s in SCOREBOARD_SEEDS]
    rows.append({"name": "Trained PPO", **_stats(rewards)})

    return rows


def main():
    print(f"Tracing one representative episode per policy (seed={TRACE_SEED}, date={HELD_OUT_DATE})...")
    traces = [trace_baseline(name, fn, TRACE_SEED) for name, fn in BASELINES]
    traces.append(trace_trained(TRACE_SEED))

    print(f"Building scoreboard over {len(SCOREBOARD_SEEDS)} windows...")
    scoreboard = build_scoreboard()

    payload = {
        "held_out_date": HELD_OUT_DATE,
        "trace_seed": TRACE_SEED,
        "scoreboard": scoreboard,
        "traces": traces,
    }
    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote dashboard data to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
