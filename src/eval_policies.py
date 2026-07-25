# eval_policies.py

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sor_env import SORGymEnv

MODEL_DIR = Path(__file__).resolve().parent / "models"


def dump_policy(_: Dict, env: SORGymEnv) -> int:
    """Always execute the largest available sell slice."""
    return env.action_space.n - 1


def baseline_twap_policy(obs: Dict, env: SORGymEnv) -> int:
    """Sell an even slice each step so the position is liquidated by the horizon.

    Desired per-step size = remaining inventory / remaining steps, mapped to the
    nearest available action fraction (of the total target).
    """
    fractions = env.engine.action_fractions
    total_target = float(env.engine.total_target_qty)

    remaining_ratio = obs[env.feature_names.index("remaining_inventory_ratio")]
    remaining_time = obs[env.feature_names.index("remaining_time")]
    remaining_steps = max(remaining_time * env.engine.max_steps, 1.0)

    desired_frac = (remaining_ratio * total_target) / remaining_steps / total_target
    if remaining_time <= 0.0:
        return len(fractions) - 1

    return min(range(len(fractions)), key=lambda i: abs(fractions[i] - desired_frac))


def twap_policy(obs: Dict, env: SORGymEnv) -> int:
    """Catch-up TWAP: sell a larger slice when behind schedule, small otherwise."""
    twap_diff = obs[env.feature_names.index("twap_diff")]
    remaining_time = obs[env.feature_names.index("remaining_time")]
    fractions = env.engine.action_fractions

    if remaining_time <= 0.0:
        return len(fractions) - 1

    if twap_diff < 0:
        desired_frac = min(0.10, max(0.005, -twap_diff / max(remaining_time, 0.05)))
    else:
        desired_frac = 0.005

    return min(range(len(fractions)), key=lambda i: abs(fractions[i] - desired_frac))


def no_trade_policy(_: Dict, env: SORGymEnv) -> int:
    """Never trade. Shows the cost of missing the schedule entirely."""
    return 0


def run_policy(env: SORGymEnv, policy_fn: Callable[[Dict, SORGymEnv], int],
               seed: Optional[int] = None, max_steps: int = 100000) -> Dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = policy_fn(obs, env)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

    return {
        "steps": steps,
        "total_reward": total_reward,
        "remaining_inventory": obs[env.feature_names.index("remaining_inventory")],
    }


def load_trained(model_path: Path, vecnorm_path: Optional[Path], date_str: str, **engine_kwargs):
    """Load a PPO model wrapped in a reusable VecNormalize eval env."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: SORGymEnv(date_str=date_str, **engine_kwargs)])
    if vecnorm_path and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("  (no VecNormalize stats found; running on raw observations)")
    model = PPO.load(str(model_path))
    return model, venv


def run_trained(model, venv, seed: int, max_steps: int = 100000) -> Dict:
    venv.seed(seed)
    obs = venv.reset()
    total_reward = 0.0
    steps = 0
    done = [False]
    while not done[0] and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = venv.step(action)
        total_reward += float(reward[0])  # raw reward, norm_reward disabled
        steps += 1
    return {"steps": steps, "total_reward": total_reward}


def _stats(values: List[float]) -> Dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {"mean": mean, "std": var ** 0.5, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", nargs="+", default=["2026-07-21"],
                        help="Date(s) to evaluate on; a random one is drawn per window if multiple.")
    parser.add_argument("--windows", type=int, default=30,
                        help="Number of random windows each policy is scored on.")
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--target-qty", type=float, default=25.0,
                        help="total_target_qty for this eval run -- spot-check the trained model at "
                             "different points (e.g. 10/25/50/75/100) to confirm it generalizes across scale.")
    parser.add_argument("--horizon-steps", type=int, default=300,
                        help="horizon_steps for this eval run -- spot-check across the trained range.")
    args = parser.parse_args()

    dates = args.date[0] if len(args.date) == 1 else args.date
    seeds = [args.seed0 + i for i in range(args.windows)]
    engine_kwargs = {"total_target_qty": args.target_qty, "horizon_steps": args.horizon_steps}

    baselines = [
        ("Dump Everything", dump_policy),
        ("Baseline TWAP", baseline_twap_policy),
        ("Catch-up TWAP", twap_policy),
        ("No Trade", no_trade_policy),
    ]

    print(f"Evaluating over {args.windows} matched random windows (date(s) {dates}, "
          f"target_qty={args.target_qty}, horizon_steps={args.horizon_steps})\n")

    rows = []
    for name, policy_fn in baselines:
        # randomize_start=True so the seed selects a distinct window; every
        # policy is scored on the identical set of seeded windows.
        env = SORGymEnv(dates, randomize_start=True, **engine_kwargs)
        rewards = [run_policy(env, policy_fn, seed=s)["total_reward"] for s in seeds]
        env.close()
        rows.append((name, _stats(rewards)))

    model_path = MODEL_DIR / "ppo_sor_final.zip"
    vecnorm_path = MODEL_DIR / "vecnormalize.pkl"
    if model_path.exists():
        model, venv = load_trained(model_path, vecnorm_path, dates, **engine_kwargs)
        rewards = [run_trained(model, venv, s)["total_reward"] for s in seeds]
        rows.append(("Trained PPO", _stats(rewards)))
    else:
        print(f"(no trained model at {model_path}; train_rl.py first to compare RL)\n")

    print(f"{'Policy':<20}{'mean bps':>12}{'std':>10}")
    print("-" * 42)
    for name, st in sorted(rows, key=lambda r: r[1]["mean"], reverse=True):
        print(f"{name:<20}{st['mean']:>12.2f}{st['std']:>10.2f}")
    print("\n(higher = better; bps of arrival notional, less negative = less slippage)")


if __name__ == "__main__":
    main()
