# train_rl.py

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sor_env import SORGymEnv


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECNORM_PATH = MODEL_DIR / "vecnormalize.pkl"


def make_env(dates, target_qty_range=None, horizon_range=None):
    def _init():
        # Monitor gives EvalCallback proper episode reward/length stats.
        return Monitor(SORGymEnv(
            date_str=dates,
            total_target_qty_range=target_qty_range,
            horizon_steps_range=horizon_range,
        ))

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dates", nargs="+", default=["2026-07-11", "2026-07-12", "2026-07-22"],
        help="Dates to train on; a random one is drawn each episode.",
    )
    parser.add_argument(
        "--held-out-date", default="2026-07-21",
        help="Date never trained on, used to pick the 'best' checkpoint on true generalization.",
    )
    parser.add_argument(
        "--target-qty-range", type=float, nargs=2, default=[25.0, 250.0], metavar=("MIN", "MAX"),
        help="total_target_qty is resampled uniformly from this range each episode. "
             "Pass the same value twice (e.g. 25 25) to disable randomization.",
    )
    parser.add_argument(
        "--horizon-range", type=int, nargs=2, default=[150, 450], metavar=("MIN", "MAX"),
        help="horizon_steps is resampled uniformly from this range each episode. "
             "Pass the same value twice (e.g. 300 300) to disable randomization.",
    )
    parser.add_argument("--timesteps", type=int, default=600000)
    args = parser.parse_args()

    qty_range = None if args.target_qty_range[0] == args.target_qty_range[1] else tuple(args.target_qty_range)
    horizon_range = None if args.horizon_range[0] == args.horizon_range[1] else tuple(args.horizon_range)

    # Prices (~64k) and the bps rewards both need normalizing for PPO to learn.
    train_env = VecNormalize(
        DummyVecEnv([make_env(args.train_dates, qty_range, horizon_range)]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    # Held-out day, never seen in training, so "best model" is selected on
    # genuine generalization instead of in-sample performance. Uses the same
    # randomized ranges as training -- otherwise "best" would be selected on
    # a single fixed point, which is exactly the narrow-distribution problem
    # this randomization exists to fix. Reports raw (non-normalized) rewards
    # so the numbers stay interpretable in bps.
    eval_env = VecNormalize(
        DummyVecEnv([make_env(args.held_out_date, qty_range, horizon_range)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=64,
        n_steps=2048,
        learning_rate=3e-4,
        ent_coef=0.01,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_sor",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(MODEL_DIR / "logs"),
        eval_freq=5000,
        # More episodes than SB3's default (5) to average out the extra
        # variance a randomized (qty, horizon) distribution adds to the
        # eval metric -- otherwise "best" checkpoint selection gets noisy.
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, eval_callback])

    model_path = MODEL_DIR / "ppo_sor_final.zip"
    model.save(str(model_path))
    # Persist the normalization stats so eval/inference can reproduce them.
    train_env.save(str(VECNORM_PATH))
    print(f"Saved final model to {model_path}")
    print(f"Saved VecNormalize stats to {VECNORM_PATH}")


if __name__ == "__main__":
    main()
