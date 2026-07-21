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


def make_env(date_str: str):
    def _init():
        # Monitor gives EvalCallback proper episode reward/length stats.
        return Monitor(SORGymEnv(date_str=date_str))

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-07-11")
    parser.add_argument("--timesteps", type=int, default=20000)
    args = parser.parse_args()

    # Prices (~64k) and the bps rewards both need normalizing for PPO to learn.
    train_env = VecNormalize(
        DummyVecEnv([make_env(args.date)]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    # Eval env normalizes obs with the training stats (synced by EvalCallback)
    # but reports raw rewards so the numbers are interpretable.
    eval_env = VecNormalize(
        DummyVecEnv([make_env(args.date)]),
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
