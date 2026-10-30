

"""Train PPO with randomized starts and world-relative PID targets."""

import argparse
import json
from pathlib import Path

import aerial_sim
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from aerial_rl.environment import AerialDroneEnv


class CheckpointCallback(BaseCallback):
    """Save intermediate policies at fixed training intervals."""

    def __init__(self, output_dir: Path, interval: int):
        super().__init__()
        self.output_dir = output_dir
        self.interval = interval
        self.next_checkpoint = interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            path = self.output_dir / (
                f"ppo_hover_{self.num_timesteps}_steps"
            )
            self.model.save(str(path))
            print(f"\nCheckpoint saved: {path}.zip")
            self.next_checkpoint += self.interval

        return True


def make_environment(episode_duration: float):
    def initialize():
        env = AerialDroneEnv(
            maximum_episode_time=episode_duration,
            randomize_initial_state=True,
            maximum_position_offset=(0.5, 0.5, 0.2),
            maximum_velocity=(0.2, 0.2, 0.0),
            pid_target_reference=(
                aerial_sim.PIDTargetReference.WORLD_RELATIVE
            ),
        )
        return Monitor(env)

    return initialize


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "training_results/ppo_hover_randomized"
        ),
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    if args.episode_duration <= 0:
        parser.error("--episode-duration must be positive")

    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    configuration = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "episode_duration": args.episode_duration,
        "seed": args.seed,
        "randomize_initial_state": True,
        "maximum_position_offset": [0.5, 0.5, 0.2],
        "maximum_velocity": [0.2, 0.2, 0.0],
        "pid_target_reference": "WORLD_RELATIVE",
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 5,
        "clip_range": 0.1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }

    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=4) + "\n"
    )

    env = DummyVecEnv([
        make_environment(args.episode_duration)
    ])

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.0,
            verbose=1,
            device="cpu",
            seed=args.seed,
            tensorboard_log=str(args.output / "tensorboard"),
        )

        callback = CheckpointCallback(
            output_dir=args.output,
            interval=args.checkpoint_interval,
        )

        print("\nProject Aerial RL — Randomized PPO Training")
        print("=" * 55)
        print(f"Timesteps:       {args.timesteps:,}")
        print(f"Episode length:  {args.episode_duration:.1f} s")
        print("Initial states:  Randomized")
        print("PID reference:   World-relative")
        print(f"Output:          {args.output}")
        print("=" * 55)

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO_randomized",
            progress_bar=True,
        )

        final_path = args.output / "ppo_hover_final"
        model.save(str(final_path))

        print(f"\nFinal model saved: {final_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
