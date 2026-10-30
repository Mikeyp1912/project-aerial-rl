

"""Train the first PPO flight controller for Project Aerial RL."""

import argparse
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from aerial_rl.environment import AerialDroneEnv


def make_environment(
    seed: int,
    duration: float,
    monitor_path: Path | None = None,
):
    env = AerialDroneEnv(
        time_step=1.0 / 120.0,
        maximum_episode_time=duration,
    )

    env.action_space.seed(seed)

    if monitor_path is not None:
        monitor_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        env = Monitor(
            env,
            filename=str(monitor_path),
        )
    else:
        env = Monitor(env)

    return env


def evaluate(
    model: PPO,
    duration: float,
    seed: int,
    episodes: int = 10,
):
    env = make_environment(
        seed=seed,
        duration=duration,
    )

    rewards = []
    lengths = []
    failures = 0

    try:
        for episode in range(episodes):
            observation, _ = env.reset(
                seed=seed + episode
            )

            total_reward = 0.0
            steps = 0

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    if terminated:
                        failures += 1
                    break

            rewards.append(total_reward)
            lengths.append(steps)

    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "termination_rate": failures / episodes,
    }


def print_evaluation(name: str, results: dict):
    print(f"\n{name}")
    print(
        f"  Mean reward:       "
        f"{results['mean_reward']:.6f}"
    )
    print(
        f"  Reward std:        "
        f"{results['std_reward']:.6f}"
    )
    print(
        f"  Mean length:       "
        f"{results['mean_length']:.1f} steps"
    )
    print(
        f"  Termination rate:  "
        f"{results['termination_rate']:.2%}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    if args.duration <= 0:
        parser.error("--duration must be positive")

    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")

    output_dir = Path("training_results/ppo_hover_conservative")
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Project Aerial RL — PPO Training")
    print("=" * 45)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seed:      {args.seed}")
    print(f"Duration:  {args.duration} s")
    print(f"Device:    {args.device}")

    # Validate the raw Gymnasium environment.
    validation_env = AerialDroneEnv(
        maximum_episode_time=args.duration,
    )

    check_env(
        validation_env,
        warn=True,
        skip_render_check=True,
    )

    validation_env.close()

    training_env = make_environment(
        seed=args.seed,
        duration=args.duration,
        monitor_path=output_dir / "training_monitor",
    )

    evaluation_env = make_environment(
        seed=args.seed + 10_000,
        duration=args.duration,
        monitor_path=output_dir / "evaluation_monitor",
    )

    try:
        model = PPO(
            policy="MlpPolicy",
            env=training_env,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.0,
            policy_kwargs={
                "net_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64],
                },
            },
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
        )

        # Same evaluation seeds before and after training.
        evaluation_seed = args.seed + 20_000

        baseline = evaluate(
            model,
            duration=args.duration,
            seed=evaluation_seed,
            episodes=args.eval_episodes,
        )

        print_evaluation(
            "Untrained PPO baseline",
            baseline,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=20_000,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_hover",
        )

        eval_callback = EvalCallback(
            evaluation_env,
            best_model_save_path=str(
                output_dir / "best_model"
            ),
            log_path=str(output_dir / "evaluations"),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
        )

        print("\nStarting PPO training...\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=[
                checkpoint_callback,
                eval_callback,
            ],
            progress_bar=True,
            tb_log_name="PPO_1",
        )

        model_path = output_dir / "ppo_hover_final"

        model.save(str(model_path))

        print(f"\nSaved model: {model_path}.zip")

        final_results = evaluate(
            model,
            duration=args.duration,
            seed=evaluation_seed,
            episodes=args.eval_episodes,
        )

        print_evaluation(
            "Trained PPO policy",
            final_results,
        )

        improvement = (
            final_results["mean_reward"]
            - baseline["mean_reward"]
        )

        print("\nTraining summary")
        print(f"  Reward improvement: {improvement:+.6f}")
        print(f"  Model: {model_path}.zip")

    finally:
        training_env.close()
        evaluation_env.close()


if __name__ == "__main__":
    main()
