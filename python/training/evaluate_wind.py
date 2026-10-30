

"""Benchmark PID and PPO hover control under randomized wind."""

import argparse
import csv
from pathlib import Path

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def make_environment(duration, wind_speed):
    return AerialDroneEnv(
        maximum_episode_time=duration,
        randomize_initial_state=True,
        maximum_position_offset=(0.5, 0.5, 0.2),
        maximum_velocity=(0.2, 0.2, 0.0),
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        maximum_horizontal_offset=1.0,
        hover_centered_altitude=True,
        maximum_altitude_offset=1.0,
        randomize_wind=wind_speed > 0.0,
        maximum_wind_speed=wind_speed,
        linear_drag=(0.1, 0.1, 0.05),
    )


def evaluate(model, seeds, duration, wind_speed):
    env = make_environment(duration, wind_speed)

    rewards = []
    final_xy_errors = []
    mean_xy_errors = []
    maximum_xy_errors = []
    altitude_errors = []
    wind_speeds = []
    failures = 0

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)

            wind_speeds.append(
                np.linalg.norm(info["wind_velocity"][:2])
            )

            total_reward = 0.0
            episode_xy_errors = []

            while True:
                if model is None:
                    action = np.zeros(4, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                position = np.asarray(
                    env.simulator.get_position(),
                    dtype=np.float64,
                )

                episode_xy_errors.append(
                    np.linalg.norm(position[:2])
                )

                if terminated or truncated:
                    failures += int(terminated)
                    break

            rewards.append(total_reward)

            final_xy_errors.append(
                episode_xy_errors[-1]
            )

            mean_xy_errors.append(
                np.mean(episode_xy_errors)
            )

            maximum_xy_errors.append(
                np.max(episode_xy_errors)
            )

            altitude_errors.append(
                abs(position[2] - 2.0)
            )

    finally:
        env.close()

    return {
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "final_xy": float(np.mean(final_xy_errors)),
        "mean_xy": float(np.mean(mean_xy_errors)),
        "peak_xy": float(np.mean(maximum_xy_errors)),
        "altitude_error": float(np.mean(altitude_errors)),
        "wind_speed": float(np.mean(wind_speeds)),
        "failure_rate": failures / len(seeds),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--wind-speeds",
        type=float,
        nargs="+",
        default=[0.0, 2.5, 5.0, 10.0, 15.0],
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("training_results/ppo_hover_scaled_50k"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/wind_evaluation.csv"),
    )

    args = parser.parse_args()

    if args.episodes <= 0 or args.duration <= 0:
        parser.error("Episodes and duration must be positive")

    if any(not np.isfinite(s) or s < 0 for s in args.wind_speeds):
        parser.error("Wind speeds must be finite and nonnegative")

    seeds = list(range(args.seed, args.seed + args.episodes))

    policies = {
        "PID": None,
        "PPO 10k": PPO.load(
            str(args.directory / "ppo_hover_10000_steps.zip"),
            device="cpu",
        ),
        "PPO final": PPO.load(
            str(args.directory / "ppo_hover_final.zip"),
            device="cpu",
        ),
    }

    rows = []

    print("Project Aerial RL — Wind Disturbance Benchmark")
    print("=" * 78)

    for wind_speed in args.wind_speeds:
        for name, model in policies.items():
            result = evaluate(
                model,
                seeds,
                args.duration,
                wind_speed,
            )

            row = {
                "controller": name,
                "maximum_wind_speed": wind_speed,
                **result,
            }

            rows.append(row)

            print(
                f"Wind max={wind_speed:>5.1f} m/s | "
                f"{name:<10} | "
                f"Reward={result['reward']:>8.2f} | "
                f"Mean XY={result['mean_xy']:.3f} m | "
                f"Final XY={result['final_xy']:.3f} m | "
                f"Failures={100 * result['failure_rate']:.1f}%"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
