

"""Diagnose PPO actions and flight performance against PID."""

import argparse
import csv
from pathlib import Path

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def make_environment(observation_mode, wind_speed):
    return AerialDroneEnv(
        maximum_episode_time=5.0,
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
        horizontal_action_weight=0.05,
        altitude_action_weight=0.10,
        yaw_action_weight=0.02,
        observation_mode=observation_mode,
    )


def evaluate_policy(model, seeds, wind_speed):
    observation_mode = (
        aerial_sim.ObservationMode.BASE
        if model is None
        else aerial_sim.ObservationMode.HOVER_AUGMENTED
    )

    env = make_environment(observation_mode, wind_speed)

    actions = []
    rewards = []
    xy_errors = []
    altitude_errors = []
    episode_rewards = []
    failures = 0

    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0

            while True:
                if model is None:
                    action = np.zeros(4, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                action = np.clip(
                    np.asarray(action, dtype=np.float64),
                    -1.0,
                    1.0,
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                position = np.asarray(
                    env.simulator.get_position(),
                    dtype=np.float64,
                )

                actions.append(action.copy())
                rewards.append(float(reward))
                xy_errors.append(
                    float(np.linalg.norm(position[:2]))
                )
                altitude_errors.append(
                    float(abs(position[2] - 2.0))
                )

                total_reward += reward

                if terminated or truncated:
                    failures += int(terminated)
                    break

            episode_rewards.append(total_reward)

    finally:
        env.close()

    actions = np.asarray(actions, dtype=np.float64)

    # Actions are normalized to [-1, 1].
    # With maximum altitude offset = 1 m,
    # action[2] is also the commanded altitude offset in metres.
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "mean_xy_error": float(np.mean(xy_errors)),
        "mean_altitude_error": float(np.mean(altitude_errors)),
        "failure_rate": failures / len(seeds),
        "mean_action_x": float(np.mean(actions[:, 0])),
        "mean_action_y": float(np.mean(actions[:, 1])),
        "mean_action_z": float(np.mean(actions[:, 2])),
        "mean_action_yaw": float(np.mean(actions[:, 3])),
        "mean_abs_x": float(np.mean(np.abs(actions[:, 0]))),
        "mean_abs_y": float(np.mean(np.abs(actions[:, 1]))),
        "mean_abs_z": float(np.mean(np.abs(actions[:, 2]))),
        "mean_abs_yaw": float(np.mean(np.abs(actions[:, 3]))),
        "std_x": float(np.std(actions[:, 0])),
        "std_y": float(np.std(actions[:, 1])),
        "std_z": float(np.std(actions[:, 2])),
        "std_yaw": float(np.std(actions[:, 3])),
        "mean_action_norm": float(
            np.mean(np.linalg.norm(actions, axis=1))
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(
            "training_results/ppo_hover_wind_augmented_100k_v2"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--wind-speeds",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0],
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if any(
        not np.isfinite(speed) or speed < 0
        for speed in args.wind_speeds
    ):
        parser.error("Wind speeds must be finite and nonnegative")

    checkpoints = {
        "PID": None,
        "PPO 10k": args.directory / "ppo_hover_10000_steps.zip",
        "PPO 50k": args.directory / "ppo_hover_50000_steps.zip",
        "PPO 100k": args.directory / "ppo_hover_100000_steps.zip",
    }

    models = {}

    for name, path in checkpoints.items():
        if path is None:
            models[name] = None
            continue

        model = PPO.load(str(path), device="cpu")

        if model.observation_space.shape != (19,):
            raise ValueError(
                f"{name} expects observation shape "
                f"{model.observation_space.shape}, not (19,)"
            )

        models[name] = model

    seeds = list(range(args.seed, args.seed + args.episodes))
    rows = []

    print("\nProject Aerial RL — PPO Action Diagnostics")
    print("=" * 76)

    for wind_speed in args.wind_speeds:
        print(f"\nMaximum wind speed: {wind_speed:.1f} m/s")
        print("-" * 76)

        for name, model in models.items():
            result = evaluate_policy(
                model,
                seeds,
                wind_speed,
            )

            rows.append({
                "policy": name,
                "maximum_wind_speed": wind_speed,
                **result,
            })

            print(
                f"{name:<10} "
                f"Reward={result['mean_reward']:>7.2f} | "
                f"XY={result['mean_xy_error']:.3f} m | "
                f"Z={result['mean_altitude_error']:.3f} m | "
                f"|action|={result['mean_action_norm']:.4f}"
            )

            print(
                " " * 11
                + "Mean action [X,Y,Z,Yaw]: "
                + ", ".join(
                    f"{result[f'mean_action_{axis}']:+.4f}"
                    for axis in ("x", "y", "z", "yaw")
                )
            )

            print(
                " " * 11
                + "Mean |action| per axis: "
                + ", ".join(
                    f"{result[f'mean_abs_{axis}']:.4f}"
                    for axis in ("x", "y", "z", "yaw")
                )
            )

    output = args.directory / "action_diagnostics.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved diagnostics: {output}")


if __name__ == "__main__":
    main()
