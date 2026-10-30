

"""Evaluate PPO checkpoints using identical randomized hover episodes."""

import argparse
import csv
import re
from pathlib import Path

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def make_environment(duration):
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
    )


def evaluate(model, seeds, duration):
    env = make_environment(duration)

    rewards = []
    xy_errors = []
    altitude_errors = []
    speeds = []
    recovery = []
    failures = 0
    actions = []

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)

            initial_xy_error = np.linalg.norm(
                info["initial_position"][:2]
            )

            total_reward = 0.0

            while True:
                if model is None:
                    action = np.zeros(4, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                action = np.asarray(
                    action, dtype=np.float32
                ).reshape(4)

                actions.append(action.copy())

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                if terminated or truncated:
                    failures += int(terminated)
                    break

            position = np.asarray(
                env.simulator.get_position(),
                dtype=np.float64,
            )

            velocity = np.asarray(
                env.simulator.get_velocity(),
                dtype=np.float64,
            )

            final_xy_error = np.linalg.norm(position[:2])

            rewards.append(total_reward)
            xy_errors.append(final_xy_error)
            altitude_errors.append(abs(position[2] - 2.0))
            speeds.append(np.linalg.norm(velocity))
            recovery.append(initial_xy_error - final_xy_error)

    finally:
        env.close()

    actions = np.asarray(actions)

    return {
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "xy_error": float(np.mean(xy_errors)),
        "altitude_error": float(np.mean(altitude_errors)),
        "speed": float(np.mean(speeds)),
        "recovery": float(np.mean(recovery)),
        "failure_rate": failures / len(seeds),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
    }


def find_checkpoints(directory):
    checkpoints = []

    for path in directory.glob("ppo_hover_*_steps.zip"):
        match = re.fullmatch(
            r"ppo_hover_(\d+)_steps\.zip",
            path.name,
        )

        if match:
            checkpoints.append((int(match.group(1)), path))

    checkpoints.sort(key=lambda item: item[0])

    final = directory / "ppo_hover_final.zip"

    if final.is_file():
        checkpoints.append(("Final", final))

    return checkpoints


def print_results(results):
    print()
    print(
        f"{'Checkpoint':<15}"
        f"{'Reward':>11}"
        f"{'Std':>10}"
        f"{'XY (m)':>10}"
        f"{'Z (m)':>10}"
        f"{'Speed':>10}"
        f"{'Recovery':>11}"
        f"{'Fail %':>9}"
    )
    print("-" * 86)

    for row in results:
        print(
            f"{str(row['checkpoint']):<15}"
            f"{row['reward']:>11.2f}"
            f"{row['reward_std']:>10.2f}"
            f"{row['xy_error']:>10.3f}"
            f"{row['altitude_error']:>10.3f}"
            f"{row['speed']:>10.3f}"
            f"{row['recovery']:>11.3f}"
            f"{row['failure_rate'] * 100:>8.1f}%"
        )


def save_csv(results, path):
    fields = [
        "checkpoint",
        "reward",
        "reward_std",
        "xy_error",
        "altitude_error",
        "speed",
        "recovery",
        "failure_rate",
        "action_mean",
        "action_std",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("training_results/ppo_hover_scaled_50k"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=10000)

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if args.duration <= 0:
        parser.error("--duration must be positive")

    checkpoints = find_checkpoints(args.directory)

    if not checkpoints:
        parser.error(
            f"No PPO checkpoints found in {args.directory}"
        )

    seeds = list(
        range(args.seed, args.seed + args.episodes)
    )

    print("Project Aerial RL — PPO Checkpoint Evaluation")
    print("=" * 60)
    print(f"Directory: {args.directory}")
    print(f"Episodes:  {args.episodes}")
    print(f"Duration:  {args.duration:.1f} seconds")
    print(f"Seeds:     {seeds[0]}–{seeds[-1]}")
    print(f"Models:    {len(checkpoints)}")

    results = []

    baseline = evaluate(None, seeds, args.duration)
    baseline["checkpoint"] = "Zero action"
    results.append(baseline)

    for step, path in checkpoints:
        print(f"Evaluating {path.name}...", flush=True)

        model = PPO.load(str(path), device="cpu")

        result = evaluate(model, seeds, args.duration)
        result["checkpoint"] = step

        results.append(result)

    print_results(results)

    print("\nMean actions [X, Y, Z, Yaw]")
    print("-" * 60)

    for row in results:
        mean = np.asarray(row["action_mean"])
        std = np.asarray(row["action_std"])

        print(
            f"{str(row['checkpoint']):<15}"
            f"mean={np.array2string(mean, precision=4)}"
        )
        print(
            f"{'':<15}"
            f"std ={np.array2string(std, precision=4)}"
        )

    # Select the highest-reward trained checkpoint.
    # The zero-action baseline is intentionally excluded.
    trained_results = results[1:]

    best = max(
        trained_results,
        key=lambda row: row["reward"],
    )

    print("\nBest trained checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {best['checkpoint']}")
    print(f"Mean reward: {best['reward']:.2f}")
    print(f"Final XY error: {best['xy_error']:.3f} m")
    print(f"Final altitude error: {best['altitude_error']:.3f} m")

    if best["reward"] > baseline["reward"]:
        print("Result: PPO outperformed zero action.")
    else:
        print("Result: Zero action remains the better baseline.")

    output_path = args.directory / "checkpoint_evaluation.csv"

    save_csv(results, output_path)

    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
