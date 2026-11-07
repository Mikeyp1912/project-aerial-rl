
"""Benchmark target tracking and sustained hover stabilization."""

import argparse
import csv
from pathlib import Path

import numpy as np

from compare_control_frequencies import (
    load_model,
    make_environment,
    target_aware_action,
)


POSITION_THRESHOLD = 0.2
SPEED_THRESHOLD = 0.2
HOLD_DURATION = 1.0
EPISODE_DURATION = 10.0


def evaluate_episode(name, model, mode, seed, action_repeat):
    env = make_environment(action_repeat)

    try:
        observation, info = env.reset(seed=int(seed))
        target = np.asarray(info["target_position"], dtype=float)
        reference = target_aware_action(target)

        total_reward = 0.0
        weighted_error = 0.0
        total_steps = 0
        samples = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            if mode == "reference":
                action = reference.copy()
            else:
                action, _ = model.predict(
                    observation, deterministic=True
                )

            observation, reward, terminated, truncated, step_info = (
                env.step(action)
            )

            position = np.asarray(
                env.simulator.get_position(), dtype=float
            )
            velocity = np.asarray(
                env.simulator.get_velocity(), dtype=float
            )

            error = float(np.linalg.norm(position - target))
            speed = float(np.linalg.norm(velocity))
            time = float(step_info["simulation_time"])
            steps = int(step_info["physics_steps"])

            total_reward += float(reward)
            weighted_error += error * steps
            total_steps += steps

            samples.append((time, error, speed))

        final_time, final_error, final_speed = samples[-1]

        # Success at the end of the episode, regardless of velocity.
        position_success = (
            not terminated and final_error < POSITION_THRESHOLD
        )

        # Require a full episode and stabilization throughout
        # the final second, rather than just passing the target.
        window_start = EPISODE_DURATION - HOLD_DURATION
        final_window = [
            (time, error, speed)
            for time, error, speed in samples
            if time > window_start - 1e-9
        ]

        sustained_success = (
            not terminated
            and truncated
            and final_time >= EPISODE_DURATION - 1e-6
            and bool(final_window)
            and all(
                error < POSITION_THRESHOLD
                and speed < SPEED_THRESHOLD
                for _, error, speed in final_window
            )
        )

        return {
            "policy": name,
            "seed": int(seed),
            "reward": total_reward,
            "mean_position_error": weighted_error / total_steps,
            "final_position_error": final_error,
            "final_speed": final_speed,
            "position_success": int(position_success),
            "final_stabilized": int(
                not terminated
                and final_error < POSITION_THRESHOLD
                and final_speed < SPEED_THRESHOLD
            ),
            "sustained_success": int(sustained_success),
            "terminated": int(terminated),
            "episode_time": final_time,
        }

    finally:
        env.close()


def summarize(rows):
    result = {"policy": rows[0]["policy"], "episodes": len(rows)}

    for key in (
        "reward",
        "mean_position_error",
        "final_position_error",
        "final_speed",
    ):
        result[f"mean_{key}"] = float(
            np.mean([row[key] for row in rows])
        )

    for key in (
        "position_success",
        "final_stabilized",
        "sustained_success",
        "terminated",
    ):
        result[f"{key}_rate"] = 100.0 * float(
            np.mean([row[key] for row in rows])
        )

    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original-500k", type=Path, required=True
    )
    parser.add_argument(
        "--original-2m", type=Path, required=True
    )
    parser.add_argument(
        "--stabilization-500k", type=Path, required=True
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/stabilization_benchmark"),
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    policies = [
        ("Target-aware PID", None, "reference", 1),
        (
            "Original PPO 500k",
            load_model(args.original_500k, 6),
            "ppo",
            6,
        ),
        (
            "Original PPO 2M",
            load_model(args.original_2m, 6),
            "ppo",
            6,
        ),
        (
            "Stabilization PPO 500k",
            load_model(args.stabilization_500k, 6),
            "ppo",
            6,
        ),
    ]

    all_rows = []
    summaries = []

    print("\nProject Aerial RL — Stabilization Benchmark")
    print("=" * 100)

    for name, model, mode, repeat in policies:
        rows = [
            evaluate_episode(
                name, model, mode, seed, repeat
            )
            for seed in range(args.seed, args.seed + args.episodes)
        ]

        all_rows.extend(rows)
        result = summarize(rows)
        summaries.append(result)

        print(
            f"{name:<24} | "
            f"Error: {result['mean_final_position_error']:.3f} m | "
            f"Speed: {result['mean_final_speed']:.3f} m/s | "
            f"Position: {result['position_success_rate']:.1f}% | "
            f"Stabilized: {result['final_stabilized_rate']:.1f}% | "
            f"Held 1s: {result['sustained_success_rate']:.1f}% | "
            f"Failures: {result['terminated_rate']:.1f}%"
        )

    write_csv(args.output / "episodes.csv", all_rows)
    write_csv(args.output / "summary.csv", summaries)

    print(f"\nSaved benchmark results to: {args.output}")


if __name__ == "__main__":
    main()

