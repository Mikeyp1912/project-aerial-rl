

"""Evaluate wind-trained PPO checkpoints against PID."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from evaluate_wind import evaluate


def find_checkpoints(directory: Path):
    checkpoints = []

    for path in directory.glob("ppo_hover_*_steps.zip"):
        match = re.fullmatch(
            r"ppo_hover_(\d+)_steps\.zip",
            path.name,
        )

        if match:
            checkpoints.append(
                (int(match.group(1)), path)
            )

    checkpoints.sort(key=lambda item: item[0])

    final_path = directory / "ppo_hover_final.zip"

    if final_path.is_file():
        checkpoints.append(("Final", final_path))

    return checkpoints


def load_configuration(directory: Path):
    path = directory / "configuration.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing training configuration: {path}"
        )

    return json.loads(path.read_text())


def print_results(rows, wind_speed):
    print(f"\nMaximum wind: {wind_speed:.1f} m/s")
    print("-" * 96)

    print(
        f"{'Checkpoint':<15}"
        f"{'Reward':>11}"
        f"{'Std':>10}"
        f"{'Mean XY':>11}"
        f"{'Final XY':>11}"
        f"{'Peak XY':>11}"
        f"{'Z error':>11}"
        f"{'Fail %':>10}"
    )

    for row in rows:
        print(
            f"{str(row['checkpoint']):<15}"
            f"{row['reward']:>11.2f}"
            f"{row['reward_std']:>10.2f}"
            f"{row['mean_xy']:>11.3f}"
            f"{row['final_xy']:>11.3f}"
            f"{row['peak_xy']:>11.3f}"
            f"{row['altitude_error']:>11.3f}"
            f"{100 * row['failure_rate']:>9.1f}%"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(
            "training_results/ppo_hover_wind_100k"
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
        default=[0.0, 2.5, 5.0, 10.0, 15.0],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if any(
        not np.isfinite(speed) or speed < 0
        for speed in args.wind_speeds
    ):
        parser.error(
            "Wind speeds must be finite and nonnegative"
        )

    config = load_configuration(args.directory)

    # The shared evaluator currently uses these fixed settings.
    expected = {
        "maximum_horizontal_offset": 1.0,
        "maximum_altitude_offset": 1.0,
        "episode_duration": 5.0,
        "linear_drag": [0.1, 0.1, 0.05],
    }

    for key, value in expected.items():
        if config.get(key) != value:
            parser.error(
                f"Training configuration {key}="
                f"{config.get(key)!r} differs from "
                f"evaluation setting {value!r}. "
                "Update evaluate_wind.py to match "
                "before comparing policies."
            )

    checkpoints = find_checkpoints(args.directory)

    if not checkpoints:
        parser.error(
            f"No checkpoints found in {args.directory}"
        )

    seeds = list(
        range(args.seed, args.seed + args.episodes)
    )

    print("\nProject Aerial RL — Wind PPO Checkpoints")
    print("=" * 70)
    print(f"Directory:   {args.directory}")
    print(f"Episodes:    {args.episodes}")
    print(f"Seeds:       {seeds[0]}–{seeds[-1]}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Wind speeds: {args.wind_speeds}")

    rows = []

    for wind_speed in args.wind_speeds:
        baseline = evaluate(
            None,
            seeds,
            duration=5.0,
            wind_speed=wind_speed,
        )

        baseline_row = {
            "checkpoint": "PID",
            "maximum_wind_speed": wind_speed,
            **baseline,
        }

        group = [baseline_row]
        rows.append(baseline_row)

        for step, path in checkpoints:
            print(
                f"Evaluating {path.name} "
                f"at {wind_speed:.1f} m/s...",
                flush=True,
            )

            model = PPO.load(
                str(path),
                device="cpu",
            )

            result = evaluate(
                model,
                seeds,
                duration=5.0,
                wind_speed=wind_speed,
            )

            row = {
                "checkpoint": step,
                "maximum_wind_speed": wind_speed,
                **result,
            }

            group.append(row)
            rows.append(row)

        print_results(group, wind_speed)

        trained = group[1:]

        best = max(
            trained,
            key=lambda row: row["reward"],
        )

        improvement = (
            best["reward"] - baseline["reward"]
        )

        print(
            f"\nBest PPO: {best['checkpoint']} | "
            f"Reward difference vs PID: "
            f"{improvement:+.2f}"
        )

    output = args.output or (
        args.directory / "wind_checkpoint_evaluation.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results: {output}")


if __name__ == "__main__":
    main()
