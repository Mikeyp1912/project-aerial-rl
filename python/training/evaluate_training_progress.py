

"""Evaluate PPO checkpoints across cumulative training steps."""

import argparse
import csv
import re
from pathlib import Path

from stable_baselines3 import PPO

from compare_control_frequencies import evaluate, load_model


def find_checkpoints(initial_dir: Path, resumed_dir: Path):
    checkpoints = {}

    def register(path: Path):
        model = PPO.load(str(path), device="cpu")
        steps = int(model.num_timesteps)

        if steps <= 0:
            raise ValueError(f"Invalid timestep count in {path}")

        # Prefer an explicitly named checkpoint over a final model.
        is_final = path.name == "ppo_target_final.zip"

        if steps not in checkpoints or not is_final:
            checkpoints[steps] = path

    initial_final = initial_dir / "ppo_target_final.zip"
    if not initial_final.is_file():
        raise FileNotFoundError(initial_final)

    register(initial_final)

    for path in sorted(resumed_dir.glob("ppo_target_*_steps.zip")):
        if not re.fullmatch(r"ppo_target_\d+_steps\.zip", path.name):
            continue

        register(path)

    resumed_final = resumed_dir / "ppo_target_final.zip"
    if resumed_final.is_file():
        register(resumed_final)

    return sorted(checkpoints.items())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--initial-dir",
        type=Path,
        default=Path("training_results/ppo_target_20hz_100k"),
    )
    parser.add_argument(
        "--resumed-dir",
        type=Path,
        default=Path("training_results/ppo_target_20hz_500k"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/ppo_learning_curve.csv"),
    )

    args = parser.parse_args()

    if args.episodes < 2:
        parser.error("--episodes must be at least 2")

    checkpoints = find_checkpoints(
        args.initial_dir,
        args.resumed_dir,
    )

    if len(checkpoints) < 2:
        parser.error("At least two checkpoints are required")

    seeds = range(args.seed, args.seed + args.episodes)
    rows = []

    print("\nProject Aerial RL — PPO Learning Progress")
    print("=" * 100)

    for steps, path in checkpoints:
        model = load_model(path, expected_repeat=6)

        result = evaluate(
            name=f"PPO {steps:,}",
            model=model,
            mode="ppo",
            action_repeat=6,
            seeds=seeds,
        )

        result["training_steps"] = steps
        result["checkpoint"] = str(path)
        rows.append(result)

        print(
            f"{steps:>9,} steps | "
            f"Reward: {result['mean_reward']:>8.2f} | "
            f"Mean error: {result['mean_position_error']:.3f} m | "
            f"Final: {result['mean_final_error']:.3f} m | "
            f"Success: {result['success_rate']:5.1f}% | "
            f"Failures: {result['failure_rate']:5.1f}%"
        )
        print(
            "  Target correlations [X,Y,Z]: "
            + ", ".join(
                f"{result[f'correlation_{axis}']:+.3f}"
                for axis in ("x", "y", "z")
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: row["mean_final_error"])

    print("\nBest checkpoint by mean final error:")
    print(
        f"  {best['training_steps']:,} steps | "
        f"{best['mean_final_error']:.3f} m | "
        f"{best['success_rate']:.1f}% success"
    )
    print(f"  {best['checkpoint']}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
