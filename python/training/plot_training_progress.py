

"""Plot PPO target-tracking performance across checkpoints."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training_results/ppo_learning_curve.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/ppo_learning_curve.png"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input).sort_values("training_steps")
    steps = data["training_steps"] / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    axes[0].plot(
        steps, data["mean_final_error"], marker="o",
        label="PPO final error",
    )
    axes[0].axhline(
        1.578, linestyle="--", label="Zero-action baseline",
    )
    axes[0].set_ylabel("Final position error (m)")
    axes[0].set_title("Target-tracking performance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        steps, data["success_rate"], marker="o",
    )
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    for axis in ("x", "y", "z"):
        axes[2].plot(
            steps,
            data[f"correlation_{axis}"],
            marker="o",
            label=axis.upper(),
        )

    axes[2].set_xlabel("Training steps (thousands)")
    axes[2].set_ylabel("Target/action correlation")
    axes[2].set_ylim(-1, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
