

"""Record and plot target-tracking trajectories for PPO and PID."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_control_frequencies import (
    load_model,
    make_environment,
    target_aware_action,
)


def record_episode(name, model, mode, seed, action_repeat):
    env = make_environment(action_repeat)
    rows = []

    try:
        observation, info = env.reset(seed=seed)
        target = np.asarray(info["target_position"], dtype=float)
        reference = target_aware_action(target)

        def record(time, position, velocity, action, reward, status):
            error = target - position
            rows.append({
                "policy": name,
                "seed": seed,
                "time": time,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "vx": velocity[0],
                "vy": velocity[1],
                "vz": velocity[2],
                "target_x": target[0],
                "target_y": target[1],
                "target_z": target[2],
                "error_x": error[0],
                "error_y": error[1],
                "error_z": error[2],
                "position_error": float(np.linalg.norm(error)),
                "speed": float(np.linalg.norm(velocity)),
                "action_x": action[0],
                "action_y": action[1],
                "action_z": action[2],
                "action_yaw": action[3],
                "reward": reward,
                "status": status,
            })

        record(
            0.0,
            np.asarray(env.simulator.get_position(), dtype=float),
            np.asarray(env.simulator.get_velocity(), dtype=float),
            np.zeros(4),
            0.0,
            "initial",
        )

        while True:
            if mode == "reference":
                action = reference.copy()
            else:
                action, _ = model.predict(
                    observation, deterministic=True
                )

            action = np.clip(
                np.asarray(action, dtype=np.float32),
                -1.0,
                1.0,
            )

            observation, reward, terminated, truncated, step_info = (
                env.step(action)
            )

            status = (
                "terminated" if terminated
                else "truncated" if truncated
                else "running"
            )

            record(
                float(step_info["simulation_time"]),
                np.asarray(env.simulator.get_position(), dtype=float),
                np.asarray(env.simulator.get_velocity(), dtype=float),
                action,
                float(reward),
                status,
            )

            if terminated or truncated:
                break

    finally:
        env.close()

    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(series, seed, output):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    for name, rows in series.items():
        t = np.array([row["time"] for row in rows])
        error = np.array([row["position_error"] for row in rows])

        axes[0].plot(t, error, label=name)

    axes[0].axhline(
        0.2, linestyle="--", color="gray",
        label="Success threshold (0.2 m)",
    )
    axes[0].set_title(f"Target tracking — seed {seed}")
    axes[0].set_ylabel("Position error (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, rows in series.items():
        t = np.array([row["time"] for row in rows])
        x = np.array([row["x"] for row in rows])
        y = np.array([row["y"] for row in rows])
        z = np.array([row["z"] for row in rows])

        axes[1].plot(t, x, label=f"{name} X")
        axes[1].plot(t, y, label=f"{name} Y", linestyle="--")
        axes[1].plot(t, z, label=f"{name} Z", linestyle=":")

    reference_rows = next(iter(series.values()))
    for axis, color in zip(("x", "y", "z"), ("C0", "C1", "C2")):
        axes[1].axhline(
            reference_rows[0][f"target_{axis}"],
            color=color,
            alpha=0.45,
            linewidth=1,
        )

    axes[1].set_ylabel("World position (m)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for name, rows in series.items():
        if name == "Target-aware PID":
            continue

        t = np.array([row["time"] for row in rows])

        for axis, linestyle in zip(
            ("x", "y", "z"), ("-", "--", ":")
        ):
            axes[2].plot(
                t,
                [row[f"action_{axis}"] for row in rows],
                label=f"{name} {axis.upper()}",
                linestyle=linestyle,
            )

    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("Normalized PPO action")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-300k",
        type=Path,
        default=Path(
            "training_results/ppo_target_20hz_500k/"
            "ppo_target_300000_steps.zip"
        ),
    )
    parser.add_argument(
        "--model-500k",
        type=Path,
        default=Path(
            "training_results/ppo_target_20hz_500k/"
            "ppo_target_final.zip"
        ),
    )
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/trajectory_diagnostics"),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_300 = load_model(args.model_300k, expected_repeat=6)
    model_500 = load_model(args.model_500k, expected_repeat=6)

    args.output.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seed, args.seed + args.episodes):
        series = {
            "Target-aware PID": record_episode(
                "Target-aware PID", None, "reference", seed, 1
            ),
            "PPO 300k": record_episode(
                "PPO 300k", model_300, "ppo", seed, 6
            ),
            "PPO 500k": record_episode(
                "PPO 500k", model_500, "ppo", seed, 6
            ),
        }

        for name, rows in series.items():
            filename = name.lower().replace(" ", "_")
            save_csv(
                rows,
                args.output / f"seed_{seed}_{filename}.csv",
            )

            final = rows[-1]
            print(
                f"Seed {seed} | {name:<18} | "
                f"Final error: {final['position_error']:.3f} m | "
                f"Final speed: {final['speed']:.3f} m/s | "
                f"Status: {final['status']}"
            )

        plot_episode(
            series,
            seed,
            args.output / f"seed_{seed}_comparison.png",
        )

    print(f"\nSaved trajectory diagnostics to: {args.output}")


if __name__ == "__main__":
    main()
