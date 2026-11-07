

"""Evaluate target-tracking PPO checkpoints against PID baselines."""

import argparse
import csv
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
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        hover_centered_altitude=True,
        maximum_horizontal_offset=2.0,
        maximum_altitude_offset=1.0,
        observation_mode=(
            aerial_sim.ObservationMode.HOVER_AUGMENTED
        ),
        task=aerial_sim.FlightTask.TARGET_TRACKING,
        randomize_target=True,
        maximum_target_offset=(2.0, 2.0, 1.0),
        randomize_wind=False,
        linear_drag=(0.1, 0.1, 0.05),
        horizontal_action_weight=0.0,
        altitude_action_weight=0.0,
        yaw_action_weight=0.0,
    )


def get_action(policy_name, model, observation, target):
    if policy_name == "Zero action":
        return np.zeros(4, dtype=np.float32)

    if policy_name == "Target-aware PID":
        return np.array(
            [
                target[0] / 2.0,
                target[1] / 2.0,
                target[2] - 2.0,
                0.0,
            ],
            dtype=np.float32,
        )

    action, _ = model.predict(
        observation,
        deterministic=True,
    )
    return action


def evaluate(policy_name, model, seeds, duration):
    env = make_environment(duration)

    rewards = []
    mean_errors = []
    final_errors = []
    successes = []
    failures = 0

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            target = np.asarray(info["target_position"])

            episode_reward = 0.0
            errors = []

            while True:
                action = get_action(
                    policy_name,
                    model,
                    observation,
                    target,
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                episode_reward += reward

                position = np.asarray(
                    env.simulator.get_position(),
                    dtype=np.float64,
                )

                errors.append(
                    np.linalg.norm(position - target)
                )

                if terminated or truncated:
                    failures += int(terminated)
                    break

            rewards.append(episode_reward)
            mean_errors.append(np.mean(errors))
            final_errors.append(errors[-1])
            successes.append(errors[-1] < 0.2)

    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_position_error": float(np.mean(mean_errors)),
        "final_position_error": float(np.mean(final_errors)),
        "success_rate": float(np.mean(successes)),
        "failure_rate": failures / len(seeds),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20000)

    parser.add_argument(
        "--initial-directory",
        type=Path,
        default=Path(
            "training_results/ppo_target_tracking_200k"
        ),
    )

    parser.add_argument(
        "--continued-directory",
        type=Path,
        default=Path(
            "training_results/ppo_target_tracking_1m"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "training_results/target_tracking_evaluation.csv"
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0 or args.duration <= 0:
        parser.error("Episodes and duration must be positive")

    seeds = range(args.seed, args.seed + args.episodes)

    policies = [
        ("Zero action", None),
        ("Target-aware PID", None),
    ]

    initial_model = (
        args.initial_directory / "ppo_target_final.zip"
    )

    if initial_model.exists():
        policies.append(
            ("PPO 200k", PPO.load(str(initial_model), device="cpu"))
        )

    checkpoints = sorted(
        args.continued_directory.glob("ppo_target_*_steps.zip"),
        key=lambda path: int(path.stem.split("_")[-2]),
    )

    for checkpoint in checkpoints:
        steps = int(checkpoint.stem.split("_")[-2])

        policies.append(
            (
                f"PPO {steps:,}",
                PPO.load(str(checkpoint), device="cpu"),
            )
        )

    final_model = (
        args.continued_directory / "ppo_target_final.zip"
    )

    if final_model.exists():
        policies.append(
            ("PPO final", PPO.load(str(final_model), device="cpu"))
        )

    rows = []

    print("\nProject Aerial RL — Target Tracking Benchmark")
    print("=" * 110)

    for name, model in policies:
        result = evaluate(name, model, seeds, args.duration)

        rows.append({"controller": name, **result})

        print(
            f"{name:<22} | "
            f"Reward: {result['mean_reward']:>9.2f} | "
            f"Mean error: {result['mean_position_error']:.3f} m | "
            f"Final error: {result['final_position_error']:.3f} m | "
            f"Success: {100 * result['success_rate']:>5.1f}% | "
            f"Failures: {100 * result['failure_rate']:>5.1f}%"
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
