

"""Compare PPO policies trained with different target-error scales."""

import argparse
import csv
from pathlib import Path

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def make_environment(target_error_scale):
    return AerialDroneEnv(
        maximum_episode_time=10.0,
        randomize_initial_state=True,
        maximum_position_offset=(0.5, 0.5, 0.2),
        maximum_velocity=(0.2, 0.2, 0.0),
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
        pid_target_reference=aerial_sim.PIDTargetReference.WORLD_RELATIVE,
        maximum_horizontal_offset=2.0,
        hover_centered_altitude=True,
        maximum_altitude_offset=1.0,
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
        target_error_scale=target_error_scale,
        task=aerial_sim.FlightTask.TARGET_TRACKING,
        randomize_target=True,
        maximum_target_offset=(2.0, 2.0, 1.0),
        randomize_wind=False,
        linear_drag=(0.1, 0.1, 0.05),
        horizontal_action_weight=0.0,
        altitude_action_weight=0.0,
        yaw_action_weight=0.0,
        reward_mode=aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING,
    )


def target_aware_action(target):
    # Diagnostic reference, not used for PPO training.
    return np.clip(
        np.array([
            target[0] / 2.0,
            target[1] / 2.0,
            target[2] - 2.0,
            0.0,
        ], dtype=np.float32),
        -1.0,
        1.0,
    )


def evaluate(name, model, scale, seeds, mode):
    env = make_environment(scale)

    episode_rewards = []
    mean_errors = []
    final_errors = []
    successes = 0
    failures = 0

    initial_actions = []
    reference_actions = []
    all_actions = []

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            target = np.asarray(info["target_position"])

            reference = target_aware_action(target)

            total_reward = 0.0
            errors = []

            first_action = None

            while True:
                if mode == "zero":
                    action = np.zeros(4, dtype=np.float32)
                elif mode == "reference":
                    action = reference.copy()
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    action = np.asarray(action, dtype=np.float32)

                action = np.clip(action, -1.0, 1.0)

                if first_action is None:
                    first_action = action.copy()

                all_actions.append(action.copy())

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                position = np.asarray(
                    env.simulator.get_position(),
                    dtype=np.float64,
                )

                errors.append(float(np.linalg.norm(position - target)))
                total_reward += float(reward)

                if terminated or truncated:
                    failures += int(terminated)
                    break

            initial_actions.append(first_action)
            reference_actions.append(reference)

            episode_rewards.append(total_reward)
            mean_errors.append(np.mean(errors))
            final_errors.append(errors[-1])

            successes += int(errors[-1] < 0.2)

    finally:
        env.close()

    initial_actions = np.asarray(initial_actions)
    reference_actions = np.asarray(reference_actions)
    all_actions = np.asarray(all_actions)

    correlations = []

    for axis in range(3):
        x = initial_actions[:, axis]
        y = reference_actions[:, axis]

        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))

    return {
        "policy": name,
        "target_error_scale": scale,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_position_error": float(np.mean(mean_errors)),
        "mean_final_error": float(np.mean(final_errors)),
        "success_rate": 100.0 * successes / len(seeds),
        "failure_rate": 100.0 * failures / len(seeds),
        "correlation_x": correlations[0],
        "correlation_y": correlations[1],
        "correlation_z": correlations[2],
        "action_std_x": float(np.std(all_actions[:, 0])),
        "action_std_y": float(np.std(all_actions[:, 1])),
        "action_std_z": float(np.std(all_actions[:, 2])),
        "action_std_yaw": float(np.std(all_actions[:, 3])),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quadratic-model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bounded-model",
        type=Path,
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "training_results/reward_mode_comparison.csv"
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    for path in (args.quadratic_model, args.bounded_model):
        if not path.exists():
            parser.error(f"Model not found: {path}")

    quadratic_model = PPO.load(
        str(args.quadratic_model),
        device="cpu",
    )

    bounded_model = PPO.load(
        str(args.bounded_model),
        device="cpu",
    )

    for model in (quadratic_model, bounded_model):
        if model.observation_space.shape != (19,):
            raise ValueError(
                "Expected a 19-dimensional PPO observation"
            )

    seeds = range(args.seed, args.seed + args.episodes)

    policies = [
        ("Zero action", None, 2.0, "zero"),
        ("Target-aware PID", None, 2.0, "reference"),
        ("PPO quadratic", quadratic_model, 2.0, "ppo"),
        ("PPO bounded", bounded_model, 2.0, "ppo"),
    ]

    rows = []

    print(
        "\nProject Aerial RL — Reward Mode Comparison"
    )
    print("=" * 88)

    for name, model, scale, mode in policies:
        result = evaluate(name, model, scale, seeds, mode)
        rows.append(result)

        print(
            f"{name:<20} | "
            f"Reward: {result['mean_reward']:>9.2f} | "
            f"Mean error: {result['mean_position_error']:.3f} m | "
            f"Final: {result['mean_final_error']:.3f} m | "
            f"Success: {result['success_rate']:5.1f}%"
        )

        if mode == "ppo":
            print(
                "  Target correlations [X,Y,Z]: "
                + ", ".join(
                    f"{result[f'correlation_{axis}']:+.3f}"
                    for axis in ("x", "y", "z")
                )
            )
            print(
                "  Action std [X,Y,Z,Yaw]: "
                + ", ".join(
                    f"{result[f'action_std_{axis}']:.4f}"
                    for axis in ("x", "y", "z", "yaw")
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
