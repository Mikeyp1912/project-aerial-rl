

"""Diagnose PPO target awareness, actions, and critic predictions."""

import argparse
import csv
from pathlib import Path

import aerial_sim
import numpy as np
import torch
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def make_environment(
    target_error_scale=2.0,
):
    return AerialDroneEnv(
        maximum_episode_time=10.0,
        randomize_initial_state=True,
        maximum_position_offset=(0.5, 0.5, 0.2),
        maximum_velocity=(0.2, 0.2, 0.0),
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
        pid_target_reference=aerial_sim.PIDTargetReference.WORLD_RELATIVE,
        hover_centered_altitude=True,
        maximum_horizontal_offset=2.0,
        maximum_altitude_offset=1.0,
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
        task=aerial_sim.FlightTask.TARGET_TRACKING,
        randomize_target=True,
        maximum_target_offset=(2.0, 2.0, 1.0),
        randomize_wind=False,
        linear_drag=(0.1, 0.1, 0.05),
        horizontal_action_weight=0.0,
        altitude_action_weight=0.0,
        yaw_action_weight=0.0,
        target_error_scale=target_error_scale,
        reward_mode=aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING,
    )


def policy_value(model, observation):
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)

    return float(value.cpu().numpy().reshape(-1)[0])


def safe_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def diagnose(name, model, seeds):
    env = make_environment()

    initial_actions = []
    expert_actions = []
    initial_values = []
    initial_errors = []
    all_actions = []
    rewards = []
    final_errors = []

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            target = np.asarray(
                info["target_position"], dtype=np.float64
            )

            expert_action = np.clip(
                np.array(
                    [
                        target[0] / 2.0,
                        target[1] / 2.0,
                        target[2] - 2.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                -1.0,
                1.0,
            )

            action, _ = model.predict(
                observation, deterministic=True
            )

            initial_actions.append(
                np.asarray(action, dtype=np.float64)
            )
            expert_actions.append(expert_action)
            initial_values.append(policy_value(model, observation))
            initial_errors.append(
                np.linalg.norm(
                    np.asarray(info["initial_position"]) - target
                )
            )

            episode_reward = 0.0

            while True:
                action, _ = model.predict(
                    observation, deterministic=True
                )
                action = np.clip(
                    np.asarray(action, dtype=np.float64),
                    -1.0,
                    1.0,
                )

                all_actions.append(action.copy())

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                episode_reward += reward

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            final_errors.append(
                np.linalg.norm(
                    np.asarray(env.simulator.get_position()) - target
                )
            )

    finally:
        env.close()

    initial_actions = np.asarray(initial_actions)
    expert_actions = np.asarray(expert_actions)
    all_actions = np.asarray(all_actions)
    initial_values = np.asarray(initial_values)

    correlations = [
        safe_correlation(
            initial_actions[:, axis],
            expert_actions[:, axis],
        )
        for axis in range(3)
    ]

    result = {
        "policy": name,
        "mean_reward": float(np.mean(rewards)),
        "mean_final_error": float(np.mean(final_errors)),
        "initial_action_mae": float(
            np.mean(np.abs(initial_actions[:, :3] - expert_actions[:, :3]))
        ),
        "correlation_x": correlations[0],
        "correlation_y": correlations[1],
        "correlation_z": correlations[2],
        "mean_action_x": float(np.mean(all_actions[:, 0])),
        "mean_action_y": float(np.mean(all_actions[:, 1])),
        "mean_action_z": float(np.mean(all_actions[:, 2])),
        "mean_action_yaw": float(np.mean(all_actions[:, 3])),
        "std_action_x": float(np.std(all_actions[:, 0])),
        "std_action_y": float(np.std(all_actions[:, 1])),
        "std_action_z": float(np.std(all_actions[:, 2])),
        "std_action_yaw": float(np.std(all_actions[:, 3])),
        "saturation_fraction": float(
            np.mean(np.abs(all_actions) >= 0.99)
        ),
        "mean_predicted_value": float(np.mean(initial_values)),
        "std_predicted_value": float(np.std(initial_values)),
        "value_vs_initial_error_correlation": safe_correlation(
            initial_values, initial_errors
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "training_results/target_tracking_diagnostics.csv"
        ),
    )
    args = parser.parse_args()

    if args.episodes <= 1:
        parser.error("--episodes must be greater than 1")

    checkpoints = {
        "PPO quadratic scale 2": Path(
            "training_results/ppo_target_tracking_scale2_1m/"
            "ppo_target_final.zip"
        ),
        "PPO bounded 100k": Path(
            "training_results/ppo_target_bounded_100k/"
            "ppo_target_final.zip"
        ),
    }

    rows = []
    seeds = range(args.seed, args.seed + args.episodes)

    print("\nProject Aerial RL — Target Tracking Diagnostics")
    print("=" * 76)

    for name, path in checkpoints.items():
        if not path.exists():
            print(f"Skipping missing checkpoint: {path}")
            continue

        model = PPO.load(str(path), device="cpu")

        if model.observation_space.shape != (19,):
            raise ValueError(
                f"{name} has unexpected observation shape "
                f"{model.observation_space.shape}"
            )

        result = diagnose(name, model, seeds)
        rows.append(result)

        print(f"\n{name}")
        print(
            f"  Reward: {result['mean_reward']:.2f}"
            f" | Final error: {result['mean_final_error']:.3f} m"
        )
        print(
            "  Target/action correlations [X,Y,Z]: "
            + ", ".join(
                f"{result[f'correlation_{axis}']:+.3f}"
                for axis in ("x", "y", "z")
            )
        )
        print(
            f"  Initial action MAE: "
            f"{result['initial_action_mae']:.4f}"
        )
        print(
            "  Mean actions [X,Y,Z,Yaw]: "
            + ", ".join(
                f"{result[f'mean_action_{axis}']:+.4f}"
                for axis in ("x", "y", "z", "yaw")
            )
        )
        print(
            "  Action std [X,Y,Z,Yaw]: "
            + ", ".join(
                f"{result[f'std_action_{axis}']:.4f}"
                for axis in ("x", "y", "z", "yaw")
            )
        )
        print(
            f"  Action saturation: "
            f"{100 * result['saturation_fraction']:.2f}%"
        )
        print(
            f"  Critic value mean/std: "
            f"{result['mean_predicted_value']:.2f} / "
            f"{result['std_predicted_value']:.2f}"
        )
        print(
            f"  Critic value vs initial error correlation: "
            f"{result['value_vs_initial_error_correlation']:+.3f}"
        )

    if not rows:
        raise RuntimeError("No checkpoints were found")

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
