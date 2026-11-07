

"""Compare PPO policies trained at different control frequencies."""

import argparse
import csv
import json
from pathlib import Path

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


PHYSICS_DT = 1.0 / 120.0


def make_environment(action_repeat):
    return AerialDroneEnv(
        time_step=PHYSICS_DT,
        maximum_episode_time=10.0,
        action_repeat=action_repeat,
        randomize_initial_state=True,
        maximum_position_offset=(0.5, 0.5, 0.2),
        maximum_velocity=(0.2, 0.2, 0.0),
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        maximum_horizontal_offset=2.0,
        hover_centered_altitude=True,
        maximum_altitude_offset=1.0,
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
        target_error_scale=2.0,
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
    return np.clip(
        np.array(
            [
                target[0] / 2.0,
                target[1] / 2.0,
                target[2] - 2.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def evaluate(name, model, mode, action_repeat, seeds):
    env = make_environment(action_repeat)

    rewards = []
    mean_errors = []
    final_errors = []
    successes = []
    failures = []

    initial_actions = []
    reference_actions = []
    actions = []

    try:
        for seed in seeds:
            observation, info = env.reset(seed=int(seed))
            target = np.asarray(
                info["target_position"], dtype=np.float64
            )
            reference = target_aware_action(target)

            total_reward = 0.0
            weighted_error = 0.0
            total_physics_steps = 0
            final_error = float("nan")
            failed = False
            first_action = None

            while True:
                if mode == "zero":
                    action = np.zeros(4, dtype=np.float32)
                elif mode == "reference":
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

                if first_action is None:
                    first_action = action.copy()

                actions.append(action.copy())

                observation, reward, terminated, truncated, step_info = (
                    env.step(action)
                )

                position = np.asarray(
                    env.simulator.get_position(),
                    dtype=np.float64,
                )
                final_error = float(np.linalg.norm(position - target))

                steps = int(step_info["physics_steps"])

                # Time-weighted endpoint approximation of mean error.
                weighted_error += final_error * steps
                total_physics_steps += steps
                total_reward += float(reward)

                if terminated or truncated:
                    failed = bool(terminated)
                    break

            initial_actions.append(first_action)
            reference_actions.append(reference)
            rewards.append(total_reward)
            mean_errors.append(
                weighted_error / total_physics_steps
            )
            final_errors.append(final_error)
            successes.append(final_error < 0.2 and not failed)
            failures.append(failed)

    finally:
        env.close()

    initial_actions = np.asarray(initial_actions)
    reference_actions = np.asarray(reference_actions)
    actions = np.asarray(actions)

    return {
        "policy": name,
        "action_repeat": action_repeat,
        "control_frequency_hz": 120.0 / action_repeat,
        "mean_reward": float(np.mean(rewards)),
        "mean_position_error": float(np.mean(mean_errors)),
        "mean_final_error": float(np.mean(final_errors)),
        "success_rate": 100.0 * float(np.mean(successes)),
        "failure_rate": 100.0 * float(np.mean(failures)),
        **{
            f"correlation_{axis}": correlation(
                initial_actions[:, index],
                reference_actions[:, index],
            )
            for index, axis in enumerate(("x", "y", "z"))
        },
        **{
            f"action_std_{axis}": float(np.std(actions[:, index]))
            for index, axis in enumerate(("x", "y", "z", "yaw"))
        },
    }


def load_model(path, expected_repeat):
    if not path.exists():
        raise FileNotFoundError(path)

    config_path = path.parent / "configuration.json"

    if config_path.exists():
        config = json.loads(config_path.read_text())

        actual_repeat = int(config.get("action_repeat", 1))
        if actual_repeat != expected_repeat:
            raise ValueError(
                f"{path}: expected action_repeat={expected_repeat}, "
                f"configuration has {actual_repeat}"
            )

        if config.get("reward_mode") != "bounded":
            raise ValueError(
                f"{path}: expected bounded reward training"
            )

        if float(config.get("target_error_scale", 2.0)) != 2.0:
            raise ValueError(
                f"{path}: expected target_error_scale=2.0"
            )
    else:
        print(f"Warning: no configuration found for {path}")

    model = PPO.load(str(path), device="cpu")

    if model.observation_space.shape != (19,):
        raise ValueError(
            f"{path}: expected 19 observation dimensions"
        )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-120hz", type=Path, required=True
    )
    parser.add_argument(
        "--model-20hz", type=Path, required=True
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "training_results/control_frequency_comparison.csv"
        ),
    )

    args = parser.parse_args()

    if args.episodes < 2:
        parser.error("--episodes must be at least 2")

    model_120 = load_model(args.model_120hz, 1)
    model_20 = load_model(args.model_20hz, 6)

    seeds = range(args.seed, args.seed + args.episodes)

    policies = [
        ("Zero action", None, "zero", 1),
        ("Target-aware PID", None, "reference", 1),
        ("PPO 120 Hz", model_120, "ppo", 1),
        ("PPO 20 Hz", model_20, "ppo", 6),
    ]

    results = []

    print("\nProject Aerial RL — Control Frequency Comparison")
    print("=" * 100)

    for name, model, mode, repeat in policies:
        result = evaluate(name, model, mode, repeat, seeds)
        results.append(result)

        print(
            f"{name:<20} | "
            f"Reward: {result['mean_reward']:>8.2f} | "
            f"Mean error: {result['mean_position_error']:.3f} m | "
            f"Final: {result['mean_final_error']:.3f} m | "
            f"Success: {result['success_rate']:5.1f}% | "
            f"Failures: {result['failure_rate']:5.1f}%"
        )

        if mode == "ppo":
            print(
                "  Target/action correlations [X,Y,Z]: "
                + ", ".join(
                    f"{result[f'correlation_{axis}']:+.3f}"
                    for axis in ("x", "y", "z")
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
