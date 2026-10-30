# /python/training/diagnose_ppo.py

"""Diagnose the behaviour of a trained PPO flight policy."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


MODEL_PATH = Path(
    "training_results/ppo_hover_conservative/ppo_hover_final.zip"
)


def evaluate_controller(name, controller, episodes=5):
    rewards = []
    max_actions = []
    final_positions = []

    env = AerialDroneEnv(maximum_episode_time=10.0)

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=1000 + episode)

            total_reward = 0.0
            peak_action = 0.0

            while True:
                action = np.asarray(
                    controller(observation),
                    dtype=np.float32,
                )

                peak_action = max(
                    peak_action,
                    float(np.max(np.abs(action))),
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            max_actions.append(peak_action)

            # Position occupies the first three observation entries.
            # Normalization uses a 10 m position scale.
            final_positions.append(
                observation[:3].astype(np.float64) * 10.0
            )

    finally:
        env.close()

    print(f"\n{name}")
    print(f"  Mean reward:       {np.mean(rewards):.4f}")
    print(f"  Reward std:        {np.std(rewards):.4f}")
    print(f"  Maximum |action|:  {np.max(max_actions):.4f}")
    print(
        "  Final positions:  "
        f"{np.mean(final_positions, axis=0)}"
    )


def main():
    model = PPO.load(str(MODEL_PATH), device="cpu")

    evaluate_controller(
        "Zero-action baseline",
        lambda obs: np.zeros(4, dtype=np.float32),
    )

    evaluate_controller(
        "Trained PPO",
        lambda obs: model.predict(
            obs,
            deterministic=True,
        )[0],
    )

    env = AerialDroneEnv(maximum_episode_time=10.0)

    try:
        observation, _ = env.reset(seed=1000)

        print("\nInitial PPO actions")

        for step in range(10):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            print(f"  Step {step:2d}: {action}")

            observation, _, terminated, truncated, _ = (
                env.step(action)
            )

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Diagnose the behaviour of a trained PPO flight policy."""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


MODEL_PATH = Path(
    "training_results/ppo_hover/ppo_hover_final.zip"
)


def evaluate_controller(name, controller, episodes=5):
    rewards = []
    max_actions = []
    final_positions = []

    env = AerialDroneEnv(maximum_episode_time=10.0)

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=1000 + episode)

            total_reward = 0.0
            peak_action = 0.0

            while True:
                action = np.asarray(
                    controller(observation),
                    dtype=np.float32,
                )

                peak_action = max(
                    peak_action,
                    float(np.max(np.abs(action))),
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            max_actions.append(peak_action)

            # Position occupies the first three observation entries.
            # Normalization uses a 10 m position scale.
            final_positions.append(
                observation[:3].astype(np.float64) * 10.0
            )

    finally:
        env.close()

    print(f"\n{name}")
    print(f"  Mean reward:       {np.mean(rewards):.4f}")
    print(f"  Reward std:        {np.std(rewards):.4f}")
    print(f"  Maximum |action|:  {np.max(max_actions):.4f}")
    print(
        "  Final positions:  "
        f"{np.mean(final_positions, axis=0)}"
    )


def main():
    model = PPO.load(str(MODEL_PATH), device="cpu")

    evaluate_controller(
        "Zero-action baseline",
        lambda obs: np.zeros(4, dtype=np.float32),
    )

    evaluate_controller(
        "Trained PPO",
        lambda obs: model.predict(
            obs,
            deterministic=True,
        )[0],
    )

    env = AerialDroneEnv(maximum_episode_time=10.0)

    try:
        observation, _ = env.reset(seed=1000)

        print("\nInitial PPO actions")

        for step in range(10):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            print(f"  Step {step:2d}: {action}")

            observation, _, terminated, truncated, _ = (
                env.step(action)
            )

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

