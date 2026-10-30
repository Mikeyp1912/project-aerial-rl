

"""Inspect PPO action outputs and final drone state."""

import argparse

import aerial_sim
import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv


def evaluate(model_path, episodes):
    model = PPO.load(model_path, device="cpu")

    env = AerialDroneEnv(
        maximum_episode_time=5.0,
        randomize_initial_state=True,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
    )

    actions = []
    final_positions = []
    final_velocities = []
    rewards = []

    try:
        for seed in range(10000, 10000 + episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0

            while True:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                actions.append(np.asarray(action).copy())

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                if terminated or truncated:
                    break

            final_positions.append(
                env.simulator.get_position()
            )
            final_velocities.append(
                env.simulator.get_velocity()
            )
            rewards.append(total_reward)

    finally:
        env.close()

    actions = np.asarray(actions)
    positions = np.asarray(final_positions)
    velocities = np.asarray(final_velocities)

    print("\nPolicy diagnostics")
    print("=" * 55)

    print("Mean action:", actions.mean(axis=0))
    print("Action std: ", actions.std(axis=0))
    print("Action min: ", actions.min(axis=0))
    print("Action max: ", actions.max(axis=0))

    print("\nMean final position:", positions.mean(axis=0))
    print("Mean final velocity:", velocities.mean(axis=0))

    print(
        "Mean final XY error:",
        np.linalg.norm(positions[:, :2], axis=1).mean(),
    )

    print(
        "Mean final altitude error:",
        np.abs(positions[:, 2] - 2.0).mean(),
    )

    print("Mean reward:", np.mean(rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    evaluate(args.model, args.episodes)
