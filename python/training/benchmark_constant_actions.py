# python/training/benchmark_constant_actions.py


"""Compare constant PID-target actions in Project Aerial RL."""

import numpy as np

from aerial_rl.environment import AerialDroneEnv


DURATION = 5.0
SEED = 42

# Each action represents [X offset, Y offset, altitude, yaw].
ACTIONS = {
    "zero": [0.0, 0.0, 0.0, 0.0],
    "x +0.1": [0.1, 0.0, 0.0, 0.0],
    "x -0.1": [-0.1, 0.0, 0.0, 0.0],
    "y +0.1": [0.0, 0.1, 0.0, 0.0],
    "y -0.1": [0.0, -0.1, 0.0, 0.0],
    "x +0.5": [0.5, 0.0, 0.0, 0.0],
    "y +0.5": [0.0, 0.5, 0.0, 0.0],
    "altitude +0.1": [0.0, 0.0, 0.1, 0.0],
    "altitude -0.1": [0.0, 0.0, -0.1, 0.0],
    "yaw +0.1": [0.0, 0.0, 0.0, 0.1],
}


def evaluate(action_values):
    env = AerialDroneEnv(maximum_episode_time=DURATION)

    action = np.asarray(action_values, dtype=np.float32)

    observation, _ = env.reset(seed=SEED)

    total_reward = 0.0
    steps = 0

    try:
        while True:
            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break
    finally:
        env.close()

    # First three normalized observation entries are position.
    final_position = observation[:3].astype(np.float64) * 10.0

    return total_reward, steps, final_position


def main():
    print("Project Aerial RL — Constant Action Benchmark")
    print("=" * 75)

    results = []

    for name, action in ACTIONS.items():
        reward, steps, position = evaluate(action)

        results.append((name, reward, steps, position))

    results.sort(key=lambda item: item[1], reverse=True)

    print(
        f"{'Action':<20}"
        f"{'Reward':>14}"
        f"{'Steps':>10}"
        f"{'Final X':>12}"
        f"{'Final Y':>12}"
    )

    print("-" * 75)

    for name, reward, steps, position in results:
        print(
            f"{name:<20}"
            f"{reward:>14.3f}"
            f"{steps:>10}"
            f"{position[0]:>12.3f}"
            f"{position[1]:>12.3f}"
        )


if __name__ == "__main__":
    main()

