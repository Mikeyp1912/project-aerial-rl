

"""Evaluate PPO hover policies across randomized starting states."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from aerial_rl.environment import AerialDroneEnv

import aerial_sim

def evaluate_policy(
    name: str,
    model: PPO | None,
    seeds: list[int],
    duration: float,
    randomized: bool,
    reference: aerial_sim.PIDTargetReference,
):
    env = AerialDroneEnv(
        maximum_episode_time=duration,
        randomize_initial_state=randomized,
        pid_target_reference=reference,
    )

    rewards = []
    final_errors = []
    initial_errors = []
    improvements = []
    terminated_count = 0

    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)

            initial_position = info["initial_position"]
            initial_error = np.linalg.norm(
                initial_position[:2]
            )

            total_reward = 0.0

            while True:
                if model is None:
                    action = np.zeros(4, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

                if terminated or truncated:
                    break

            final_position = np.asarray(
                env.simulator.get_position(),
                dtype=np.float64,
            )

            final_error = np.linalg.norm(
                final_position[:2]
            )

            rewards.append(total_reward)
            initial_errors.append(initial_error)
            final_errors.append(final_error)
            improvements.append(initial_error - final_error)
            terminated_count += int(terminated)

    finally:
        env.close()

    return {
        "name": name,
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_initial_error": float(np.mean(initial_errors)),
        "mean_final_error": float(np.mean(final_errors)),
        "mean_improvement": float(np.mean(improvements)),
        "termination_rate": terminated_count / len(seeds),
    }


def print_results(results):
    print()
    print(
        f"{'Policy':<24}"
        f"{'Reward':>13}"
        f"{'Std':>12}"
        f"{'Initial XY':>13}"
        f"{'Final XY':>13}"
        f"{'Recovery':>13}"
        f"{'Failures':>11}"
    )
    print("-" * 99)

    for result in results:
        print(
            f"{result['name']:<24}"
            f"{result['mean_reward']:>13.2f}"
            f"{result['reward_std']:>12.2f}"
            f"{result['mean_initial_error']:>13.3f}"
            f"{result['mean_final_error']:>13.3f}"
            f"{result['mean_improvement']:>13.3f}"
            f"{result['termination_rate'] * 100:>10.1f}%"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=10000)

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "training_results/ppo_hover_conservative/"
            "ppo_hover_final.zip"
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if args.duration <= 0:
        parser.error("--duration must be positive")

    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")

    model = PPO.load(str(args.model), device="cpu")

    seeds = [
        args.seed + index
        for index in range(args.episodes)
    ]

    print("Project Aerial RL — PPO Generalization")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Duration: {args.duration:.1f} s")

    for randomized in (False, True):
        for reference_name, reference in (
            (
                "Episode-relative",
                aerial_sim.PIDTargetReference.EPISODE_RELATIVE,
            ),
            (
                "World-relative",
                aerial_sim.PIDTargetReference.WORLD_RELATIVE,
            ),
        ):
            print()
            print(
                f"{reference_name} | "
                f"{'Randomized' if randomized else 'Fixed'} start"
            )

            results = [
                evaluate_policy(
                    "Zero action",
                    None,
                    seeds,
                    args.duration,
                    randomized,
                    reference,
                ),
                evaluate_policy(
                    "Conservative PPO",
                    model,
                    seeds,
                    args.duration,
                    randomized,
                    reference,
                ),
            ]

            print_results(results)

if __name__ == "__main__":
    main()
