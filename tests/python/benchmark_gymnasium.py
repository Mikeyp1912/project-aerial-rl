# /tests/python/benchmark_gymnasium.py

"""Validate and benchmark the Project Aerial Gymnasium environment."""

import argparse
import math
import time

import numpy as np

import aerial_sim
from aerial_rl.environment import AerialDroneEnv


def validate_determinism(
    seed: int = 42,
    steps: int = 1000,
) -> None:
    """Verify identical seeds and actions produce identical trajectories."""

    env_a = AerialDroneEnv(maximum_episode_time=20.0)
    env_b = AerialDroneEnv(maximum_episode_time=20.0)

    try:
        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)

        np.testing.assert_array_equal(obs_a, obs_b)

        rng = np.random.default_rng(seed)

        for step in range(steps):
            action = rng.uniform(
                -1.0, 1.0, size=4
            ).astype(np.float32)

            result_a = env_a.step(action)
            result_b = env_b.step(action)

            np.testing.assert_array_equal(
                result_a[0], result_b[0]
            )

            assert result_a[1] == result_b[1]
            assert result_a[2] == result_b[2]
            assert result_a[3] == result_b[3]

            if result_a[2] or result_a[3]:
                obs_a, _ = env_a.reset(seed=seed + step + 1)
                obs_b, _ = env_b.reset(seed=seed + step + 1)
                np.testing.assert_array_equal(obs_a, obs_b)

        print(f"Determinism: PASS ({steps} steps)")

    finally:
        env_a.close()
        env_b.close()


def validate_observations(steps: int = 1000) -> None:
    """Check observation bounds and numerical validity."""

    env = AerialDroneEnv(maximum_episode_time=10.0)
    rng = np.random.default_rng(123)

    try:
        observation, _ = env.reset(seed=123)

        assert env.observation_space.contains(observation)

        for _ in range(steps):
            action = rng.uniform(
                -1.0, 1.0, size=4
            ).astype(np.float32)

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            assert observation.shape == (16,)
            assert observation.dtype == np.float32
            assert np.all(np.isfinite(observation))
            assert env.observation_space.contains(observation)
            assert math.isfinite(reward)

            if terminated or truncated:
                observation, _ = env.reset()

        print(f"Observation validation: PASS ({steps} steps)")

    finally:
        env.close()


def benchmark_gymnasium(
    steps: int,
    seed: int,
) -> float:
    """Measure throughput through the Gymnasium wrapper."""

    env = AerialDroneEnv(maximum_episode_time=10.0)
    action = np.zeros(4, dtype=np.float32)

    try:
        env.reset(seed=seed)

        # Warm up the simulator before timing.
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset()

        start = time.perf_counter()

        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset()

        elapsed = time.perf_counter() - start
        throughput = steps / elapsed

        print("\nGymnasium benchmark")
        print(f"  Steps:       {steps:,}")
        print(f"  Duration:    {elapsed:.4f} s")
        print(f"  Throughput:  {throughput:,.0f} steps/s")

        return throughput

    finally:
        env.close()


def benchmark_cpp_binding(
    steps: int,
    seed: int,
) -> float:
    """Measure throughput using pybind11 without Gymnasium."""

    env = aerial_sim.FlightEnvironment(
        maximum_episode_time=10.0,
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
    )

    action = [0.0, 0.0, 0.0, 0.0]

    env.reset(seed=seed)

    for _ in range(100):
        _, _, terminated, truncated = env.step(action)

        if terminated or truncated:
            env.reset(seed=seed)

    start = time.perf_counter()

    for _ in range(steps):
        _, _, terminated, truncated = env.step(action)

        if terminated or truncated:
            env.reset(seed=seed)

    elapsed = time.perf_counter() - start
    throughput = steps / elapsed

    print("\nDirect pybind11 benchmark")
    print(f"  Steps:       {steps:,}")
    print(f"  Duration:    {elapsed:.4f} s")
    print(f"  Throughput:  {throughput:,.0f} steps/s")

    return throughput


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps", type=int, default=100000
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must fit in uint32")

    print("Project Aerial RL — Environment Benchmark")
    print("=" * 45)

    validate_determinism()
    validate_observations()

    cpp_rate = benchmark_cpp_binding(
        steps=args.steps,
        seed=args.seed,
    )

    gym_rate = benchmark_gymnasium(
        steps=args.steps,
        seed=args.seed,
    )

    print("\nPerformance comparison")
    print(f"  Direct binding: {cpp_rate:,.0f} steps/s")
    print(f"  Gymnasium:      {gym_rate:,.0f} steps/s")
    print(f"  Relative rate:  {gym_rate / cpp_rate:.2%}")

    print("\nEnvironment benchmark complete.")


if __name__ == "__main__":
    main()

