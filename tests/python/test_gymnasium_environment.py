# /tests/python/test_gymnasium_environment.py


"""Integration tests for the Gymnasium drone environment."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from aerial_rl.environment import AerialDroneEnv


def test_environment_checker():
    env = AerialDroneEnv(maximum_episode_time=2.0)

    check_env(env, skip_render_check=True)

    env.close()


def test_observation_and_action_spaces():
    env = AerialDroneEnv(maximum_episode_time=2.0)

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert observation.shape == (16,)
    assert observation.dtype == np.float32
    assert isinstance(info, dict)

    action = np.zeros(4, dtype=np.float32)

    next_obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "simulation_time" in info

    env.close()


def test_deterministic_reset():
    env = AerialDroneEnv()

    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)

    np.testing.assert_array_equal(first, second)

    env.close()


def test_complete_episode():
    env = AerialDroneEnv(maximum_episode_time=2.0)

    env.reset(seed=42)

    steps = 0
    total_reward = 0.0

    while True:
        action = np.zeros(4, dtype=np.float32)

        _, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    assert steps > 0
    assert steps <= 241

    print(f"Episode steps: {steps}")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    env.close()


if __name__ == "__main__":
    test_environment_checker()
    test_observation_and_action_spaces()
    test_deterministic_reset()
    test_complete_episode()

    print("Gymnasium integration tests PASSED")

