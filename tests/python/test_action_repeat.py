

"""Tests for decoupled PPO and physics control frequencies."""

import numpy as np
import pytest

from aerial_rl.environment import AerialDroneEnv


ACTION = np.zeros(4, dtype=np.float32)


def test_default_action_repeat():
    env = AerialDroneEnv(maximum_episode_time=1.0)

    try:
        env.reset(seed=42)

        _, _, terminated, truncated, info = env.step(ACTION)

        assert not terminated
        assert not truncated
        assert info["physics_steps"] == 1
        assert info["simulation_time"] == pytest.approx(1 / 120)
    finally:
        env.close()


def test_six_physics_steps():
    env = AerialDroneEnv(
        maximum_episode_time=1.0,
        action_repeat=6,
    )

    try:
        env.reset(seed=42)

        _, _, terminated, truncated, info = env.step(ACTION)

        assert not terminated
        assert not truncated
        assert info["physics_steps"] == 6
        assert info["simulation_time"] == pytest.approx(6 / 120)
        assert info["elapsed_time"] == pytest.approx(0.05)
    finally:
        env.close()


def test_reward_accumulation():
    single = AerialDroneEnv(
        maximum_episode_time=1.0,
        action_repeat=1,
    )

    repeated = AerialDroneEnv(
        maximum_episode_time=1.0,
        action_repeat=6,
    )

    try:
        single.reset(seed=42)
        repeated.reset(seed=42)

        expected_reward = 0.0

        for _ in range(6):
            _, reward, terminated, truncated, _ = single.step(ACTION)

            expected_reward += reward

            assert not terminated
            assert not truncated

        _, actual_reward, _, _, _ = repeated.step(ACTION)

        assert actual_reward == pytest.approx(expected_reward)
    finally:
        single.close()
        repeated.close()


def test_early_truncation():
    env = AerialDroneEnv(
        maximum_episode_time=3 / 120,
        action_repeat=6,
    )

    try:
        env.reset(seed=42)

        _, _, terminated, truncated, info = env.step(ACTION)

        assert not terminated
        assert truncated
        assert info["physics_steps"] == 3
        assert info["simulation_time"] == pytest.approx(3 / 120)

        with pytest.raises(RuntimeError):
            env.step(ACTION)
    finally:
        env.close()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_action_repeat(value):
    with pytest.raises(ValueError):
        AerialDroneEnv(action_repeat=value)
