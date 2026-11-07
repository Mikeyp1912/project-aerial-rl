

"""Integration tests for selectable C++ reward modes."""

import numpy as np
import pytest

import aerial_sim
from aerial_rl.environment import AerialDroneEnv


def make_env(reward_mode, **kwargs):
    return AerialDroneEnv(
        maximum_episode_time=1.0,
        reward_mode=reward_mode,
        **kwargs,
    )


def test_default_reward_matches_explicit_quadratic():
    default = AerialDroneEnv(maximum_episode_time=1.0)
    explicit = make_env(aerial_sim.RewardMode.QUADRATIC_HOVER)

    try:
        default_obs, _ = default.reset(seed=42)
        explicit_obs, _ = explicit.reset(seed=42)

        np.testing.assert_allclose(default_obs, explicit_obs)

        action = np.zeros(4, dtype=np.float32)

        _, default_reward, _, _, _ = default.step(action)
        _, explicit_reward, _, _, _ = explicit.step(action)

        assert default_reward == pytest.approx(explicit_reward)
    finally:
        default.close()
        explicit.close()


def test_bounded_reward_is_selectable():
    env = make_env(
        aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING
    )

    try:
        observation, _ = env.reset(seed=42)

        assert observation.shape == (16,)

        _, reward, _, _, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )

        assert np.isfinite(reward)
    finally:
        env.close()


def test_invalid_bounded_reward_sigma():
    with pytest.raises((ValueError, RuntimeError)):
        make_env(
            aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING,
            reward_position_sigma=0.0,
        )


def test_invalid_bounded_reward_weight():
    with pytest.raises((ValueError, RuntimeError)):
        make_env(
            aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING,
            reward_velocity_weight=-1.0,
        )
