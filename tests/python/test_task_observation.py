"""Integration tests for configurable RL observations."""

import numpy as np
import aerial_sim

from aerial_rl.environment import AerialDroneEnv


def test_default_observation():
    env = AerialDroneEnv()

    try:
        observation, _ = env.reset(seed=42)

        assert observation.shape == (16,)
        assert env.observation_space.shape == (16,)
        assert env.simulator.get_observation_size() == 16

    finally:
        env.close()


def test_augmented_observation():
    env = AerialDroneEnv(
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
    )

    try:
        observation, _ = env.reset(seed=42)

        assert observation.shape == (19,)
        assert env.observation_space.shape == (19,)
        assert env.simulator.get_observation_size() == 19

        np.testing.assert_allclose(
            observation[16:19],
            np.array([0.0, 0.0, -0.1]),
            atol=1e-6,
        )

    finally:
        env.close()


def test_base_features_unchanged():
    base = AerialDroneEnv()

    augmented = AerialDroneEnv(
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
    )

    try:
        base_observation, _ = base.reset(seed=123)
        augmented_observation, _ = augmented.reset(seed=123)

        np.testing.assert_array_equal(
            base_observation,
            augmented_observation[:16],
        )

    finally:
        base.close()
        augmented.close()


def test_augmented_step():
    env = AerialDroneEnv(
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
    )

    try:
        env.reset(seed=42)

        observation, reward, terminated, truncated, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )

        assert observation.shape == (19,)
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated

    finally:
        env.close()


def test_randomized_initial_position_error():
    env = AerialDroneEnv(
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
        randomize_initial_state=True,
    )

    try:
        observation, info = env.reset(seed=123)

        expected = (
            info["initial_position"] -
            np.array([0.0, 0.0, 2.0])
        ) / 10.0

        np.testing.assert_allclose(
            observation[16:19],
            expected,
            atol=1e-6,
        )

    finally:
        env.close()


if __name__ == "__main__":
    test_default_observation()
    test_augmented_observation()
    test_base_features_unchanged()
    test_augmented_step()
    test_randomized_initial_position_error()

    print("Task observation integration tests PASSED")
