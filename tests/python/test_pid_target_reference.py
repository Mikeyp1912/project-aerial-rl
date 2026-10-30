

"""Integration tests for PID target reference modes."""

import numpy as np
import aerial_sim

from gymnasium.utils.env_checker import check_env

from aerial_rl.environment import AerialDroneEnv


def make_environment(reference):
    return AerialDroneEnv(
        maximum_episode_time=2.0,
        randomize_initial_state=True,
        pid_target_reference=reference,
    )


def test_default_mode():
    env = AerialDroneEnv(maximum_episode_time=2.0)

    try:
        observation, _ = env.reset(seed=42)

        assert env.observation_space.contains(observation)

        action = np.zeros(4, dtype=np.float32)

        observation, reward, terminated, truncated, _ = (
            env.step(action)
        )

        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)

        print("Default reference mode: PASS")
    finally:
        env.close()


def test_world_relative_mode():
    env = make_environment(
        aerial_sim.PIDTargetReference.WORLD_RELATIVE
    )

    try:
        observation, info = env.reset(seed=123)

        assert env.observation_space.contains(observation)
        assert info["initial_position"].shape == (3,)

        for _ in range(240):
            action = np.zeros(4, dtype=np.float32)

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)

            if terminated or truncated:
                break

        print("World-relative episode: PASS")
    finally:
        env.close()


def test_deterministic_reset():
    env = make_environment(
        aerial_sim.PIDTargetReference.WORLD_RELATIVE
    )

    try:
        first, info_a = env.reset(seed=456)
        second, info_b = env.reset(seed=456)

        np.testing.assert_array_equal(first, second)

        np.testing.assert_array_equal(
            info_a["initial_position"],
            info_b["initial_position"],
        )

        print("World-relative deterministic reset: PASS")
    finally:
        env.close()


def test_gymnasium_checker():
    env = make_environment(
        aerial_sim.PIDTargetReference.WORLD_RELATIVE
    )

    try:
        check_env(env, skip_render_check=True)

        print("Gymnasium checker: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    test_default_mode()
    test_world_relative_mode()
    test_deterministic_reset()
    test_gymnasium_checker()

    print("PID target reference integration tests PASSED")
