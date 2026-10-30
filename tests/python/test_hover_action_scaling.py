

"""Integration tests for hover-centred PID action scaling."""

import aerial_sim
import numpy as np

from gymnasium.utils.env_checker import check_env
from aerial_rl.environment import AerialDroneEnv


def make_env():
    return AerialDroneEnv(
        maximum_episode_time=2.0,
        randomize_initial_state=True,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        maximum_horizontal_offset=1.0,
        hover_centered_altitude=True,
        maximum_altitude_offset=1.0,
    )


def test_hover_configuration():
    env = make_env()

    try:
        observation, info = env.reset(seed=42)

        assert env.observation_space.contains(observation)
        assert info["initial_position"].shape == (3,)

        total_reward = 0.0

        for _ in range(240):
            observation, reward, terminated, truncated, _ = (
                env.step(np.zeros(4, dtype=np.float32))
            )

            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)

            total_reward += reward

            if terminated or truncated:
                break

        position = np.asarray(
            env.simulator.get_position(),
            dtype=np.float64,
        )

        print("Final position:", position)
        print("Total reward:", total_reward)

        assert np.all(np.isfinite(position))
        assert abs(position[2] - 2.0) < 0.25
        assert np.linalg.norm(position[:2]) < 0.25

        print("Hover-centred configuration: PASS")

    finally:
        env.close()


def test_determinism():
    env = make_env()

    try:
        first, first_info = env.reset(seed=123)
        second, second_info = env.reset(seed=123)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first_info["initial_position"],
            second_info["initial_position"],
        )

        print("Deterministic reset: PASS")

    finally:
        env.close()


def test_gymnasium():
    env = make_env()

    try:
        check_env(env, skip_render_check=True)
        print("Gymnasium checker: PASS")

    finally:
        env.close()


if __name__ == "__main__":
    test_hover_configuration()
    test_determinism()
    test_gymnasium()

    print("Hover action scaling integration tests PASSED")
