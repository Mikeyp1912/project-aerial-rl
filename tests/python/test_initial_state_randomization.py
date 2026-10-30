

"""Test seeded initial-state randomization through Gymnasium."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from aerial_rl.environment import AerialDroneEnv


def make_environment(randomized: bool) -> AerialDroneEnv:
    return AerialDroneEnv(
        maximum_episode_time=2.0,
        randomize_initial_state=randomized,
        maximum_position_offset=(0.5, 0.5, 0.2),
        maximum_velocity=(0.2, 0.2, 0.0),
    )


def test_default_state():
    env = make_environment(False)

    try:
        _, info = env.reset(seed=42)

        np.testing.assert_array_equal(
            info["initial_position"],
            [0.0, 0.0, 1.0],
        )

        np.testing.assert_array_equal(
            info["initial_velocity"],
            [0.0, 0.0, 0.0],
        )

        print("Default state: PASS")
    finally:
        env.close()


def test_same_seed():
    env = make_environment(True)

    try:
        observation_a, info_a = env.reset(seed=123)
        observation_b, info_b = env.reset(seed=123)

        np.testing.assert_array_equal(
            observation_a, observation_b
        )

        np.testing.assert_array_equal(
            info_a["initial_position"],
            info_b["initial_position"],
        )

        np.testing.assert_array_equal(
            info_a["initial_velocity"],
            info_b["initial_velocity"],
        )

        print("Same-seed reproducibility: PASS")
    finally:
        env.close()


def test_different_seeds():
    env = make_environment(True)

    try:
        _, info_a = env.reset(seed=123)
        _, info_b = env.reset(seed=456)

        position_differs = not np.array_equal(
            info_a["initial_position"],
            info_b["initial_position"],
        )

        velocity_differs = not np.array_equal(
            info_a["initial_velocity"],
            info_b["initial_velocity"],
        )

        assert position_differs or velocity_differs

        print("Different-seed variation: PASS")
    finally:
        env.close()


def test_randomization_bounds():
    env = make_environment(True)

    try:
        for seed in range(100):
            observation, info = env.reset(seed=seed)

            position = info["initial_position"]
            velocity = info["initial_velocity"]

            offset = position - np.array(
                [0.0, 0.0, 1.0]
            )

            assert np.all(
                np.abs(offset)
                <= np.array([0.5, 0.5, 0.2]) + 1e-12
            )

            assert np.all(
                np.abs(velocity)
                <= np.array([0.2, 0.2, 0.0]) + 1e-12
            )

            assert env.observation_space.contains(observation)

        print("Randomization bounds: PASS")
    finally:
        env.close()


def test_complete_randomized_episode():
    env = make_environment(True)

    try:
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)

        for _ in range(241):
            _, reward, terminated, truncated, _ = env.step(
                action
            )

            assert np.isfinite(reward)

            if terminated or truncated:
                break
        else:
            raise AssertionError(
                "Randomized episode did not finish"
            )

        print("Randomized episode: PASS")
    finally:
        env.close()


def test_gymnasium_checker():
    env = make_environment(True)

    try:
        check_env(env, skip_render_check=True)

        print("Gymnasium checker: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    test_default_state()
    test_same_seed()
    test_different_seeds()
    test_randomization_bounds()
    test_complete_randomized_episode()
    test_gymnasium_checker()

    print("Initial-state randomization tests PASSED")
