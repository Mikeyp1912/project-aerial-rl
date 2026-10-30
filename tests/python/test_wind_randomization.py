

"""Integration tests for seeded wind randomization."""

import aerial_sim
import numpy as np

from aerial_rl.environment import AerialDroneEnv


def make_env(randomize_wind=True, maximum_wind_speed=5.0):
    return AerialDroneEnv(
        maximum_episode_time=5.0,
        randomize_initial_state=True,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        maximum_horizontal_offset=1.0,
        hover_centered_altitude=True,
        maximum_altitude_offset=1.0,
        randomize_wind=randomize_wind,
        maximum_wind_speed=maximum_wind_speed,
    )


def test_deterministic_wind():
    env = make_env()

    try:
        _, first = env.reset(seed=42)
        _, second = env.reset(seed=42)

        np.testing.assert_array_equal(
            first["wind_velocity"],
            second["wind_velocity"],
        )

        assert np.linalg.norm(first["wind_velocity"]) <= 5.0 + 1e-12
        assert first["wind_velocity"][2] == 0.0

    finally:
        env.close()


def test_different_seeds():
    env = make_env()

    try:
        _, first = env.reset(seed=42)
        _, second = env.reset(seed=43)

        assert not np.array_equal(
            first["wind_velocity"],
            second["wind_velocity"],
        )

    finally:
        env.close()


def test_wind_disabled():
    env = make_env(randomize_wind=False)

    try:
        _, info = env.reset(seed=42)

        np.testing.assert_array_equal(
            info["wind_velocity"],
            np.zeros(3),
        )

    finally:
        env.close()


def test_zero_wind_speed():
    env = make_env(maximum_wind_speed=0.0)

    try:
        _, info = env.reset(seed=42)

        np.testing.assert_array_equal(
            info["wind_velocity"],
            np.zeros(3),
        )

    finally:
        env.close()


def test_initial_state_independence():
    no_wind = make_env(randomize_wind=False)
    with_wind = make_env(randomize_wind=True)

    try:
        _, first = no_wind.reset(seed=123)
        _, second = with_wind.reset(seed=123)

        np.testing.assert_array_equal(
            first["initial_position"],
            second["initial_position"],
        )

        np.testing.assert_array_equal(
            first["initial_velocity"],
            second["initial_velocity"],
        )

    finally:
        no_wind.close()
        with_wind.close()


def test_wind_remains_constant():
    env = make_env()

    try:
        _, info = env.reset(seed=42)
        initial_wind = info["wind_velocity"]

        for _ in range(120):
            _, _, terminated, truncated, step_info = env.step(
                np.zeros(4, dtype=np.float32)
            )

            np.testing.assert_array_equal(
                initial_wind,
                step_info["wind_velocity"],
            )

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    test_deterministic_wind()
    test_different_seeds()
    test_wind_disabled()
    test_zero_wind_speed()
    test_initial_state_independence()
    test_wind_remains_constant()

    print("Wind randomization integration tests PASSED")
