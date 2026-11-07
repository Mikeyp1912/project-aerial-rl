

"""Test configurable target-error normalization."""

import numpy as np

import aerial_sim
from aerial_rl.environment import AerialDroneEnv


def make_env(scale):
    return AerialDroneEnv(
        observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
        target_error_scale=scale,
    )


def test_target_error_scale():
    env_default = make_env(10.0)
    env_scaled = make_env(2.0)

    try:
        obs_default, info_default = env_default.reset(seed=42)
        obs_scaled, info_scaled = env_scaled.reset(seed=42)

        assert obs_default.shape == (19,)
        assert obs_scaled.shape == (19,)

        # Changing target-error normalization must not change
        # the underlying simulation state.
        np.testing.assert_allclose(
            info_default["initial_position"],
            info_scaled["initial_position"],
        )

        # The first 16 normalized observation values must
        # remain unchanged.
        np.testing.assert_allclose(
            obs_default[:16],
            obs_scaled[:16],
        )

        # Error features should scale by 10 / 2 = 5,
        # unless clipping is active.
        expected = np.clip(
            obs_default[16:19] * 5.0,
            -1.0,
            1.0,
        )

        np.testing.assert_allclose(
            obs_scaled[16:19],
            expected,
            atol=1e-6,
        )

        # The target itself should not change.
        np.testing.assert_allclose(
            info_default["target_position"],
            info_scaled["target_position"],
        )

    finally:
        env_default.close()
        env_scaled.close()


def test_invalid_scale():
    try:
        env = make_env(0.0)
    except (ValueError, RuntimeError):
        return

    env.close()
    raise AssertionError("Zero target-error scale was accepted")


if __name__ == "__main__":
    test_target_error_scale()
    test_invalid_scale()

    print("Target-error normalization Python tests PASSED")
