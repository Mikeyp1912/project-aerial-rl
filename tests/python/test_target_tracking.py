

"""Integration tests for target-tracking Gymnasium environments."""

import numpy as np

import aerial_sim
from aerial_rl.environment import AerialDroneEnv


def make_environment(
    *,
    task=aerial_sim.FlightTask.TARGET_TRACKING,
    randomize_target=True,
):
    return AerialDroneEnv(
        maximum_episode_time=5.0,
        action_mode=aerial_sim.ActionMode.PID_TARGETS,
        pid_target_reference=(
            aerial_sim.PIDTargetReference.WORLD_RELATIVE
        ),
        hover_centered_altitude=True,
        maximum_horizontal_offset=2.0,
        maximum_altitude_offset=1.0,
        observation_mode=(
            aerial_sim.ObservationMode.HOVER_AUGMENTED
        ),
        task=task,
        randomize_target=randomize_target,
        maximum_target_offset=(2.0, 2.0, 1.0),
    )


def test_target_observation():
    env = make_environment()

    observation, info = env.reset(seed=123)

    assert observation.shape == (19,)
    assert env.observation_space.contains(observation)

    target = info["target_position"]
    assert target.shape == (3,)

    assert np.all(np.abs(target[:2]) <= 2.0)
    assert 1.0 <= target[2] <= 3.0

    # Compare against the sensor-derived position observation.
    # The default position normalization scale is 10 m.
    expected_error = (
        observation[:3].astype(np.float64)
        - target / 10.0
    )

    np.testing.assert_allclose(
        observation[16:19],
        expected_error,
        atol=1e-5,
    )

    env.close()


def test_reproducible_targets():
    env = make_environment()

    _, first_info = env.reset(seed=456)
    _, second_info = env.reset(seed=456)

    np.testing.assert_array_equal(
        first_info["target_position"],
        second_info["target_position"],
    )

    _, third_info = env.reset(seed=789)

    assert not np.array_equal(
        first_info["target_position"],
        third_info["target_position"],
    )

    env.close()


def test_target_remains_constant():
    env = make_environment()

    _, info = env.reset(seed=321)
    target = info["target_position"].copy()

    for _ in range(20):
        _, reward, terminated, truncated, step_info = (
            env.step(np.zeros(4, dtype=np.float32))
        )

        np.testing.assert_array_equal(
            step_info["target_position"],
            target,
        )

        assert np.isfinite(reward)

        if terminated or truncated:
            break

    env.close()


def test_hover_compatibility():
    env = make_environment(
        task=aerial_sim.FlightTask.HOVER,
        randomize_target=True,
    )

    _, info = env.reset(seed=123)

    np.testing.assert_allclose(
        info["target_position"],
        [0.0, 0.0, 2.0],
        atol=1e-12,
    )

    env.close()


def test_zero_action_does_not_follow_target():
    env = make_environment()

    # Test several seeds so a single target near the
    # original hover position cannot dominate the result.
    errors = []

    for seed in range(10):
        env.reset(seed=seed)

        target = np.asarray(
            env.simulator.get_target_position(),
            dtype=np.float64,
        )

        for _ in range(240):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(4, dtype=np.float32)
            )

            if terminated or truncated:
                break

        position = np.asarray(
            env.simulator.get_position(),
            dtype=np.float64,
        )

        errors.append(np.linalg.norm(position - target))

    # This checks that zero action is not a perfect
    # target-tracking solution.
    assert np.mean(errors) > 0.2

    env.close()


if __name__ == "__main__":
    test_target_observation()
    test_reproducible_targets()
    test_target_remains_constant()
    test_hover_compatibility()
    test_zero_action_does_not_follow_target()

    print("Python target tracking tests PASSED.")
