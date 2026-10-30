# tests/python/test_aerial_sim.py

import math

import aerial_sim

def main():
    env = aerial_sim.FlightEnvironment(
            time_step = 1.0 / 120.0,
            maximum_episode_time = 2.0,
            action_mode = aerial_sim.ActionMode.PID_TARGETS,
            )

    observation = env.reset(seed = 42)

    assert len(observation) == 16
    assert all(math.isfinite(value) for value in observation)

    print("Initial observation")
    print(observation)

    action = [0.0, 0.0, 0.0, 0.0]

    total_reward = 0.0
    steps = 0

    while True:
        observation, reward, terminated, truncated = env.step(
                action
                )

        assert len(observation) == 16
        assert all(math.isfinite(value) for value in observation)
        assert math.isfinite(reward)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    assert steps > 0
    assert env.get_simulation_time() > 0.0

    print(f"Steps: {steps}")
    print(f"Simulation time: {env.get_simulation_time():.3f} s")
    print(f"Total reward: {total_reward:.6f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    # Check deterministic reset behaviour
    observation_a = env.reset(seed = 123)
    observation_b = env.reset(seed = 123)

    assert observation_a == observation_b

    print("Python bindings test PASSED")

if __name__ == "__main__":
    main()
