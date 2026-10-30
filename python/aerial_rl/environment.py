# /python/aerial_rl/environment.py

"""Gymnasium environment for Project Aerial RL."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import aerial_sim


class AerialDroneEnv(gym.Env):
    """Gymnasium wrapper around the C++ quadcopter simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        time_step: float = 1.0 / 120.0,
        maximum_episode_time: float = 10.0,
        action_mode: aerial_sim.ActionMode = aerial_sim.ActionMode.PID_TARGETS,
    ):
        super().__init__()

        self.simulator = aerial_sim.FlightEnvironment(
            time_step=time_step,
            maximum_episode_time=maximum_episode_time,
            action_mode=action_mode,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self._episode_finished = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Gymnasium manages a seeded RNG. Draw a C++-compatible
        # uint32 seed so resets remain reproducible.
        cpp_seed = int(
            self.np_random.integers(0, 2**32, dtype=np.uint32)
        )

        observation = self.simulator.reset(seed=cpp_seed)

        self._episode_finished = False

        return (
            np.asarray(observation, dtype=np.float32),
            {},
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._episode_finished:
            raise RuntimeError(
                "Episode finished. Call reset() before step()."
            )

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (4,):
            raise ValueError(
                f"Expected action shape (4,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or infinity")

        # The C++ action adapter expects normalized actions.
        action = np.clip(action, -1.0, 1.0)

        observation, reward, terminated, truncated = (
            self.simulator.step(action.tolist())
        )

        self._episode_finished = bool(terminated or truncated)

        info = {
            "simulation_time": float(
                self.simulator.get_simulation_time()
            ),
        }

        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        self._episode_finished = True
