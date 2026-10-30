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
        randomize_initial_state: bool = False,
        maximum_position_offset: tuple[float, float, float] = (
            0.5, 0.5, 0.2
        ),
        maximum_velocity: tuple[float, float, float] = (
            0.2, 0.2, 0.0
        ),
        pid_target_reference: aerial_sim.PIDTargetReference = (
            aerial_sim.PIDTargetReference.EPISODE_RELATIVE
        ),
        maximum_horizontal_offset: float = 5.0,
        hover_centered_altitude: bool = False,
        maximum_altitude_offset: float = 1.0,
        randomize_wind: bool = False,
        maximum_wind_speed: float = 5.0,
        linear_drag: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()

        self.simulator = aerial_sim.FlightEnvironment(
            time_step=time_step,
            maximum_episode_time=maximum_episode_time,
            action_mode=action_mode,
            randomize_initial_state=randomize_initial_state,
            maximum_position_offset=maximum_position_offset,
            maximum_velocity=maximum_velocity,
            pid_target_reference=pid_target_reference,
            maximum_horizontal_offset=maximum_horizontal_offset,
            hover_centered_altitude=hover_centered_altitude,
            maximum_altitude_offset=maximum_altitude_offset,
            randomize_wind=randomize_wind,
            maximum_wind_speed=maximum_wind_speed,
            linear_drag=linear_drag,
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
            {
                "initial_position": np.asarray(
                    self.simulator.get_position(),
                    dtype=np.float64,
                ),
                "initial_velocity": np.asarray(
                    self.simulator.get_velocity(),
                    dtype=np.float64,
                ),
                "wind_velocity": np.asarray(
                    self.simulator.get_wind_velocity(),
                    dtype=np.float64,
                ),
            },
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
            "wind_velocity": np.asarray(
                self.simulator.get_wind_velocity(),
                dtype=np.float64,
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
