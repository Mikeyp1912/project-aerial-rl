

"""Train PPO hover control with randomized aerodynamic wind."""

import argparse
import json
from pathlib import Path

import aerial_sim
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from aerial_rl.environment import AerialDroneEnv


class CheckpointCallback(BaseCallback):
    """Save PPO policies at fixed training-step intervals."""

    def __init__(self, output_dir: Path, interval: int):
        super().__init__()
        self.output_dir = output_dir
        self.interval = interval
        self.next_checkpoint = interval

    def _on_training_start(self) -> None:
        current = self.model.num_timesteps

        self.next_checkpoint = (
            (current // self.interval) + 1
        ) * self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            path = self.output_dir / (
                f"ppo_target_{self.num_timesteps}_steps"
            )

            self.model.save(str(path))
            print(f"\nCheckpoint saved: {path}.zip")

            self.next_checkpoint += self.interval

        return True


def make_environment(
    episode_duration: float,
    horizontal_offset: float,
    altitude_offset: float,
    maximum_wind_speed: float,
    linear_drag: tuple[float, float, float],
    horizontal_action_weight: float,
    altitude_action_weight: float,
    yaw_action_weight: float,
    target_error_scale: float,
    reward_mode: aerial_sim.RewardMode,
    action_repeat: int,
    reward_velocity_weight: float,
):
    def initialize():
        env = AerialDroneEnv(
            maximum_episode_time=episode_duration,
            randomize_initial_state=True,
            maximum_position_offset=(0.5, 0.5, 0.2),
            maximum_velocity=(0.2, 0.2, 0.0),
            pid_target_reference=(
                aerial_sim.PIDTargetReference.WORLD_RELATIVE
            ),
            maximum_horizontal_offset=horizontal_offset,
            hover_centered_altitude=True,
            maximum_altitude_offset=altitude_offset,
            randomize_wind=False,
            maximum_wind_speed=maximum_wind_speed,
            linear_drag=linear_drag,
            horizontal_action_weight=0.0,
            altitude_action_weight=0.0,
            yaw_action_weight=0.0,
            observation_mode=aerial_sim.ObservationMode.HOVER_AUGMENTED,
            task=aerial_sim.FlightTask.TARGET_TRACKING,
            randomize_target=True,
            maximum_target_offset=(2.0, 2.0, 1.0),
            target_error_scale=target_error_scale,
            reward_mode=reward_mode,
            action_repeat=action_repeat,
            reward_velocity_weight=reward_velocity_weight,
        )

        return Monitor(env)

    return initialize


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--episode-duration",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--maximum-wind-speed",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--horizontal-offset",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--altitude-offset",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--linear-drag",
        type=float,
        nargs=3,
        default=[0.1, 0.1, 0.05],
        metavar=("X", "Y", "Z"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        # Output directory
        default=Path(
            "training_results/ppo_target_tracking_200k"
        ),
    )

    parser.add_argument(
        "--horizontal-action-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--altitude-action-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--yaw-action-weight",
        type=float,
        default=0.02,
    )
   
    parser.add_argument(
        "--target-error-scale",
        type=float,
        default=2.0,
        help="Normalization scale for target-relative position errors",
    )

    parser.add_argument(
        "--reward-mode",
        choices=("quadratic", "bounded"),
        default="quadratic",
        help="Reward function used during PPO training",
    )

    parser.add_argument(
        "--reward-velocity-weight",
        type=float,
        default=0.02,
        help="Penalty weight for squared velocity in bounded reward",
    )

    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Physics steps per PPO action",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO discount factor per policy step",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Existing PPO checkpoint to continue training",
    )

    args = parser.parse_args()

    reward_mode = (
        aerial_sim.RewardMode.BOUNDED_TARGET_TRACKING
        if args.reward_mode == "bounded"
        else aerial_sim.RewardMode.QUADRATIC_HOVER
    )

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    if not np.isfinite(args.episode_duration) or args.episode_duration <= 0:
        parser.error("--episode-duration must be finite and positive")

    if args.checkpoint_interval <= 0:
        parser.error("--checkpoint-interval must be positive")

    if not np.isfinite(args.maximum_wind_speed) or args.maximum_wind_speed < 0:
        parser.error("--maximum-wind-speed must be finite and nonnegative")

    if not np.isfinite(args.horizontal_offset) or args.horizontal_offset <= 0:
        parser.error("--horizontal-offset must be finite and positive")

    if not np.isfinite(args.altitude_offset) or args.altitude_offset <= 0:
        parser.error("--altitude-offset must be finite and positive")

    if not np.all(np.isfinite(args.linear_drag)) or any(
        value < 0 for value in args.linear_drag
    ):
        parser.error("--linear-drag must contain nonnegative finite values")

    if (
        not np.isfinite(args.target_error_scale)
        or args.target_error_scale <= 0.0
    ):
        parser.error("--target-error-scale must be finite and positive")

    if args.action_repeat < 1:
        parser.error("--action-repeat must be positive")

    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    
    if (
        not np.isfinite(args.reward_velocity_weight)
        or args.reward_velocity_weight < 0.0
    ):
        parser.error(
            "--reward-velocity-weight must be finite and nonnegative"
        )

    for name in (
        "horizontal_action_weight",
        "altitude_action_weight",
        "yaw_action_weight",
    ):
        value = getattr(args, name)

        if not np.isfinite(value) or value < 0:
            parser.error(
                f"--{name.replace('_', '-')} must be "
                "finite and nonnegative"
            )

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            parser.error(
                f"Checkpoint not found: {args.resume_from}"
            )

        previous_config_path = (
            args.resume_from.parent / "configuration.json"
        )

        if not previous_config_path.is_file():
            parser.error(
                "Resume requires the original configuration.json"
            )

        previous = json.loads(previous_config_path.read_text())

        expected = {
            "episode_duration": args.episode_duration,
            "maximum_horizontal_offset": args.horizontal_offset,
            "maximum_altitude_offset": args.altitude_offset,
            "linear_drag": list(args.linear_drag),
            "target_error_scale": args.target_error_scale,
            "reward_mode": args.reward_mode,
            "action_repeat": args.action_repeat,
            "gamma": args.gamma,
        }

        for key, value in expected.items():
            previous_value = previous.get(
                key,
                0.02 if key == "reward_velocity_weight" else None,
            )

            if previous_value != value:
                parser.error(
                    f"Resume configuration mismatch: {key}. "
                    f"Previous={previous.get(key)}, requested={value}"
                )

    args.output.mkdir(parents=True, exist_ok=True)

    linear_drag = tuple(args.linear_drag)

    configuration = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "episode_duration": args.episode_duration,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "randomize_initial_state": True,
        "maximum_position_offset": [0.5, 0.5, 0.2],
        "maximum_velocity": [0.2, 0.2, 0.0],
        "pid_target_reference": "WORLD_RELATIVE",
        "maximum_horizontal_offset": args.horizontal_offset,
        "hover_centered_altitude": True,
        "maximum_altitude_offset": args.altitude_offset,
        "randomize_wind": False,
        "linear_drag": list(linear_drag),
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": args.gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.0,
        "observation_mode": "HOVER_AUGMENTED",
        "observation_size": 19,
        "task": "TARGET_TRACKING",
        "randomize_target": True,
        "maximum_target_offset": [2.0, 2.0, 1.0],
        "maximum_wind_speed": 0.0,
        "horizontal_action_weight": 0.0,
        "altitude_action_weight": 0.0,
        "yaw_action_weight": 0.0,
        "target_error_scale": args.target_error_scale,
        "reward_mode": args.reward_mode,
        "action_repeat": args.action_repeat,
        "physics_frequency_hz": 120.0,
        "policy_frequency_hz": 120.0 / args.action_repeat,
        "reward_velocity_weight": args.reward_velocity_weight,
    }

    configuration["resume_from"] = (
        str(args.resume_from)
        if args.resume_from is not None
        else None
    )

    configuration["additional_timesteps"] = args.timesteps

    env = DummyVecEnv([
        make_environment(
            args.episode_duration,
            args.horizontal_offset,
            args.altitude_offset,
            args.maximum_wind_speed,
            linear_drag,
            args.horizontal_action_weight,
            args.altitude_action_weight,
            args.yaw_action_weight,
            target_error_scale=args.target_error_scale,
            reward_mode=reward_mode,
            action_repeat=args.action_repeat,
            reward_velocity_weight=args.reward_velocity_weight,
        )
    ])

    expected_observation_shape = (19,)

    if env.observation_space.shape != expected_observation_shape:
        env.close()
        raise RuntimeError(
            "Augmented PPO training requires observation shape "
            f"{expected_observation_shape}, "
            f"got {env.observation_space.shape}"
        )

    try:
        if args.resume_from is not None:
            model = PPO.load(
                str(args.resume_from),
                env=env,
                device="cpu",
                tensorboard_log=str(args.output / "tensorboard"),
            )

            if model.observation_space != env.observation_space:
                raise ValueError("Observation space mismatch")

            if model.action_space != env.action_space:
                raise ValueError("Action space mismatch")

            if not np.isclose(model.gamma, args.gamma):
                raise ValueError("PPO gamma mismatch")

            initial_steps = model.num_timesteps

            print(
                f"\nResuming PPO from {initial_steps:,} steps"
            )

        else:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=5,
                gamma=args.gamma,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.0,
                verbose=1,
                device="cpu",
                seed=args.seed,
                tensorboard_log=str(args.output / "tensorboard"),
            )

            initial_steps = 0

        configuration["initial_timesteps"] = initial_steps
        configuration["requested_total_timesteps"] = (
            initial_steps + args.timesteps
        )

        (args.output / "configuration.json").write_text(
            json.dumps(configuration, indent=4) + "\n"
        )

        callback = CheckpointCallback(
            output_dir=args.output,
            interval=args.checkpoint_interval,
        )

        print("\nProject Aerial RL — PPO Target Tracking Training")
        print("=" * 60)
        print(f"Timesteps:        {args.timesteps:,}")
        print(f"Episode duration: {args.episode_duration:.1f} s")
        print(f"Maximum wind:     {args.maximum_wind_speed:.1f} m/s")
        print(f"Linear drag:      {linear_drag}")
        print(f"Horizontal range: ±{args.horizontal_offset:.2f} m")
        print(f"Altitude range:   ±{args.altitude_offset:.2f} m")
        print(f"Target error scale: {args.target_error_scale:.2f} m")
        print(f"Output:           {args.output}")
        print("=" * 60)

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO_target_tracking",
            reset_num_timesteps=(
                args.resume_from is None
            ),
            progress_bar=True,
        )

        configuration["completed_total_timesteps"] = (
            model.num_timesteps
        )

        (args.output / "configuration.json").write_text(
            json.dumps(configuration, indent=4) + "\n"
        )

        final_path = args.output / "ppo_target_final"
        model.save(str(final_path))

        print(f"\nFinal model saved: {final_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()
