cmake_minimum_required(VERSION 3.20)

project(
    ProjectAerialRL
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# Dependencies
find_package(Eigen3 REQUIRED)

# raylib
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(raylib)

# Physics library — independent of raylib
add_library(aerial_physics STATIC
    src/physics/integrator.cpp
    src/physics/thrust.cpp
    src/physics/rotational_dynamics.cpp
    src/physics/motor_model.cpp
    src/physics/motor_commands.cpp
    src/physics/motor_mixer.cpp
    src/physics/motor_dynamics.cpp
    src/physics/simulation.cpp
    src/physics/simulation_environment.cpp
    src/physics/aerodynamic_drag.cpp
)

target_include_directories(aerial_physics
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(aerial_physics
    PUBLIC
        Eigen3::Eigen
)

target_compile_options(aerial_physics
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Sensor models
add_library(aerial_sensors STATIC
    src/sensors/sensor_model.cpp
    src/sensors/sensor_scheduler.cpp
)

target_include_directories(aerial_sensors
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(aerial_sensors
    PUBLIC
        Eigen3::Eigen
        aerial_physics
)

target_compile_options(aerial_sensors
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Reinforcement learning environment — headless
add_library(aerial_rl STATIC
    src/rl/flight_environment.cpp
    src/rl/observation.cpp
    src/rl/action_adapter.cpp
    src/rl/hover_reward.cpp
    src/rl/episode_termination.cpp
    src/rl/episode_statistics.cpp
    src/rl/observation_normalizer.cpp
    src/rl/policy.cpp
    src/rl/policy_runner.cpp
    src/rl/policy_evaluator.cpp
    src/rl/neural_network_policy.cpp
    src/rl/adam_optimizer.cpp
    src/rl/batch_trainer.cpp
    src/rl/training_dataset.cpp
    src/rl/epoch_trainer.cpp
    src/rl/imitation_dataset_collector.cpp
)

target_include_directories(aerial_rl
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(aerial_rl
    PUBLIC
        aerial_physics
        aerial_sensors
        aerial_control
)

target_compile_options(aerial_rl
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Flight control library
add_library(aerial_control STATIC
    src/control/attitude_controller.cpp
    src/control/rate_controller.cpp
    src/control/flight_controller.cpp
    src/control/flight_control_mode.cpp
    src/control/altitude_controller.cpp
    src/control/horizontal_velocity_controller.cpp
    src/control/horizontal_position_controller.cpp
    src/control/flight_control_targets.cpp
)

target_include_directories(aerial_control
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(aerial_control
    PUBLIC
        Eigen3::Eigen
        aerial_physics
)

target_compile_options(aerial_control
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Simulator executable
add_executable(simulate
    apps/simulate.cpp
    src/rendering/drone_renderer.cpp
    src/rendering/camera_controller.cpp
)

target_include_directories(simulate
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simulate
    PRIVATE
        aerial_physics
        aerial_control
        Eigen3::Eigen
        raylib
)

target_compile_options(simulate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Headless physics benchmark
add_executable(benchmark_simulation
    apps/benchmark_simulation.cpp
)

target_link_libraries(benchmark_simulation
    PRIVATE
        aerial_physics
)

target_compile_options(benchmark_simulation
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Headless RL policy benchmark
add_executable(benchmark_rl_policies
    apps/benchmark_rl_policies.cpp
)

target_link_libraries(benchmark_rl_policies
    PRIVATE
        aerial_rl
)

target_compile_options(benchmark_rl_policies
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(benchmark_imitation_learning
    apps/benchmark_imitation_learning.cpp
)

target_link_libraries(benchmark_imitation_learning
    PRIVATE aerial_rl
)

target_compile_options(benchmark_imitation_learning
    PRIVATE -Wall -Wextra -Wpedantic
)

# Python bindings
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(aerial_sim
    src/python/aerial_sim_bindings.cpp
)

target_link_libraries(aerial_sim
    PRIVATE
        aerial_rl
)

target_compile_options(aerial_sim
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

set_target_properties(aerial_sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/python"
)

# Tests
include(CTest)

if(BUILD_TESTING)
    add_compile_options(-UNDEBUG)

    add_executable(drone_state_test
        tests/drone_state_test.cpp
    )

    target_include_directories(drone_state_test
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(drone_state_test
        PRIVATE
            Eigen3::Eigen
    )

    target_compile_options(drone_state_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME drone_state
        COMMAND drone_state_test
    )

    add_executable(coordinate_conversion_test
        tests/coordinate_conversion_test.cpp
    )

    target_include_directories(coordinate_conversion_test
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(coordinate_conversion_test
        PRIVATE
            Eigen3::Eigen
            raylib
    )

    target_compile_options(coordinate_conversion_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME coordinate_conversion
        COMMAND coordinate_conversion_test
    )

    add_executable(integrator_test
        tests/integrator_test.cpp
    )

    target_link_libraries(integrator_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(integrator_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME integrator
        COMMAND integrator_test
    )

    add_executable(thrust_test
        tests/thrust_test.cpp
    )

    target_link_libraries(thrust_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(thrust_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME thrust
        COMMAND thrust_test
    )

    add_executable(rotational_dynamics_test
        tests/rotational_dynamics_test.cpp
    )

    target_link_libraries(rotational_dynamics_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(rotational_dynamics_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME rotational_dynamics
        COMMAND rotational_dynamics_test
    )

    add_executable(motor_model_test
        tests/motor_model_test.cpp
    )

    target_link_libraries(motor_model_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(motor_model_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME motor_model
        COMMAND motor_model_test
    )

    add_executable(motor_commands_test
        tests/motor_commands_test.cpp
    )

    target_link_libraries(motor_commands_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(motor_commands_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME motor_commands
        COMMAND motor_commands_test
    )

    add_executable(motor_dynamics_test
        tests/motor_dynamics_test.cpp
    )

    target_link_libraries(motor_dynamics_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(motor_dynamics_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME motor_dynamics
        COMMAND motor_dynamics_test
    )

    add_executable(simulation_test
        tests/simulation_test.cpp
    )

    target_link_libraries(simulation_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(simulation_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME simulation
        COMMAND simulation_test
    )

    add_executable(simulation_environment_test
        tests/simulation_environment_test.cpp
    )

    target_link_libraries(simulation_environment_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(simulation_environment_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME simulation_environment
        COMMAND simulation_environment_test
    )

    add_executable(motor_mixer_test
        tests/motor_mixer_test.cpp
    )

    target_link_libraries(motor_mixer_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(motor_mixer_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME motor_mixer
        COMMAND motor_mixer_test
    )

    add_executable(attitude_controller_test
        tests/attitude_controller_test.cpp
    )

    target_link_libraries(attitude_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(attitude_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME attitude_controller
        COMMAND attitude_controller_test
    )

    add_executable(rate_controller_test
        tests/rate_controller_test.cpp
    )

    target_link_libraries(rate_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(rate_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME rate_controller
        COMMAND rate_controller_test
    )

    add_executable(flight_controller_test
        tests/flight_controller_test.cpp
    )

    target_link_libraries(flight_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(flight_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME flight_controller
        COMMAND flight_controller_test
    )

    add_executable(closed_loop_attitude_test
        tests/closed_loop_attitude_test.cpp
    )

    target_link_libraries(closed_loop_attitude_test
        PRIVATE
            aerial_control
    )

    target_compile_options(closed_loop_attitude_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME closed_loop_attitude
        COMMAND closed_loop_attitude_test
    )

    add_executable(attitude_tracking_test
        tests/attitude_tracking_test.cpp
    )

    target_link_libraries(attitude_tracking_test
        PRIVATE
            aerial_control
            aerial_physics
    )

    target_compile_options(attitude_tracking_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME attitude_tracking
        COMMAND attitude_tracking_test
    )

    add_executable(flight_control_mode_test
        tests/flight_control_mode_test.cpp
    )

    target_link_libraries(flight_control_mode_test
        PRIVATE
            aerial_control
    )

    target_compile_options(flight_control_mode_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME flight_control_mode
        COMMAND flight_control_mode_test
    )

    add_executable(telemetry_history_test
        tests/telemetry_history_test.cpp
    )

    target_include_directories(telemetry_history_test
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_compile_options(telemetry_history_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME telemetry_history
        COMMAND telemetry_history_test
    )

    add_executable(altitude_controller_test
        tests/altitude_controller_test.cpp
    )

    target_link_libraries(altitude_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(altitude_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME altitude_controller
        COMMAND altitude_controller_test
    )

    add_executable(closed_loop_altitude_test
        tests/closed_loop_altitude_test.cpp
    )

    target_link_libraries(closed_loop_altitude_test
        PRIVATE
            aerial_control
            aerial_physics
    )

    target_compile_options(closed_loop_altitude_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME closed_loop_altitude
        COMMAND closed_loop_altitude_test
    )

    add_executable(horizontal_velocity_controller_test
        tests/horizontal_velocity_controller_test.cpp
    )

    target_link_libraries(horizontal_velocity_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(horizontal_velocity_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME horizontal_velocity_controller
        COMMAND horizontal_velocity_controller_test
    )

    add_executable(closed_loop_horizontal_velocity_test
        tests/closed_loop_horizontal_velocity_test.cpp
    )

    target_link_libraries(closed_loop_horizontal_velocity_test
        PRIVATE
            aerial_control
            aerial_physics
    )

    target_compile_options(closed_loop_horizontal_velocity_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME closed_loop_horizontal_velocity
        COMMAND closed_loop_horizontal_velocity_test
    )

    add_executable(horizontal_position_controller_test
        tests/horizontal_position_controller_test.cpp
    )

    target_link_libraries(horizontal_position_controller_test
        PRIVATE
            aerial_control
    )

    target_compile_options(horizontal_position_controller_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME horizontal_position_controller
        COMMAND horizontal_position_controller_test
    )

    add_executable(closed_loop_horizontal_position_test
        tests/closed_loop_horizontal_position_test.cpp
    )

    target_link_libraries(closed_loop_horizontal_position_test
        PRIVATE
            aerial_control
            aerial_physics
    )

    target_compile_options(closed_loop_horizontal_position_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME closed_loop_horizontal_position
        COMMAND closed_loop_horizontal_position_test
    )

    add_executable(flight_control_targets_test
        tests/flight_control_targets_test.cpp
    )

    target_link_libraries(flight_control_targets_test
        PRIVATE aerial_control
    )

    target_compile_options(flight_control_targets_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME flight_control_targets
        COMMAND flight_control_targets_test
    )

    add_executable(disturbance_recovery_test
        tests/disturbance_recovery_test.cpp
    )

    target_link_libraries(disturbance_recovery_test
        PRIVATE aerial_control aerial_physics
    )

    target_compile_options(disturbance_recovery_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME disturbance_recovery
        COMMAND disturbance_recovery_test
    )

    add_executable(aerodynamic_drag_test
        tests/aerodynamic_drag_test.cpp
    )

    target_link_libraries(aerodynamic_drag_test
        PRIVATE aerial_physics
    )

    target_compile_options(aerodynamic_drag_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME aerodynamic_drag
        COMMAND aerodynamic_drag_test
    )

    add_executable(wind_simulation_test
        tests/wind_simulation_test.cpp
    )

    target_link_libraries(wind_simulation_test
        PRIVATE aerial_physics
    )

    target_compile_options(wind_simulation_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME wind_simulation
        COMMAND wind_simulation_test
    )

    add_executable(external_disturbance_test
        tests/external_disturbance_test.cpp
    )

    target_link_libraries(external_disturbance_test
        PRIVATE aerial_physics
    )

    target_compile_options(external_disturbance_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME external_disturbance
        COMMAND external_disturbance_test
    )

    add_executable(closed_loop_disturbance_injection_test
        tests/closed_loop_disturbance_injection_test.cpp
    )

    target_link_libraries(closed_loop_disturbance_injection_test
        PRIVATE aerial_control aerial_physics
    )

    target_compile_options(closed_loop_disturbance_injection_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME closed_loop_disturbance_injection
        COMMAND closed_loop_disturbance_injection_test
    )

    add_executable(sensor_model_test
        tests/sensor_model_test.cpp
    )

    target_link_libraries(sensor_model_test
        PRIVATE aerial_sensors
    )

    target_compile_options(sensor_model_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME sensor_model
        COMMAND sensor_model_test
    )

    add_executable(accelerometer_test
        tests/accelerometer_test.cpp
    )

    target_link_libraries(accelerometer_test
        PRIVATE aerial_sensors
    )

    target_compile_options(accelerometer_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME accelerometer
        COMMAND accelerometer_test
    )

    add_executable(physics_integrated_imu_test
        tests/physics_integrated_imu_test.cpp
    )

    target_link_libraries(physics_integrated_imu_test
        PRIVATE aerial_physics aerial_sensors
    )

    target_compile_options(physics_integrated_imu_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME physics_integrated_imu
        COMMAND physics_integrated_imu_test
    )

    add_executable(sensor_scheduler_test
        tests/sensor_scheduler_test.cpp
    )

    target_link_libraries(sensor_scheduler_test
        PRIVATE aerial_sensors
    )

    target_compile_options(sensor_scheduler_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME sensor_scheduler
        COMMAND sensor_scheduler_test
    )

    add_executable(sensor_driven_flight_test
        tests/sensor_driven_flight_test.cpp
    )

    target_link_libraries(sensor_driven_flight_test
        PRIVATE aerial_control aerial_physics aerial_sensors
    )

    target_compile_options(sensor_driven_flight_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME sensor_driven_flight
        COMMAND sensor_driven_flight_test
    )

    add_executable(flight_environment_test
        tests/flight_environment_test.cpp
    )

    target_link_libraries(flight_environment_test
        PRIVATE aerial_rl
    )

    target_compile_options(flight_environment_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME flight_environment
        COMMAND flight_environment_test
    )

    add_executable(rl_observation_test
        tests/rl_observation_test.cpp
    )

    target_link_libraries(rl_observation_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_observation_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_observation
        COMMAND rl_observation_test
    )

    add_executable(rl_action_adapter_test
        tests/rl_action_adapter_test.cpp
    )

    target_link_libraries(rl_action_adapter_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_action_adapter_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_action_adapter
        COMMAND rl_action_adapter_test
    )

    add_executable(rl_hover_reward_test
        tests/rl_hover_reward_test.cpp
    )

    target_link_libraries(rl_hover_reward_test
        PRIVATE
            aerial_rl
    )

    target_compile_options(rl_hover_reward_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME rl_hover_reward
        COMMAND rl_hover_reward_test
    )

    add_executable(rl_episode_termination_test
        tests/rl_episode_termination_test.cpp
    )

    target_link_libraries(rl_episode_termination_test
        PRIVATE
            aerial_rl
    )

    target_compile_options(rl_episode_termination_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME rl_episode_termination
        COMMAND rl_episode_termination_test
    )

    add_executable(rl_episode_statistics_test
        tests/rl_episode_statistics_test.cpp
    )

    target_link_libraries(rl_episode_statistics_test
        PRIVATE
            aerial_rl
    )

    target_compile_options(rl_episode_statistics_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -UNDEBUG
    )

    add_test(
        NAME rl_episode_statistics
        COMMAND rl_episode_statistics_test
    )

    add_executable(rl_observation_normalizer_test
        tests/rl_observation_normalizer_test.cpp
    )

    target_link_libraries(rl_observation_normalizer_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_observation_normalizer_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_observation_normalizer
        COMMAND rl_observation_normalizer_test
    )

    add_executable(rl_policy_test
        tests/rl_policy_test.cpp
    )

    target_link_libraries(rl_policy_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_policy_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_policy
        COMMAND rl_policy_test
    )

    add_executable(rl_policy_evaluator_test
        tests/rl_policy_evaluator_test.cpp
    )

    target_link_libraries(rl_policy_evaluator_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_policy_evaluator_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_policy_evaluator
        COMMAND rl_policy_evaluator_test
    )

    add_executable(rl_neural_network_policy_test
        tests/rl_neural_network_policy_test.cpp
    )

    target_link_libraries(rl_neural_network_policy_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_neural_network_policy_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_neural_network_policy
        COMMAND rl_neural_network_policy_test
    )

    add_executable(rl_neural_network_parameters_test
        tests/rl_neural_network_parameters_test.cpp
    )

    target_link_libraries(rl_neural_network_parameters_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_neural_network_parameters_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_neural_network_parameters
        COMMAND rl_neural_network_parameters_test
    )

    add_executable(rl_neural_network_gradient_test
        tests/rl_neural_network_gradient_test.cpp
    )

    target_link_libraries(rl_neural_network_gradient_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_neural_network_gradient_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_neural_network_gradient
        COMMAND rl_neural_network_gradient_test
    )

    add_executable(rl_batch_training_test
        tests/rl_batch_training_test.cpp
    )

    target_link_libraries(rl_batch_training_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_batch_training_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_batch_training
        COMMAND rl_batch_training_test
    )

    add_executable(rl_training_dataset_test
        tests/rl_training_dataset_test.cpp
    )

    target_link_libraries(rl_training_dataset_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_training_dataset_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_training_dataset
        COMMAND rl_training_dataset_test
    )

    add_executable(rl_imitation_dataset_test
        tests/rl_imitation_dataset_test.cpp
    )

    target_link_libraries(rl_imitation_dataset_test
        PRIVATE aerial_rl
    )

    target_compile_options(rl_imitation_dataset_test
        PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG
    )

    add_test(
        NAME rl_imitation_dataset
        COMMAND rl_imitation_dataset_test
    )

endif()
