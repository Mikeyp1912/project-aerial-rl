// /apps/simulate.cpp
#include <raylib.h>
#include <algorithm>
#include <cstddef>
#include <cmath>

#include "aerial/control/flight_controller.hpp"
#include "aerial/control/flight_control_mode.hpp"
#include "aerial/control/altitude_controller.hpp"

#include "aerial/physics/integrator.hpp"
#include "aerial/physics/simulation_environment.hpp"

#include "aerial/rendering/drone_renderer.hpp"
#include "aerial/rendering/camera_controller.hpp"
#include "aerial/rendering/coordinate_conversion.hpp"

#include "aerial/telemetry/telemetry_history.hpp"

int main() {
    constexpr int screen_width = 1600;
    constexpr int screen_height = 900;

    InitWindow(
            screen_width,
            screen_height,
            "Project Aerial RL - Simulator"
            );

    SetTargetFPS(60);

    aerial::CameraController camera_controller;

    aerial::physics::Simulation simulation;

    const auto& drone_parameters = simulation.GetParameters();

    const double hover_command =
        drone_parameters.mass * aerial::physics::gravity /
        (4.0 * drone_parameters.max_motor_thrust);

    // Small roll input while preserving total thrust.
    constexpr double roll_input = 0.002;

    const aerial::physics::MotorCommands motor_commands = {
        hover_command + roll_input,
        hover_command + roll_input,
        hover_command - roll_input,
        hover_command - roll_input
    };

    constexpr double physics_dt = 1.0 / 120.0;
    constexpr double max_frame_time = 0.25;

    double accumulator = 0.0;

    bool paused = false;

    aerial::control::FlightControlModeState flight_control;
    aerial::control::FlightControllerParameters controller_parameters;

    aerial::control::FlightControllerOutput controller_output;

    aerial::control::AltitudeControllerParameters altitude_parameters;
    aerial::control::AltitudeControllerOutput altitude_output;

    double target_altitude = simulation.GetDroneState().position.z();

    constexpr double altitude_step = 0.5;
    constexpr double minimum_target_altitude = 0.0;
    constexpr double maximum_target_altitude = 100.0;

    bool altitude_output_valid = false;

    bool controller_output_valid = false;

    constexpr double pi = 3.14159265358979323846;
    constexpr double degrees_to_radians = pi / 180.0;

    double target_roll = 0.0;
    double target_pitch = 0.0;
    double target_yaw = 0.0;

    constexpr double attitude_step = 5.0 * degrees_to_radians;
    constexpr double maximum_tilt = 20.0 * degrees_to_radians; 

    struct RollTelemetrySample
    {
        float actual_degrees = 0.0f;
        float target_degrees = 0.0f;
    };

    constexpr std::size_t history_size = 600;

    aerial::telemetry::TelemetryHistory<RollTelemetrySample, history_size> roll_history;

    const double hover_thrust = aerial::control::CalculateHoverThrust(drone_parameters);

    while (!WindowShouldClose()) {
        const double frame_time = std::min(
            static_cast<double>(GetFrameTime()),
            max_frame_time
        );

        // Pause or resume physics
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;

            // Discard any unconsumed partial timestep
            accumulator = 0.0;
        }

        // Toggle between open-loop and attitude-hold control
        if (IsKeyPressed(KEY_M)) {
            aerial::control::ToggleFlightControlMode(flight_control);

            if (flight_control.mode == aerial::control::FlightControlMode::AltitudeHold) {
                // Capture current altitude when entering altitude hold
                target_altitude = simulation.GetDroneState().position.z();
            }

            controller_output_valid = false;
            altitude_output_valid = false;

            accumulator = 0.0;
        }

        // Adjust target orientation while attitude hold is enabled.
        if (flight_control.mode != aerial::control::FlightControlMode::OpenLoop) {
            if (IsKeyPressed(KEY_LEFT)) {
                target_roll = std::clamp(
                    target_roll - attitude_step,
                    -maximum_tilt,
                    maximum_tilt
                );
            }

            if (IsKeyPressed(KEY_RIGHT)) {
                target_roll = std::clamp(
                    target_roll + attitude_step,
                    -maximum_tilt,
                    maximum_tilt
                );
            }

            if (IsKeyPressed(KEY_UP)) {
                target_pitch = std::clamp(
                    target_pitch + attitude_step,
                    -maximum_tilt,
                    maximum_tilt
                );
            }

            if (IsKeyPressed(KEY_DOWN)) {
                target_pitch = std::clamp(
                    target_pitch - attitude_step,
                    -maximum_tilt,
                    maximum_tilt
                );
            }

            if (IsKeyPressed(KEY_Q)) {
                target_yaw -= attitude_step;
            }

            if (IsKeyPressed(KEY_E)) {
                target_yaw += attitude_step;
            }

            // Return target orientation to level.
            if (IsKeyPressed(KEY_L)) {
                target_roll = 0.0;
                target_pitch = 0.0;
            }
        }

        // Adjust altitude target only in altitude-hold mode.
        if (flight_control.mode == aerial::control::FlightControlMode::AltitudeHold) {

            if (IsKeyPressed(KEY_PAGE_UP)) {
                target_altitude = std::clamp(
                    target_altitude + altitude_step,
                    minimum_target_altitude,
                    maximum_target_altitude
                );
            }

            if (IsKeyPressed(KEY_PAGE_DOWN)) {
                target_altitude = std::clamp(
                    target_altitude - altitude_step,
                    minimum_target_altitude,
                    maximum_target_altitude
                );
            }
        }

        // Reset the simulation to its initial state
        if (IsKeyPressed(KEY_R)) {
            simulation.Reset();

            aerial::control::ResetFlightController(flight_control.controller_state);

            target_roll = 0.0;
            target_pitch = 0.0;
            target_yaw = 0.0;

            roll_history.Clear();

            controller_output_valid = false;

            target_altitude = simulation.GetDroneState().position.z();

            altitude_output = {};
            altitude_output_valid = false;

            accumulator = 0.0;
        }

        // Advance physics and controller together.
        const auto step_simulation = [&]() {
            aerial::physics::MotorCommands commands = motor_commands;

            const Eigen::Quaterniond target_orientation =
                Eigen::Quaterniond(
                        Eigen::AngleAxisd(
                            target_yaw,
                            Eigen::Vector3d::UnitZ()
                            )
                        ) * 
                Eigen::Quaterniond(
                        Eigen::AngleAxisd(
                            target_pitch,
                            Eigen::Vector3d::UnitY()
                            )
                        ) * 
                Eigen::Quaterniond(
                        Eigen::AngleAxisd(
                            target_roll,
                            Eigen::Vector3d::UnitX()
                            )
                        );


            if (flight_control.mode != aerial::control::FlightControlMode::OpenLoop) {

                double collective_thrust = hover_thrust;

                if (flight_control.mode == aerial::control::FlightControlMode::AltitudeHold) {

                    altitude_output =
                        aerial::control::UpdateAltitudeController(
                            simulation.GetDroneState(),
                            drone_parameters,
                            target_altitude,
                            altitude_parameters
                        );

                    collective_thrust = altitude_output.collective_thrust;
                    altitude_output_valid = true;
                }
                else {
                    altitude_output_valid = false;
                }

                controller_output =
                    aerial::control::UpdateFlightController(
                        simulation.GetDroneState(),
                        drone_parameters,
                        target_orientation,
                        collective_thrust,
                        physics_dt,
                        controller_parameters,
                        flight_control.controller_state
                    );

                controller_output_valid = true;

                commands = controller_output.motor_commands;
            }

            simulation.Step(commands, physics_dt);
        };

        // Single-step while paused.
        if (paused) {
            if (IsKeyPressed(KEY_N)) {
                step_simulation();
            }
        }
        else {
            accumulator += frame_time;

            while (accumulator >= physics_dt) {
                step_simulation();
                accumulator -= physics_dt;
            }
        }
        
        const auto& drone_state = simulation.GetDroneState();

        const Eigen::Matrix3d rotation = drone_state.orientation.toRotationMatrix();

        // ZYX yaw-pitch-roll decomposition.
        const double actual_roll = std::atan2(
            rotation(2, 1),
            rotation(2, 2)
        );

        const double actual_pitch = std::asin(
            std::clamp(-rotation(2, 0), -1.0, 1.0)
        );

        const double actual_yaw = std::atan2(
            rotation(1, 0),
            rotation(0, 0)
        );

        const double roll_error = target_roll - actual_roll;
        const double pitch_error = target_pitch - actual_pitch;

        if (!paused && simulation.GetStepCount() > 0) {
            roll_history.Push({
                static_cast<float>(actual_roll / degrees_to_radians),
                static_cast<float>(target_roll / degrees_to_radians)
            });
        }

        camera_controller.Update();

        BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera_controller.GetCamera());

        // Ground reference grid
        DrawGrid(20, 1.0f);

        // Drone representation
        aerial::DrawDrone(
                aerial::rendering::ToRaylib(drone_state.position),
                aerial::rendering::ToRaylibRotation(drone_state.orientation)
                );

        EndMode3D();

        DrawText(
                "PROJECT AERIAL RL",
                20,
                20,
                24,
                DARKGRAY
                );

        DrawFPS(20, 90);

        DrawText(
            TextFormat(
                "Position: X %.2f | Y %.2f | Z %.2f",
                drone_state.position.x(),
                drone_state.position.y(),
                drone_state.position.z()
            ),
            20,
            120,
            20,
            DARKGRAY
        );

        DrawText(
            TextFormat(
                "Angular velocity: X %.2f | Y %.2f | Z %.2f",
                drone_state.angular_velocity.x(),
                drone_state.angular_velocity.y(),
                drone_state.angular_velocity.z()
            ),
            20,
            150,
            20,
            DARKGRAY
        );

        DrawText(
            TextFormat(
                "Target: Roll %.1f | Pitch %.1f | Yaw %.1f deg",
                target_roll / degrees_to_radians,
                target_pitch / degrees_to_radians,
                target_yaw / degrees_to_radians
            ),
            20,
            270,
            20,
            DARKGRAY
        );

        DrawText(
            TextFormat(
                "Actual: Roll %.1f | Pitch %.1f | Yaw %.1f deg",
                actual_roll / degrees_to_radians,
                actual_pitch / degrees_to_radians,
                actual_yaw / degrees_to_radians
            ),
            20,
            300,
            20,
            DARKGRAY
        );

        DrawText(
            TextFormat(
                "Attitude error: Roll %.1f | Pitch %.1f deg",
                roll_error / degrees_to_radians,
                pitch_error / degrees_to_radians
            ),
            20,
            330,
            20,
            DARKGRAY
        );

        if (flight_control.mode != aerial::control::FlightControlMode::OpenLoop && controller_output_valid) {
            const auto& rate =
                controller_output.desired_angular_rate;

            const auto& torque =
                controller_output.requested_body_torque;

            DrawText(
                TextFormat(
                    "Desired rate: X %.2f | Y %.2f | Z %.2f rad/s",
                    rate.x(),
                    rate.y(),
                    rate.z()
                ),
                20,
                370,
                20,
                DARKGRAY
            );

            DrawText(
                TextFormat(
                    "Requested torque: X %.3f | Y %.3f | Z %.3f Nm",
                    torque.x(),
                    torque.y(),
                    torque.z()
                ),
                20,
                400,
                20,
                DARKGRAY
            );

            DrawText(
                TextFormat(
                    "Motor saturation: %s",
                    controller_output.motor_saturated ? "YES" : "NO"
                ),
                20,
                430,
                20,
                controller_output.motor_saturated ? RED : DARKGREEN
            );
        }

        if (flight_control.mode == aerial::control::FlightControlMode::AltitudeHold &&
            altitude_output_valid) {

            const double altitude_error =
                target_altitude - drone_state.position.z();

            DrawText(
                TextFormat(
                    "Altitude: %.2f m | Target: %.2f m | Error: %.2f m",
                    drone_state.position.z(),
                    target_altitude,
                    altitude_error
                ),
                20,
                470,
                20,
                DARKGRAY
            );

            DrawText(
                TextFormat(
                    "Vertical velocity: %.2f m/s | Target: %.2f m/s",
                    drone_state.velocity.z(),
                    altitude_output.desired_vertical_velocity
                ),
                20,
                500,
                20,
                DARKGRAY
            );

            DrawText(
                TextFormat(
                    "Collective thrust: %.2f N | Saturated: %s",
                    altitude_output.collective_thrust,
                    altitude_output.saturated ? "YES" : "NO"
                ),
                20,
                530,
                20,
                altitude_output.saturated ? RED : DARKGREEN
            );
        }

        DrawText(
            TextFormat(
                "Simulation: %s",
                paused ? "PAUSED" : "RUNNING"
            ),
            20,
            180,
            20,
            paused ? ORANGE : DARKGREEN
        );

        const char* mode_name = "OPEN LOOP";

        switch (flight_control.mode) {
            case aerial::control::FlightControlMode::OpenLoop:
                mode_name = "OPEN LOOP";
                break;

            case aerial::control::FlightControlMode::AttitudeHold:
                mode_name = "ATTITUDE HOLD";
                break;

            case aerial::control::FlightControlMode::AltitudeHold:
                mode_name = "ALTITUDE HOLD";
                break;
        }

        DrawText(
            TextFormat("Control mode: %s", mode_name),
            20,
            240,
            20,
            flight_control.mode ==
                aerial::control::FlightControlMode::OpenLoop
                    ? DARKGRAY
                    : DARKGREEN
        );

        DrawText(
            TextFormat(
                "Simulation time: %.3f s | Steps: %llu",
                simulation.GetSimulationTime(),
                static_cast<unsigned long long>(
                    simulation.GetStepCount()
                )
            ),
            20,
            210,
            20,
            DARKGRAY
        );

        DrawText(
            "SPACE: Pause | N: Step | R: Reset | M: Mode | Arrows: Roll/Pitch | Q/E: Yaw | L: Level | T: Camera Reset | F: Focus",
            20,
            screen_height - 60,
            18,
            DARKGRAY
        );

        DrawText(
            "RMB: Orbit | Scroll: Zoom | MMB: Pan",
            20,
            screen_height - 35,
            18,
            DARKGRAY
        );

        // Attitude response graph.
        constexpr float graph_x = 1000.0f;
        constexpr float graph_y = 80.0f;
        constexpr float graph_width = 550.0f;
        constexpr float graph_height = 280.0f;

        constexpr float graph_range = 30.0f;

        DrawRectangle(
            static_cast<int>(graph_x),
            static_cast<int>(graph_y),
            static_cast<int>(graph_width),
            static_cast<int>(graph_height),
            Fade(RAYWHITE, 0.92f)
        );

        DrawRectangleLines(
            static_cast<int>(graph_x),
            static_cast<int>(graph_y),
            static_cast<int>(graph_width),
            static_cast<int>(graph_height),
            LIGHTGRAY
        );

        DrawText(
            "ROLL RESPONSE",
            static_cast<int>(graph_x + 15.0f),
            static_cast<int>(graph_y + 15.0f),
            20,
            DARKGRAY
        );

        // Horizontal zero-degree reference.
        const float zero_y = graph_y + graph_height * 0.5f;

        DrawLine(
            static_cast<int>(graph_x),
            static_cast<int>(zero_y),
            static_cast<int>(graph_x + graph_width),
            static_cast<int>(zero_y),
            LIGHTGRAY
        );

        // Convert degrees to graph Y coordinate.
        const auto roll_to_y = [&](float roll_degrees) {
            const float normalized = std::clamp(
                roll_degrees / graph_range,
                -1.0f,
                1.0f
            );

            return zero_y - normalized * graph_height * 0.45f;
        };

        // Draw history oldest to newest.
        if (roll_history.Size() >= 2) {
            for (std::size_t i = 1; i < roll_history.Size(); ++i) {
                const auto& previous = roll_history.At(i - 1);
                const auto& current = roll_history.At(i);

                const float x0 =
                    graph_x +
                    static_cast<float>(i - 1) /
                    static_cast<float>(history_size - 1) * graph_width;

                const float x1 =
                    graph_x +
                    static_cast<float>(i) /
                    static_cast<float>(history_size - 1) * graph_width;

                DrawLineV(
                    {x0, roll_to_y(previous.actual_degrees)},
                    {x1, roll_to_y(current.actual_degrees)},
                    BLUE
                );

                DrawLineV(
                    {x0, roll_to_y(previous.target_degrees)},
                    {x1, roll_to_y(current.target_degrees)},
                    RED
                );
            }
        }

        DrawText(
            "Blue: Actual    Red: Target",
            static_cast<int>(graph_x + 15.0f),
            static_cast<int>(graph_y + graph_height - 30.0f),
            16,
            DARKGRAY
        );

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
