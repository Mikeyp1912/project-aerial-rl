// /apps/simulate.cpp
#include <raylib.h>
#include <algorithm>

#include "aerial/physics/drone_state.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/integrator.hpp"
#include "aerial/physics/motor_commands.hpp"
#include "aerial/physics/motor_dynamics.hpp"
#include "aerial/physics/simulation.hpp"

#include "aerial/rendering/drone_renderer.hpp"
#include "aerial/rendering/camera_controller.hpp"
#include "aerial/rendering/coordinate_conversion.hpp"

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

    aerial::physics::DroneState drone_state;
    aerial::physics::DroneParameters drone_parameters;
    aerial::physics::MotorState motor_state;

    drone_state.position = {0.0, 0.0, 1.0};
    drone_state.velocity = Eigen::Vector3d::Zero();

    

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

    while (!WindowShouldClose()) {
        const double frame_time = std::min(
            static_cast<double>(GetFrameTime()),
            max_frame_time
        );

        accumulator += frame_time;

        while (accumulator >= physics_dt) {
            aerial::physics::StepSimulation(
                drone_state,
                motor_state,
                drone_parameters,
                motor_commands,
                physics_dt
            );

            accumulator -= physics_dt;
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
            "RMB: Orbit | Scroll: Zoom | MMB: Pan | R: Reset | F: Focus",
            20,
            screen_height - 35,
            18,
            DARKGRAY
        );

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
