// /apps/simulate.cpp
#include <raylib.h>
#include <algorithm>

#include "aerial/physics/drone_state.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/integrator.hpp"
#include "aerial/physics/thrust.hpp"
#include "aerial/physics/rotational_dynamics.hpp"
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

    const double total_thrust = drone_parameters.mass * aerial::physics::gravity;

    drone_state.position = {0.0, 0.0, 1.0};
    drone_state.velocity = Eigen::Vector3d::Zero();

    const Eigen::Vector3d applied_torque(0.002, 0.0, 0.0);

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
            // Calculate thrust using the current orientation.
            const Eigen::Vector3d thrust_force =
                aerial::physics::CalculateThrustForce(
                    drone_state,
                    total_thrust
                );

            // Integrate translational motion.
            aerial::physics::Integrate(
                drone_state,
                drone_parameters,
                thrust_force,
                physics_dt
            );

            // Integrate rotational motion.
            aerial::physics::IntegrateRotation(
                drone_state,
                drone_parameters,
                applied_torque,
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
