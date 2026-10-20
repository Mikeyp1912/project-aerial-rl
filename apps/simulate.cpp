
#include <raylib.h>

#include "aerial/rendering/drone_renderer.hpp"

int main()
{
    constexpr int screen_width = 1600;
    constexpr int screen_height = 900;

    InitWindow(
        screen_width,
        screen_height,
        "Project Aerial RL - Simulator"
    );

    SetTargetFPS(60);

    Camera3D camera = {};
    camera.position = {5.0f, 5.0f, 5.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    while (!WindowShouldClose())
    {
        UpdateCamera(&camera, CAMERA_ORBITAL);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        DrawGrid(20, 1.0f);

        aerial::DrawDrone({0.0f, 1.0f, 0.0f});

        EndMode3D();

        DrawText(
            "PROJECT AERIAL RL",
            20,
            20,
            24,
            DARKGRAY
        );

        DrawText(
            "3D Quadcopter Renderer - Milestone 1.3",
            20,
            55,
            18,
            GRAY
        );        

        DrawFPS(20, 90);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}

