// /src/rendering/drone_renderer.cpp 

#include "aerial/rendering/drone_renderer.hpp"

#include <cmath>
#include <rlgl.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <raymath.h>
#pragma GCC diagnostic pop

namespace aerial
{

    void DrawDrone(Vector3 position, Matrix orientation)
    {
        rlPushMatrix();
        rlTranslatef(position.x, position.y, position.z);
        rlMultMatrixf(MatrixToFloat(orientation));

        // Geometry is now defined aroun the local origin.
        position = {0.0f, 0.0f, 0.0f};

        // Dimensions in metres
        constexpr float arm_length = 0.16f;
        constexpr float arm_radius = 0.012f;
        constexpr float motor_radius = 0.018f;
        constexpr float motor_height = 0.025f;
        constexpr float prop_radius = 0.0635f;

        constexpr float pi = 3.14159265358979323846f;

        // Central body
        DrawCube(position, 0.12f, 0.035f, 0.09f, DARKGRAY);

        // Four arms in X configuration
        for (int i = 0; i < 4; ++i)
        {
            float angle = pi / 4.0f + i * pi / 2.0f;

            float dx = std::cos(angle) * arm_length;
            float dz = std::sin(angle) * arm_length;

            Vector3 motor_position = {
                position.x + dx,
                position.y,
                position.z + dz
            };

            // Arm
            DrawCylinderEx(
                position,
                motor_position,
                arm_radius,
                arm_radius,
                8,
                DARKGRAY
            );

            // Motor housing
            Vector3 motor_top = motor_position;
            motor_top.y += motor_height;

            DrawCylinderEx(
                motor_position,
                motor_top,
                motor_radius,
                motor_radius,
                16,
                GRAY
            );

            // Propeller disc
            Vector3 prop_position = motor_top;
            prop_position.y += 0.005f;

            DrawCylinder(
                prop_position,
                prop_radius,
                prop_radius,
                0.003f,
                24,
                Fade(SKYBLUE, 0.45f)
            );

            // Front motor indicators
            if (dx > 0.0f)
            {
                DrawSphere(
                    motor_top,
                    0.008f,
                    RED
                );
            }
        }

        // Top plate
        Vector3 top_position = position;
        top_position.y += 0.025f;

        DrawCube(
            top_position,
            0.09f,
            0.01f,
            0.07f,
            BLACK
        );

        rlPopMatrix();
    }

}

