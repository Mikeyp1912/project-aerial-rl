// /src/rendering/camera_controller.cpp

#include "aerial/rendering/camera_controller.hpp"

#include <algorithm>
#include <cmath>

namespace aerial
{
    CameraController::CameraController()
    {
        Reset();
    }

    void CameraController::Reset()
    {
        target_ = {0.0f, 1.0f, 0.0f};

        // Elevated rear three-quarter view.
        distance_ = 2.5f;
        yaw_ = -1.15f;
        pitch_ = 0.55f;

        camera_.up = {0.0f, 1.0f, 0.0f};
        camera_.fovy = 45.0f;
        camera_.projection = CAMERA_PERSPECTIVE;

        UpdatePosition();
    }

    void CameraController::SetTarget(Vector3 target)
    {
        target_ = target;
        UpdatePosition();
    }

    const Camera3D& CameraController::GetCamera() const
    {
        return camera_;
    }
    
    void CameraController::UpdatePosition()
    {
        const float horizontal_distance = distance_ * std::cos(pitch_);

        camera_.position = {
            target_.x + horizontal_distance * std::sin(yaw_),
            target_.y + distance_ * std::sin(pitch_),
            target_.z + horizontal_distance * std::cos(yaw_)
        };

        camera_.target = target_;
    }

    void CameraController::Update()
    {
        constexpr float rotation_speed = 0.005f;
        constexpr float pan_speed = 0.001f;

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();

            yaw_ -= delta.x * rotation_speed;
            pitch_ += delta.y *rotation_speed;

            pitch_ = std::clamp(pitch_, -1.4f, 1.4f);
        }

        float wheel = GetMouseWheelMove();

        if (wheel != 0.0f) {
            distance_ *= std::pow(0.9f, wheel);
            distance_ = std::clamp(distance_, 0.3f, 20.0f);
        }

        if (IsMouseButtonDown(MOUSE_MIDDLE_BUTTON)) {
            Vector2 delta = GetMouseDelta();

            const float scale = distance_ * pan_speed;

            Vector3 right = {
                std::cos(yaw_),
                0.0f,
                -std::sin(yaw_)
            };

            Vector3 up = {0.0f, 1.0f, 0.0f};

            target_.x -= right.x * delta.x * scale;
            target_.z -= right.z * delta.x * scale;
            target_.y += up.y * delta.y *scale;
        }

        if (IsKeyPressed(KEY_R)) {
            Reset();
        }

        if (IsKeyPressed(KEY_F)) {
            SetTarget({0.0f, 1.0f, 0.0f});
        }

        UpdatePosition();
    }
}
