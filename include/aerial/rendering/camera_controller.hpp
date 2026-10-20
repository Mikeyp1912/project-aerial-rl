// /include/aerial/rendering/drone_renderer.hpp 

#pragma once

#include <raylib.h>

namespace aerial 
{
    class CameraController {
        public:
            CameraController();

            void Update();

            const Camera3D& GetCamera() const;

            void SetTarget(Vector3 target);

            void Reset();

        private:
            void UpdatePosition();

            Camera3D camera_{};

            Vector3 target_{};

            float distance_ = 2.5f;
            float yaw_ = 0.785398f;
            float pitch_ = 0.45f;
    };
}
