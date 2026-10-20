// /include/aerial/rendering/coordinate_conversion.hpp

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <raylib.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <raymath.h>
#pragma GCC diagnostic pop

namespace aerial::rendering
{
    inline Vector3 ToRaylib(const Eigen::Vector3d& position)
    {
        return {
            static_cast<float>(position.x()),
            static_cast<float>(position.z()),
            static_cast<float>(-position.y()),
        };
    };

    inline Matrix ToRaylibRotation(const Eigen::Quaterniond& orientation)
    {
        // Physics Z-up -> raylib Y-up
        Eigen::Matrix3d conversion;
        conversion << 
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, -1.0, 0.0;

        // Change the basis of the body-to-world rotation
        const Eigen::Matrix3d rotation = conversion * orientation.normalized().toRotationMatrix() * conversion.transpose();

        const Eigen::Quaterniond render_quaternion(rotation);

        const Quaternion raylib_quaternion = {
            static_cast<float>(render_quaternion.x()),
            static_cast<float>(render_quaternion.y()),
            static_cast<float>(render_quaternion.z()),
            static_cast<float>(render_quaternion.w())
        };

        return QuaternionToMatrix(raylib_quaternion);
    }
}
