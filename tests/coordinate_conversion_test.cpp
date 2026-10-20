// /tests/coordinate_conversion_test.cpp
#include "aerial/rendering/coordinate_conversion.hpp"

#include <cmath>
#include <iostream>
#include <Eigen/Geometry>
#include <raymath.h>

namespace 
{
    bool Near(float actual, float expected)
    {
        return std::abs(actual - expected) < 1e-6f;
    }

    bool CheckVector(
            const Vector3& actual,
            float x,
            float y,
            float z
            )
    {
        return Near(actual.x, x) && Near(actual.y, y) && Near(actual.z, z);
    }
}

int main() {
    using aerial::rendering::ToRaylib;

    if (!CheckVector(ToRaylib({0.0, 0.0, 1.0}), 0.0f, 1.0f, 0.0f)) {
        std::cerr << "Altitude conversion failed\n";
        return 1;
    }

    if (!CheckVector(ToRaylib({1.0, 2.0, 3.0}), 1.0f, 3.0f, -2.0f)) {
        std::cerr << "Coordinate conversion failed\n";
        return 1;
    }

    if (!CheckVector(ToRaylib({0.0, 0.0, 0.0}), 0.0f, 0.0f, 0.0f)) {
        std::cerr << "Origin conversion failed\n";
        return 1;
    }


    constexpr double pi = 3.14159265358979323846;

    // A 90-degree physics yaw about +Z should become
    // a rotation about raylib +Y.
    const Eigen::Quaterniond yaw(
        Eigen::AngleAxisd(
            pi / 2.0,
            Eigen::Vector3d::UnitZ()
        )
    );

    const Matrix render_rotation = aerial::rendering::ToRaylibRotation(yaw);

    const Vector3 rotated_forward =
        Vector3Transform(
            {1.0f, 0.0f, 0.0f},
            render_rotation
        );

    // Physics +X rotated about +Z becomes physics +Y.
    // Physics +Y maps to raylib -Z.
    if (!CheckVector(
            rotated_forward,
            0.0f,
            0.0f,
            -1.0f
        ))
    {
        std::cerr << "Yaw orientation conversion failed\n";
        return 1;
    }

    // Identity orientation should not rotate the model.
    const Matrix identity_rotation = aerial::rendering::ToRaylibRotation(Eigen::Quaterniond::Identity());

    if (!CheckVector(
            Vector3Transform(
                {1.0f, 0.0f, 0.0f},
                identity_rotation
            ),
            1.0f,
            0.0f,
            0.0f
        ))
    {
        std::cerr << "Identity orientation conversion failed\n";
        return 1;
    }


    std::cout << "Coordinate Conversion tests passed\n";
    return 0;
}
