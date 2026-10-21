// /tests/thrust_test.cpp


#include <cmath>
#include <iostream>
#include <numbers>

#include <Eigen/Geometry>

#include "aerial/physics/thrust.hpp"

namespace
{
    bool Near(double a, double b)
    {
        return std::abs(a - b) < 1e-9;
    }

    bool CheckVector(
        const Eigen::Vector3d& actual,
        const Eigen::Vector3d& expected
    )
    {
        return (actual - expected).norm() < 1e-9;
    }
}

int main()
{
    using aerial::physics::DroneState;
    using aerial::physics::CalculateThrustForce;

    constexpr double thrust = 10.0;

    // Level drone: thrust points upward.
    DroneState level;

    if (!CheckVector(
        CalculateThrustForce(level, thrust),
        Eigen::Vector3d(0.0, 0.0, 10.0)))
    {
        std::cerr << "Level thrust test failed\n";
        return 1;
    }

    // 90-degree rotation around body/world Y:
    // body +Z maps to world +X.
    DroneState pitched;

    pitched.orientation = Eigen::Quaterniond(
        Eigen::AngleAxisd(
            std::numbers::pi / 2.0,
            Eigen::Vector3d::UnitY()
        )
    );

    if (!CheckVector(
        CalculateThrustForce(pitched, thrust),
        Eigen::Vector3d(10.0, 0.0, 0.0)))
    {
        std::cerr << "Pitch thrust test failed\n";
        return 1;
    }

    // 90-degree rotation around X:
    // body +Z maps to world -Y.
    DroneState rolled;

    rolled.orientation = Eigen::Quaterniond(
        Eigen::AngleAxisd(
            std::numbers::pi / 2.0,
            Eigen::Vector3d::UnitX()
        )
    );

    if (!CheckVector(
        CalculateThrustForce(rolled, thrust),
        Eigen::Vector3d(0.0, -10.0, 0.0)))
    {
        std::cerr << "Roll thrust test failed\n";
        return 1;
    }

    // Yaw around Z should not change vertical thrust.
    DroneState yawed;

    yawed.orientation = Eigen::Quaterniond(
        Eigen::AngleAxisd(
            std::numbers::pi / 2.0,
            Eigen::Vector3d::UnitZ()
        )
    );

    if (!CheckVector(
        CalculateThrustForce(yawed, thrust),
        Eigen::Vector3d(0.0, 0.0, 10.0)))
    {
        std::cerr << "Yaw thrust test failed\n";
        return 1;
    }

    // Zero thrust should produce zero force.
    if (!Near(CalculateThrustForce(level, 0.0).norm(), 0.0))
    {
        std::cerr << "Zero thrust test failed\n";
        return 1;
    }

    std::cout << "Thrust tests passed\n";
    return 0;
}
