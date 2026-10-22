// /tests/attitude_controller_test.cpp

#include "aerial/control/attitude_controller.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace aerial::control;

namespace
{
    constexpr double tolerance = 1e-9;
    constexpr double pi = 3.14159265358979323846;

    bool NearlyEqual(double a, double b)
    {
        return std::abs(a - b) < tolerance;
    }

    Eigen::Quaterniond Rotation(
            const Eigen::Vector3d& axis,
            double angle
            )
    {
        return Eigen::Quaterniond(
                Eigen::AngleAxisd(angle, axis)
                );
    }
}

int main()
{
    AttitudeControllerParameters parameters;
    parameters.kp = 4.0;
    parameters.max_angular_rate = 10.0;

    const Eigen::Quaterniond identity =
        Eigen::Quaterniond::Identity();

    // 1. Identical orientations
    {
        const auto rate = CalculateDesiredAngularRate(
                identity, identity, parameters
                );

        assert(rate.norm() < tolerance);
    }

    // 2. Positive roll
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitX(), 0.1
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, parameters
                );

        assert(NearlyEqual(rate.x(), 0.4));
        assert(NearlyEqual(rate.y(), 0.0));
        assert(NearlyEqual(rate.z(), 0.0));
    }

    // 3. Positive pitch
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitY(), 0.2
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, parameters
                );

        assert(NearlyEqual(rate.x(), 0.0));
        assert(NearlyEqual(rate.y(), 0.8));
        assert(NearlyEqual(rate.z(), 0.0));
    }

    // 4. Positive yaw
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitZ(), 0.15
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, parameters
                );

        assert(NearlyEqual(rate.x(), 0.0));
        assert(NearlyEqual(rate.y(), 0.0));
        assert(NearlyEqual(rate.z(), 0.6));
    }

    // 5. Negative roll
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitX(), -0.2
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, parameters
                );

        assert(NearlyEqual(rate.x(), -0.8));
    }

    // 6. Quaternion sign equivalence
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitZ(), 0.7
                );

        Eigen::Quaterniond negated = target;
        negated.coeffs() *= -1.0;

        const auto rate = CalculateDesiredAngularRate(
                target, negated, parameters
                );

        assert(rate.norm() < tolerance);
    }

    // 7. Shortest rotation
    {
        const auto target = Rotation(
                Eigen::Vector3d::UnitZ(),
                3.0 * pi / 2.0
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, parameters
                );

        // 270 degrees should become -90 degrees.
        assert(NearlyEqual(rate.z(), -2.0 * pi));
    }

    // 8. Rate limiting
    {
        AttitudeControllerParameters limited = parameters;
        limited.max_angular_rate = 1.0;

        const auto target = Rotation(
                Eigen::Vector3d::UnitX(), 1.0
                );

        const auto rate = CalculateDesiredAngularRate(
                identity, target, limited
                );

        assert(NearlyEqual(rate.norm(), 1.0));
        assert(rate.x() > 0.0);
    }

    // 9. Non-identity current orientation
    {
        const auto current = Rotation(
                Eigen::Vector3d::UnitZ(), 0.5
                );

        const auto target = current * Rotation(
                Eigen::Vector3d::UnitX(), 0.1
                );

        const auto rate = CalculateDesiredAngularRate(
                current, target, parameters
                );

        // Error is expressed in the current body frame.
        assert(NearlyEqual(rate.x(), 0.4));
        assert(NearlyEqual(rate.y(), 0.0));
        assert(NearlyEqual(rate.z(), 0.0));
    }

    // 10. Invalid controller parameters
    {
        AttitudeControllerParameters invalid = parameters;
        invalid.kp = -1.0;

        bool threw = false;

        try {
            CalculateDesiredAngularRate(
                    identity, identity, invalid
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    // 11. Invalid quaternion
    {
        Eigen::Quaterniond invalid(
                0.0, 0.0, 0.0, 0.0
                );

        bool threw = false;

        try {
            CalculateDesiredAngularRate(
                    invalid, identity, parameters
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    return 0;
}
