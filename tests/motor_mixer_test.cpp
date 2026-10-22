// /tests/motor_mixer_test.cpp

#include "aerial/physics/motor_mixer.hpp"
#include "aerial/physics/motor_model.hpp"
#include "aerial/physics/motor_commands.hpp"

#include <Eigen/Core>

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace aerial::physics;

namespace
{
    constexpr double tolerance = 1e-9;

    bool NearlyEqual(double a, double b)
    {
        return std::abs(a - b) < tolerance;
    }

    void VerifyOutput(
            const DroneParameters& parameters,
            double thrust,
            const Eigen::Vector3d& torque
            )
    {
        const auto mixed = MixMotors(parameters, thrust, torque);

        assert(!mixed.saturated);

        for (double command : mixed.commands) {
            assert(command >= 0.0);
            assert(command <= 1.0);
        }

        const auto motor_thrusts =
            CommandsToThrust(parameters, mixed.commands);

        const auto output =
            CalculateMotorOutput(parameters, motor_thrusts);

        assert(NearlyEqual(output.force.x(), 0.0));
        assert(NearlyEqual(output.force.y(), 0.0));
        assert(NearlyEqual(output.force.z(), thrust));

        assert((output.torque - torque).norm() < tolerance);
    }
}

int main()
{
    DroneParameters parameters;

    // Zero input
    {
        const auto result = MixMotors(
                parameters, 0.0, Eigen::Vector3d::Zero()
                );

        assert(!result.saturated);

        for (double command : result.commands) {
            assert(NearlyEqual(command, 0.0));
        }
    }

    // Equal collective thrust
    {
        const auto result = MixMotors(
                parameters, 10.0, Eigen::Vector3d::Zero()
                );

        for (double command : result.commands) {
            assert(NearlyEqual(command, 0.5));
        }

        VerifyOutput(parameters, 10.0, Eigen::Vector3d::Zero());
    }

    // Pure roll
    VerifyOutput(
            parameters, 10.0,
            Eigen::Vector3d(0.2, 0.0, 0.0)
            );

    // Pure pitch
    VerifyOutput(
            parameters, 10.0,
            Eigen::Vector3d(0.0, 0.2, 0.0)
            );

    // Pure yaw
    VerifyOutput(
            parameters, 10.0,
            Eigen::Vector3d(0.0, 0.0, 0.02)
            );

    // Combined thrust and torques
    VerifyOutput(
            parameters, 12.0,
            Eigen::Vector3d(0.1, -0.15, 0.01)
            );

    // Upper saturation
    {
        const auto result = MixMotors(
                parameters, 30.0, Eigen::Vector3d::Zero()
                );

        assert(result.saturated);

        for (double command : result.commands) {
            assert(NearlyEqual(command, 1.0));
        }
    }

    // Lower saturation
    {
        const auto result = MixMotors(
                parameters, -10.0, Eigen::Vector3d::Zero()
                );

        assert(result.saturated);

        for (double command : result.commands) {
            assert(NearlyEqual(command, 0.0));
        }
    }

    // Differential saturation
    {
        const auto result = MixMotors(
                parameters, 10.0,
                Eigen::Vector3d(10.0, 0.0, 0.0)
                );

        assert(result.saturated);

        for (double command : result.commands) {
            assert(command >= 0.0);
            assert(command <= 1.0);
        }
    }

    // Invalid parameters
    {
        DroneParameters invalid = parameters;
        invalid.arm_length = 0.0;

        bool threw = false;

        try {
            MixMotors(invalid, 10.0, Eigen::Vector3d::Zero());
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    // Non-finite input
    {
        bool threw = false;

        try {
            MixMotors(
                    parameters,
                    std::numeric_limits<double>::quiet_NaN(),
                    Eigen::Vector3d::Zero()
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    return 0;
}
