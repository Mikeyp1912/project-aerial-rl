// /tests/rate_controller_test.cpp


#include "aerial/control/rate_controller.hpp"

#include <Eigen/Core>

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace aerial::control;

namespace
{
    constexpr double tolerance = 1e-9;

    bool NearlyEqual(double a, double b)
    {
        return std::abs(a - b) < tolerance;
    }

    bool NearlyEqual(
            const Eigen::Vector3d& a,
            const Eigen::Vector3d& b
            )
    {
        return (a - b).norm() < tolerance;
    }

    RateControllerParameters TestParameters()
    {
        RateControllerParameters parameters;

        parameters.kp = Eigen::Vector3d::Constant(1.0);
        parameters.ki = Eigen::Vector3d::Zero();
        parameters.kd = Eigen::Vector3d::Zero();

        parameters.integral_limit =
            Eigen::Vector3d::Constant(10.0);

        parameters.torque_limit =
            Eigen::Vector3d::Constant(10.0);

        return parameters;
    }
}

int main()
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double dt = 0.01;

    // 1. Zero error produces zero torque.
    {
        RateControllerState state;
        const auto parameters = TestParameters();

        const auto torque = CalculateBodyTorque(
                zero, zero, dt, parameters, state
                );

        assert(NearlyEqual(torque, zero));
        assert(state.initialized);
    }

    // 2. Proportional control on all three axes.
    {
        RateControllerState state;
        const auto parameters = TestParameters();

        const Eigen::Vector3d desired(0.5, -0.25, 0.1);

        const auto torque = CalculateBodyTorque(
                desired, zero, dt, parameters, state
                );

        assert(NearlyEqual(torque, desired));
    }

    // 3. Actual angular velocity reduces the error.
    {
        RateControllerState state;
        const auto parameters = TestParameters();

        const auto torque = CalculateBodyTorque(
                Eigen::Vector3d(1.0, 0.0, 0.0),
                Eigen::Vector3d(0.4, 0.0, 0.0),
                dt,
                parameters,
                state
                );

        assert(NearlyEqual(torque.x(), 0.6));
    }

    // 4. Derivative damping opposes angular acceleration.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.kp.setZero();
        parameters.kd = Eigen::Vector3d::Constant(0.1);

        // Initialize the previous angular velocity.
        CalculateBodyTorque(
                zero, zero, dt, parameters, state
                );

        const auto torque = CalculateBodyTorque(
                zero,
                Eigen::Vector3d(0.1, 0.0, 0.0),
                dt,
                parameters,
                state
                );

        // Acceleration = 0.1 / 0.01 = 10 rad/s^2
        // Derivative torque = -0.1 * 10 = -1 Nm
        assert(NearlyEqual(torque.x(), -1.0));
        assert(NearlyEqual(torque.y(), 0.0));
        assert(NearlyEqual(torque.z(), 0.0));
    }

    // 5. Integral accumulation.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.kp.setZero();
        parameters.ki = Eigen::Vector3d::Constant(1.0);

        for (int i = 0; i < 100; ++i) {
            CalculateBodyTorque(
                    Eigen::Vector3d(1.0, 0.0, 0.0),
                    zero,
                    dt,
                    parameters,
                    state
                    );
        }

        assert(NearlyEqual(state.integral_error.x(), 1.0));
    }

    // 6. Integral state is bounded.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.kp.setZero();
        parameters.ki = Eigen::Vector3d::Constant(1.0);
        parameters.integral_limit =
            Eigen::Vector3d::Constant(0.2);

        for (int i = 0; i < 100; ++i) {
            CalculateBodyTorque(
                    Eigen::Vector3d(1.0, 0.0, 0.0),
                    zero,
                    dt,
                    parameters,
                    state
                    );
        }

        assert(NearlyEqual(state.integral_error.x(), 0.2));
    }

    // 7. Torque saturation.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.torque_limit =
            Eigen::Vector3d(0.5, 0.25, 0.1);

        const auto torque = CalculateBodyTorque(
                Eigen::Vector3d(10.0, -10.0, 10.0),
                zero,
                dt,
                parameters,
                state
                );

        assert(NearlyEqual(
                    torque,
                    Eigen::Vector3d(0.5, -0.25, 0.1)
                    ));
    }

    // 8. Anti-windup prevents further accumulation
    // when torque is saturated in the error direction.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.kp = Eigen::Vector3d::Constant(2.0);
        parameters.ki = Eigen::Vector3d::Constant(1.0);
        parameters.torque_limit =
            Eigen::Vector3d::Constant(0.5);

        for (int i = 0; i < 100; ++i) {
            CalculateBodyTorque(
                    Eigen::Vector3d(1.0, 0.0, 0.0),
                    zero,
                    dt,
                    parameters,
                    state
                    );
        }

        assert(NearlyEqual(state.integral_error.x(), 0.0));
    }

    // 9. Integral error can unwind.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        parameters.kp.setZero();
        parameters.ki = Eigen::Vector3d::Constant(1.0);
        parameters.torque_limit =
            Eigen::Vector3d::Constant(0.5);

        state.integral_error.x() = 0.5;

        CalculateBodyTorque(
                Eigen::Vector3d(-1.0, 0.0, 0.0),
                zero,
                dt,
                parameters,
                state
                );

        assert(state.integral_error.x() < 0.5);
    }

    // 10. Reset clears the controller state.
    {
        RateControllerState state;
        auto parameters = TestParameters();

        CalculateBodyTorque(
                Eigen::Vector3d(1.0, 0.0, 0.0),
                zero,
                dt,
                parameters,
                state
                );

        ResetRateController(state);

        assert(!state.initialized);
        assert(NearlyEqual(state.integral_error, zero));
        assert(NearlyEqual(state.previous_angular_velocity, zero));
    }

    // 11. Invalid time step is rejected.
    {
        RateControllerState state;
        const auto parameters = TestParameters();

        bool threw = false;

        try {
            CalculateBodyTorque(
                    zero, zero, 0.0, parameters, state
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    // 12. Non-finite input is rejected.
    {
        RateControllerState state;
        const auto parameters = TestParameters();

        Eigen::Vector3d invalid = zero;
        invalid.x() = std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try {
            CalculateBodyTorque(
                    invalid, zero, dt, parameters, state
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    return 0;
}
