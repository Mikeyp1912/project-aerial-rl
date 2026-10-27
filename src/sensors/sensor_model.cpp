// /src/sensors/sensor_model.cpp

#include "aerial/sensors/sensor_model.hpp"
#include "aerial/physics/integrator.hpp"

#include <stdexcept>

namespace aerial::sensors
{
    SensorModel::SensorModel(
            const SensorParameters& parameters,
            std::uint32_t seed
            )
        : parameters_(parameters),
        generator_(seed)
    {
        const auto valid_noise = [](const Eigen::Vector3d& value) {
            return value.allFinite() && (value.array() >= 0.0).all();
        };

        if (!valid_noise(parameters_.position_noise) ||
            !valid_noise(parameters_.velocity_noise) ||
            !valid_noise(parameters_.angular_velocity_noise) ||
            !parameters_.position_bias.allFinite() ||
            !parameters_.velocity_bias.allFinite() ||
            !parameters_.angular_velocity_bias.allFinite() ||
            !valid_noise(parameters_.orientation_noise) ||
            !parameters_.orientation_bias.allFinite() ||
            !valid_noise(parameters_.accelerometer_noise) ||
            !parameters_.accelerometer_bias.allFinite()){
            throw std::invalid_argument("Invalid sensor parameters");
        }
    }

    Eigen::Vector3d SensorModel::SampleNoise(
            const Eigen::Vector3d& standard_deviation
            )
    {
        Eigen::Vector3d noise = Eigen::Vector3d::Zero();

        for (int axis = 0; axis < 3; ++axis) {
            if (standard_deviation[axis] > 0.0) {
                noise[axis] = standard_deviation[axis] * normal_(generator_);
            }
        }

        return noise;
    }

    SensorMeasurements SensorModel::Measure(
            const physics::DroneState& true_state,
            const Eigen::Vector3d& world_acceleration
            )
    {
        SensorMeasurements measurement;

        measurement.position = 
            true_state.position +
            parameters_.position_bias +
            SampleNoise(parameters_.position_noise);

        measurement.velocity =
            true_state.velocity +
            parameters_.velocity_bias +
            SampleNoise(parameters_.velocity_noise);

        const Eigen::Vector3d rotation_error =
            parameters_.orientation_bias +
            SampleNoise(parameters_.orientation_noise);

        const double angle = rotation_error.norm();

        Eigen::Quaterniond error_quaternion = Eigen::Quaterniond::Identity();

        if (angle > 0.0) {
            error_quaternion = Eigen::Quaterniond(
                    Eigen::AngleAxisd(
                        angle,
                        rotation_error / angle
                        )
                    );
        }

        measurement.orientation = (true_state.orientation * error_quaternion).normalized();

        measurement.angular_velocity =
            true_state.angular_velocity +
            parameters_.angular_velocity_bias +
            SampleNoise(parameters_.angular_velocity_noise);

        // Gravity acceleration in world coordinates
        const Eigen::Vector3d gravity_world(
                0.0,
                0.0,
                -physics::gravity
                );

        // Accelerometers measure non-gravitational specific force
        const Eigen::Vector3d specific_force_world = world_acceleration - gravity_world;

        // Transform from world coordinates into the drone body frame.
        // Use the true orientation, not the noisy orientation measurement
        const Eigen::Vector3d specific_force_body = true_state.orientation.conjugate() * specific_force_world;

        measurement.acceleration =
            specific_force_body +
            parameters_.accelerometer_bias +
            SampleNoise(parameters_.accelerometer_noise);

        return measurement;
    }

    void SensorModel::Reset(std::uint32_t seed)
    {
        generator_.seed(seed);
        normal_.reset();
    }

    const SensorParameters& SensorModel::GetParameters() const
    {
        return parameters_;
    }
}
