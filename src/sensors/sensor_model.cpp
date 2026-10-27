// /src/sensors/sensor_model.cpp

#include "aerial/sensors/sensor_model.hpp"

#include <cmath>
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
            !parameters_.angular_velocity_bias.allFinite()) {
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
            const physics::DroneState& true_state
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

        measurement.orientation = true_state.orientation;

        measurement.angular_velocity =
            true_state.angular_velocity +
            parameters_.angular_velocity_bias +
            SampleNoise(parameters_.angular_velocity_noise);

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
