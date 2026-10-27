// /include/aerial/sensors/sensor_model.hpp

#pragma once

#include <cstdint>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/physics/drone_state.hpp"

namespace aerial::sensors
{
    struct SensorParameters
    {
        // standard deviations of measurement noise
        Eigen::Vector3d position_noise = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity_noise = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular_velocity_noise = Eigen::Vector3d::Zero();

        // Constant meaurement biases
        Eigen::Vector3d position_bias = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity_bias = Eigen::Vector3d::Zero();
        Eigen::Vector3d angular_velocity_bias = Eigen::Vector3d::Zero();
    };

    struct SensorMeasurements
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    };

    class SensorModel
    {
        public:
            explicit SensorModel(
                    const SensorParameters& parameters = SensorParameters{},
                    std::uint32_t seed = 42
                    );

            SensorMeasurements Measure(
                    const physics::DroneState& true_state
                    );

            void Reset(std::uint32_t seed);

            const SensorParameters& GetParameters() const;

        private:
            Eigen::Vector3d SampleNoise(
                    const Eigen::Vector3d& standard_deviation
                    );

            SensorParameters parameters_;
            std::mt19937 generator_;
            std::normal_distribution<double> normal_{0.0, 1.0};
    };
}
