// /include/aerial/physics/drone_state.hpp

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace aerial::physics
{
    struct DroneState
    {
        // World-frame position (metres)
        Eigen::Vector3d position = Eigen::Vector3d::Zero();

        // World-frame linear velocity (m/s)
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

        // Body-to-world orientation
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();

        // Body-frame angular velocity (rad/s)
        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    };
}
