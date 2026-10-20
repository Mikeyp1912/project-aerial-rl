// /tests/drone_state_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>

#include "aerial/physics/drone_state.hpp"

int main() {
    aerial::physics::DroneState state;

    // Default state must be stationary at the origin.
    assert(state.position.isZero());
    assert(state.velocity.isZero());
    assert(state.angular_velocity.isZero());

    // Orientation must represent no rotation
    assert(state.orientation.isApprox(Eigen::Quaterniond::Identity()));

    // Verify body-to-world rotation convention.
    constexpr double pi = 3.14159265358979323846;

    state.orientation = Eigen::Quaterniond(
        Eigen::AngleAxisd(
            pi / 2.0,
            Eigen::Vector3d::UnitZ()
        )
    );

    const Eigen::Vector3d world_x =
        state.orientation * Eigen::Vector3d::UnitX();

    assert(std::abs(world_x.x()) < 1e-12);
    assert(std::abs(world_x.y() - 1.0) < 1e-12);
    assert(std::abs(world_x.z()) < 1e-12);

    std::cout << "Drone state tests passed\n";

    return 0;
}
