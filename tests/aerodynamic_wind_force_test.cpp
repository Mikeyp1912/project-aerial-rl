

#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>

#include "aerial/physics/aerodynamic_drag.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/drone_state.hpp"

int main()
{
    using namespace aerial::physics;

    DroneParameters parameters;
    parameters.linear_drag = Eigen::Vector3d(
        0.1, 0.1, 0.05
    );
    DroneState state;

    state.velocity.setZero();

    const Eigen::Vector3d wind(5.0, -3.0, 0.0);

    const Eigen::Vector3d force =
        CalculateAerodynamicDrag(state, parameters, wind);

    // Stationary drone experiences force in wind direction.
    assert(force.x() > 0.0);
    assert(force.y() < 0.0);
    assert(std::abs(force.z()) < 1e-12);

    const Eigen::Vector3d expected =
        parameters.linear_drag.cwiseProduct(wind);

    assert((force - expected).norm() < 1e-12);

    // No relative airspeed means no aerodynamic drag.
    state.velocity = wind;

    const Eigen::Vector3d zero_force =
        CalculateAerodynamicDrag(state, parameters, wind);

    assert(zero_force.norm() < 1e-12);

    std::cout << "Aerodynamic wind force tests PASSED\n";
}
