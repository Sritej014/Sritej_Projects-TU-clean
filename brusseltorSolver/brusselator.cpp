#include "brusselator.hpp"

Eigen::Vector2d brusselator::odeSystem(
    const Eigen::Vector2d& C, 
    const double& Ca, 
    const double& Cb
    ) {
    double dC1_dt = Ca + C[0] * C[0] * C[1] - Cb * C[0] - C[0]; // dC1/dt
    double dC2_dt = Cb * C[0] - C[0] * C[0] * C[1];             // dC2/dt

    return Eigen::Vector2d(dC1_dt, dC2_dt);
    }
