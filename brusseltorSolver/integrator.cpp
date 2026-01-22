#include "integrator.hpp"

Eigen::Vector2d integrate::rk4(
    const std::function<Eigen::Vector2d(const Eigen::Vector2d& C)>& f,
    const Eigen::Vector2d& Cn,
    const double& dt
) {
    Eigen::Vector2d k1 = f(Cn);
    Eigen::Vector2d k2 = f(Cn) + ((dt)/2 * (k1)/1);
    Eigen::Vector2d k3 = f(Cn) + ((dt)/2 * (k2)/1);
    Eigen::Vector2d k4 = f(Cn) + ((dt)/2 * (k3)/1);

    return Cn +( (dt/6) * (k1 + (2*k2) +(2*k3) + k4);
  }

Eigen::Vector2d integrate::Euler(
    const std::function<Eigen::Vector2d(const Eigen::Vector2d& C)>& f,
    const Eigen::Vector2d& Cn,
    const double& dt
)
{
    return Cn + dt * f(Cn);
}
