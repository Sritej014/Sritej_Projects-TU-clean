#include <Eigen/Dense>

namespace integrate {
    /**
    * @brief Classical Runge-Kutta method.
    *
    * Computes the next state using the classical Runge-Kutta method 
    * for solving ordinary differential equations (ODEs). 
    * This method is explicit and fourth-order accurate.
    *
    * @param f Right hand side of the ODE system dC/dt = f(C)
    * @param Cn The current state vector.
    * @param dt The time step size.
    * @return The state vector at the next time step.
    */
   Eigen::Vector2d rk4(
        const std::function<Eigen::Vector2d(const Eigen::Vector2d& C)>& f,
        const Eigen::Vector2d& Cn, 
        const double& dt
        );
       
    /**
    * @brief Explicit Euler method.
    *
    * Computes the next state using the explicit Euler method for solving
    * ordinary differential equations (ODEs). 
    * This method is first-order accurate.
    *
    * @param f Right hand side of the ODE system dC/dt = f(C)
    * @param Cn The current state vector.
    * @param dt The time step size.
    * @return The state vector at the next time step.
    */
    Eigen::Vector2d Euler(
        const std::function<Eigen::Vector2d(const Eigen::Vector2d& C)>& f,
        const Eigen::Vector2d& Cn,
        const double& dt
    );

} // end namespace integrate
