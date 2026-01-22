#include <Eigen/Dense>

namespace brusselator {

    /**
    * @brief ODE System of the Brusselator.
    *
    * @param C Current state vector
    * @param Ca System parameter
    * @param Cb System parameter
    * @return The derivative vector dC / dt
    */
    Eigen::Vector2d odeSystem(
        const Eigen::Vector2d& C, 
        const double& Ca, 
        const double& Cb
        );
}
