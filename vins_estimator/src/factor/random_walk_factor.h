/**
 * @file random_walk_factor.h
 * @brief Random Walk constraint for para_a temporal consistency
 *
 * This factor enforces smooth evolution of per-frame scale parameters
 * in the sliding window, preventing abrupt jumps between consecutive frames.
 *
 * Residual: r = (a_{k} - a_{k-1}) * sqrt_info
 *
 * where:
 *   - a_{k-1}: Previous frame's scale parameter
 *   - a_{k}: Current frame's scale parameter
 *   - sqrt_info = 1 / random_walk_noise
 */

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

/**
 * @class RandomWalkFactor
 * @brief Enforces temporal consistency between adjacent para_a parameters
 *
 * Parameter blocks:
 *   [0] para_a_prev (1D): Previous frame's scale parameter a_{k-1}
 *   [1] para_a_curr (1D): Current frame's scale parameter a_{k}
 *
 * Residual dimension: 1 (scalar)
 */
class RandomWalkFactor : public ceres::SizedCostFunction<1, 1, 1>
{
public:
    /**
     * @brief Constructor
     * @param random_walk_noise Standard deviation of the random walk process
     *                          Lower values = stricter constraint (less drift)
     *                          Typical range: 0.001 ~ 0.01
     */
    explicit RandomWalkFactor(double random_walk_noise)
        : sqrt_info_(1.0 / random_walk_noise)
    {
        if (random_walk_noise <= 0.0) {
            // Prevent division by zero
            sqrt_info_ = 1e6;  // Very stiff constraint
        }
    }

    /**
     * @brief Evaluate residual and Jacobians
     */
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override
    {
        // Unpack parameters
        double a_prev = parameters[0][0];
        double a_curr = parameters[1][0];

        // Compute residual: r = (a_curr - a_prev) * sqrt_info
        residuals[0] = sqrt_info_ * (a_curr - a_prev);

        // Compute Jacobians if requested
        if (jacobians)
        {
            // Jacobian w.r.t. a_prev
            if (jacobians[0])
            {
                jacobians[0][0] = -sqrt_info_;
            }

            // Jacobian w.r.t. a_curr
            if (jacobians[1])
            {
                jacobians[1][0] = sqrt_info_;
            }
        }

        return true;
    }

private:
    double sqrt_info_;  // Square root of information matrix (1 / sigma)
};
