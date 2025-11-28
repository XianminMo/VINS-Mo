#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

/**
 * @brief 深度尺度偏移参数的随机游走先验因子
 *
 * 该因子约束深度参数 (a, b) 相对于上一帧的变化，实现随机游走模型：
 * a_{k+1} = a_k + n_a, 其中 n_a ~ N(0, sigma_a^2)
 * b_{k+1} = b_k + n_b, 其中 n_b ~ N(0, sigma_b^2)
 *
 * 残差定义：
 * residual[0] = (a_current - a_previous) / sigma_a
 * residual[1] = (b_current - b_previous) / sigma_b
 */
class DepthScaleShiftRandomWalkFactor : public ceres::SizedCostFunction<2, 2>
{
public:
    /**
     * @param a_prev 上一帧优化后的 a 值
     * @param b_prev 上一帧优化后的 b 值
     * @param sigma_a 随机游走过程噪声标准差（a 参数）
     * @param sigma_b 随机游走过程噪声标准差（b 参数）
     */
    DepthScaleShiftRandomWalkFactor(double a_prev, double b_prev,
                                    double sigma_a, double sigma_b)
        : a_prev_(a_prev), b_prev_(b_prev),
          inv_sigma_a_(1.0 / sigma_a), inv_sigma_b_(1.0 / sigma_b)
    {
    }

    virtual bool Evaluate(double const *const *parameters,
                         double *residuals,
                         double **jacobians) const
    {
        // parameters[0] = [a_current, b_current]
        const double a_current = parameters[0][0];
        const double b_current = parameters[0][1];

        // 计算残差：当前值与前值的差异，归一化到单位协方差
        residuals[0] = (a_current - a_prev_) * inv_sigma_a_;
        residuals[1] = (b_current - b_prev_) * inv_sigma_b_;

        // 计算雅可比矩阵
        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            // d(residual)/d(para_DepthScaleShift)
            // residuals 是 2x1，parameters[0] 是 2x1
            // 所以雅可比是 2x2 矩阵

            // d(residual[0])/d(a) = 1 / sigma_a
            jacobians[0][0] = inv_sigma_a_;
            // d(residual[0])/d(b) = 0
            jacobians[0][1] = 0.0;

            // d(residual[1])/d(a) = 0
            jacobians[0][2] = 0.0;
            // d(residual[1])/d(b) = 1 / sigma_b
            jacobians[0][3] = inv_sigma_b_;
        }

        return true;
    }

private:
    double a_prev_;          // 上一帧优化后的 a 值
    double b_prev_;          // 上一帧优化后的 b 值
    double inv_sigma_a_;     // 1 / sigma_a (预计算以提高效率)
    double inv_sigma_b_;     // 1 / sigma_b (预计算以提高效率)
};
