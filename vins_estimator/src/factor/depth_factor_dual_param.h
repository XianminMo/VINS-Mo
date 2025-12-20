/**
 * @file depth_factor_dual_param.h
 * @brief Dual-Parameter Depth Factor (Scale + Shift Refinement in Inverse Depth Space)
 *
 * This factor extends the original DepthFactor by introducing both scale (a) and shift (b)
 * parameters for backend optimization to model residual offsets more accurately.
 *
 * Mathematical Model (Inverse Depth Space):
 *   r = lambda_vins - (a * lambda_aligned + b)
 *
 * where:
 *   - lambda_vins: VINS triangulated inverse depth (1/d_metric) [State variable]
 *   - lambda_aligned: Frontend-aligned depth measurement (inverse) [Fixed measurement]
 *   - a: Scale refinement factor [State variable, initialized to 1.0]
 *   - b: Shift refinement factor [State variable, initialized to 0.0]
 *
 * Random Walk Process Model:
 *   - Both a and b are modeled as random walk processes (propagated frame-to-frame)
 *   - a: Standard random walk noise (tracks scale drift)
 *   - b: Very small random walk noise (acts as "buffer" for slight offsets, should be lazy)
 *
 * Jacobians:
 *   ∂r/∂lambda_vins = 1
 *   ∂r/∂a = -lambda_aligned
 *   ∂r/∂b = -1
 *
 * Parameter blocks:
 *   [0] para_Pose_i (7D): Feature first observation frame pose
 *   [1] para_Pose_j (7D): Current frame pose (depth map frame)
 *   [2] para_Ex_Pose (7D): IMU-to-camera extrinsics
 *   [3] para_Feature (1D): Feature inverse depth in frame i
 *   [4] para_a (1D): Scale refinement parameter
 *   [5] para_b (1D): Shift refinement parameter
 */

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../parameters.h"

class DepthFactorDualParam : public ceres::SizedCostFunction<1, 7, 7, 7, 1, 1, 1>
{
public:
    /**
     * @brief Constructor
     * @param _aligned_depth Frontend-aligned depth value (d_aligned = s * d_mean + t)
     * @param _aligned_sigma Frontend-aligned uncertainty (sigma_aligned = s * sigma_tta)
     * @param _pts_i Feature point in normalized camera coordinates [x, y, 1]
     */
    DepthFactorDualParam(double _aligned_depth, double _aligned_sigma, const Eigen::Vector3d& _pts_i)
        : aligned_depth(_aligned_depth), aligned_sigma(_aligned_sigma), pts_i(_pts_i)
    {
        // TTA-based adaptive weighting using global parameter DEPTH_WEIGHT_K
        const double K = DEPTH_WEIGHT_K;  // Use global parameter from config
        const double eps = 1e-6;  // Numerical stability
        const double MIN_REL_ERROR = 0.05;

        double safe_depth = std::max(aligned_depth, 0.1);
        double floor_sigma = MIN_REL_ERROR / safe_depth;

        double final_sigma = std::max(aligned_sigma, floor_sigma);
        // Weight inversely proportional to uncertainty: W = 1 / (K * sigma + eps)^2
        double variance = std::pow(K * final_sigma + eps, 2);
        sqrt_info = 1.0 / std::sqrt(variance);

        // Validity check
        if (aligned_depth <= 0.0 || std::isnan(aligned_depth) || std::isinf(aligned_depth))
        {
            sqrt_info = 0.0;  // Disable this factor
        }
        if (aligned_sigma < 0.0 || std::isnan(aligned_sigma))
        {
            sqrt_info = 0.0;  // Disable this factor
        }
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        // ========== Step 1: Unpack parameters ==========
        // Frame i pose (feature first observation)
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();

        // Frame j pose (current frame with depth map)
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();

        // IMU-to-camera extrinsics
        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        Eigen::Matrix3d ric = qic.toRotationMatrix();

        // Feature inverse depth
        double inv_dep_i = parameters[3][0];

        // Dual refinement parameters
        double a = parameters[4][0];  // Scale
        double b = parameters[5][0];  // Shift

        // ========== Step 2: Compute VINS metric inverse depth (lambda_vins) ==========
        // Transform feature from frame i to frame j
        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = ric * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Ri * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Rj.transpose() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = ric.transpose() * (pts_imu_j - tic);

        double depth_metric_j = pts_camera_j.z();

        // ========== Step 3: Depth validity check ==========
        const double min_depth_threshold = 0.05;  // Minimum valid depth (meters)
        if (depth_metric_j <= min_depth_threshold || sqrt_info == 0.0)
        {
            residuals[0] = 0.0;
            if (jacobians)
            {
                if (jacobians[0]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[0]).setZero();
                if (jacobians[1]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[1]).setZero();
                if (jacobians[2]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[2]).setZero();
                if (jacobians[3]) Eigen::Map<Eigen::Matrix<double, 1, 1>>(jacobians[3]).setZero();
                if (jacobians[4]) Eigen::Map<Eigen::Matrix<double, 1, 1>>(jacobians[4]).setZero();
                if (jacobians[5]) Eigen::Map<Eigen::Matrix<double, 1, 1>>(jacobians[5]).setZero();
            }
            return true;
        }

        double lambda_vins = 1.0 / depth_metric_j;  // VINS inverse depth
        double lambda_aligned = 1.0 / aligned_depth;  // Aligned inverse depth (measurement)

        // ========== Step 4: Compute residual ==========
        // Residual: r = (lambda_vins - (a * lambda_aligned + b)) * sqrt_info
        // This models: lambda_vins = a * lambda_aligned + b + noise
        double predicted_lambda = a * lambda_aligned + b;
        residuals[0] = sqrt_info * (lambda_vins - predicted_lambda);

        // ========== Step 5: Compute Jacobians (if requested) ==========
        if (jacobians)
        {
            // Common term: derivative of residual w.r.t. lambda_vins
            // d(residual)/d(lambda_vins) = sqrt_info
            // d(lambda_vins)/d(depth_j) = -1 / (depth_j^2)
            // d(residual)/d(P_cam_j) = [0, 0, -sqrt_info / (depth_j^2)]
            Eigen::Matrix<double, 1, 3> d_res_d_Pcam_j;
            double inv_depth_sq = 1.0 / (depth_metric_j * depth_metric_j);
            d_res_d_Pcam_j << 0, 0, -sqrt_info * inv_depth_sq;

            // ---------- Jacobian w.r.t. Pose_i (first observation frame) ----------
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
                jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri *
                                        (-Utility::skewSymmetric(pts_imu_i));

                jacobian_pose_i.leftCols<6>() = d_res_d_Pcam_j * jaco_i;
                jacobian_pose_i.rightCols<1>().setZero();
            }

            // ---------- Jacobian w.r.t. Pose_j (current frame) ----------
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

                Eigen::Matrix<double, 3, 6> jaco_j;
                jaco_j.leftCols<3>() = ric.transpose() * (-Rj.transpose());
                jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

                jacobian_pose_j.leftCols<6>() = d_res_d_Pcam_j * jaco_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }

            // ---------- Jacobian w.r.t. Ex_Pose (extrinsics) ----------
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

                Eigen::Matrix<double, 3, 6> jaco_ex;
                jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());

                Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
                jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) +
                                         Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                         Utility::skewSymmetric(ric.transpose() *
                                                               (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));

                jacobian_ex_pose.leftCols<6>() = d_res_d_Pcam_j * jaco_ex;
                jacobian_ex_pose.rightCols<1>().setZero();
            }

            // ---------- Jacobian w.r.t. Feature (inverse depth lambda_k) ----------
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 1>> jacobian_feature(jacobians[3]);

                Eigen::Vector3d d_Pcam_i_d_lambda = (-1.0 / (inv_dep_i * inv_dep_i)) * pts_i;
                Eigen::Vector3d d_Pcam_j_d_lambda = ric.transpose() * Rj.transpose() * Ri * ric * d_Pcam_i_d_lambda;

                jacobian_feature = d_res_d_Pcam_j * d_Pcam_j_d_lambda;
            }

            // ---------- Jacobian w.r.t. a (scale parameter) ----------
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 1>> jacobian_a(jacobians[4]);

                // d(residual)/d(a) = -sqrt_info * lambda_aligned
                jacobian_a(0, 0) = -sqrt_info * lambda_aligned;
            }

            // ---------- Jacobian w.r.t. b (shift parameter) ----------
            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 1>> jacobian_b(jacobians[5]);

                // d(residual)/d(b) = -sqrt_info
                jacobian_b(0, 0) = -sqrt_info;
            }
        }

        return true;
    }

private:
    double aligned_depth;   // Frontend-aligned depth (d_aligned = s * d_mean + t)
    double aligned_sigma;   // Frontend-aligned uncertainty (sigma_aligned = s * sigma_tta)
    Eigen::Vector3d pts_i;  // Feature point in frame i normalized coordinates
    double sqrt_info;       // Square root of information matrix (adaptive weight)
};
