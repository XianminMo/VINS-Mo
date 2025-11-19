/**
 * @file depth_factor.h
 * @brief 深度传感器因子 (Depth Sensor Factor)
 *
 * 将深度学习模型（如MiDaS）预测的深度图约束引入VINS后端优化中。
 *
 * 核心思想：
 * 深度学习模型输出的深度 d_nn 与真实度量深度 d_metric 之间存在仿射变换关系：
 *     1/d_metric = a * d_nn + b
 * 其中 a 为尺度因子，b 为偏移因子。
 *
 * 该因子通过比较：
 * - 测量值：深度图中的预测逆深度 d_nn (归一化的)
 * - 估计值：VINS通过特征三角化得到的度量逆深度 1/d_metric
 * 来构建约束，并通过优化求解 a 和 b，使深度先验信息指导位姿和特征深度的估计。
 *
 * 参考论文：
 * "Learned Monocular Depth Priors in Visual-Inertial Initialization"
 */

#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../parameters.h"

/**
 * @class DepthFactor
 * @brief 深度约束因子，连接5个参数块
 *
 * 残差维度：1 (标量残差)
 * 参数块：
 *   [0] para_Pose_i (7维): 特征点首次观测帧 i 的位姿 [P, Q(x,y,z,w)]
 *   [1] para_Pose_j (7维): 深度图所在帧 j 的位姿 [P, Q(x,y,z,w)]
 *   [2] para_Ex_Pose (7维): IMU到相机的外参 [t_ic, q_ic(x,y,z,w)]
 *   [3] para_Feature (1维): 特征点 k 在帧 i 下的逆深度 lambda_k
 *   [4] para_ScaleShift (2维): 深度仿射变换参数 [a, b]
 *
 * 残差计算：
 *   residual = sqrt_info * ((a * d_nn + b) - (1 / d_metric_j))
 * 其中 d_metric_j 通过特征点的世界坐标变换到帧 j 的相机系计算得到。
 */
class DepthFactor : public ceres::SizedCostFunction<1, 7, 7, 7, 1, 2>
{
public:
    /**
     * @brief 构造函数
     * @param _predicted_inv_depth 深度图中该特征点位置的预测逆深度值 d_nn
     * @param _pts_i 特征点在首次观测帧 i 中的归一化相机坐标 [x, y, 1]
     */
    DepthFactor(const double _predicted_inv_depth, const Eigen::Vector3d& _pts_i)
        : predicted_inv_depth(_predicted_inv_depth), pts_i(_pts_i)
    {
        // 设置信息矩阵的平方根（权重）
        // sqrt_info 越大，该因子对优化的影响越大
        sqrt_info = DEPTH_FACTOR_WEIGHT;
    }

    /**
     * @brief Ceres求解器调用的残差和雅可比计算函数
     *
     * @param parameters 输入参数块指针数组
     *   parameters[0]: para_Pose_i (首次观测帧位姿)
     *   parameters[1]: para_Pose_j (当前帧位姿)
     *   parameters[2]: para_Ex_Pose (外参)
     *   parameters[3]: para_Feature (特征点逆深度)
     *   parameters[4]: para_ScaleShift (尺度偏移参数 [a, b])
     *
     * @param residuals 输出残差 (1维)
     * @param jacobians 输出雅可比矩阵 (如果不为nullptr)
     * @return true 表示计算成功
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // ========== 步骤1: 解包所有输入参数 ==========
        // 帧 i 的位姿 (特征点首次观测帧)
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();

        // 帧 j 的位姿 (深度图所在帧，即当前观测帧)
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();

        // IMU到相机的外参
        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        Eigen::Matrix3d ric = qic.toRotationMatrix();

        // 特征点在帧 i 下的逆深度
        double inv_dep_i = parameters[3][0];

        // 深度仿射变换参数
        double a = parameters[4][0];  // 尺度因子
        double b = parameters[4][1];  // 偏移因子

        // ========== 步骤2: 计算VINS估计的度量逆深度 1/d_metric_j ==========
        // 2.1 将特征点从归一化平面转换到帧 i 的相机坐标系
        //     P_cam_i = (1/lambda_k) * [x, y, 1]^T
        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;

        // 2.2 转换到帧 i 的IMU坐标系
        //     P_imu_i = R_ic * P_cam_i + t_ic
        Eigen::Vector3d pts_imu_i = ric * pts_camera_i + tic;

        // 2.3 转换到世界坐标系
        //     P_w = R_i * P_imu_i + P_i
        Eigen::Vector3d pts_w = Ri * pts_imu_i + Pi;

        // 2.4 转换到帧 j 的IMU坐标系
        //     P_imu_j = R_j^T * (P_w - P_j)
        Eigen::Vector3d pts_imu_j = Rj.transpose() * (pts_w - Pj);

        // 2.5 转换到帧 j 的相机坐标系
        //     P_cam_j = R_ic^T * (P_imu_j - t_ic)
        Eigen::Vector3d pts_camera_j = ric.transpose() * (pts_imu_j - tic);

        // 2.6 提取深度值 d_metric_j = P_cam_j.z
        double depth_metric_j = pts_camera_j.z();

        // ========== 步骤3: 深度有效性检查 ==========
        // 如果深度为负或过小，说明该点在相机后方或退化，不添加约束
        const double min_depth_threshold = 0.05;  // 最小深度阈值（米）
        if (depth_metric_j <= min_depth_threshold)
        {
            residuals[0] = 0.0;
            if (jacobians)
            {
                // 所有雅可比置零
                if (jacobians[0]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[0]).setZero();
                if (jacobians[1]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[1]).setZero();
                if (jacobians[2]) Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jacobians[2]).setZero();
                if (jacobians[3]) Eigen::Map<Eigen::Matrix<double, 1, 1>>(jacobians[3]).setZero();
                if (jacobians[4]) Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>>(jacobians[4]).setZero();
            }
            return true;
        }

        // 计算逆深度
        double estimated_metric_inv_depth = 1.0 / depth_metric_j;

        // ========== 步骤4: 计算残差 ==========
        // 测量值（经过仿射变换）: a * d_nn + b
        // 估计值: 1 / d_metric_j
        // 残差: r = (测量值 - 估计值) * 权重
        double predicted_metric_inv_depth = a * predicted_inv_depth + b;
        residuals[0] = sqrt_info * (predicted_metric_inv_depth - estimated_metric_inv_depth);

        // ========== 步骤5: 计算雅可比矩阵（如果需要）==========
        if (jacobians)
        {
            // 5.1 计算残差对 P_cam_j 的导数
            //     residual = sqrt_info * (... - 1/d_metric_j)
            //     d(residual)/d(d_metric_j) = sqrt_info * (1 / d_metric_j^2)
            //     d(residual)/d(P_cam_j) = d(residual)/d(d_metric_j) * d(d_metric_j)/d(P_cam_j)
            //                             = [0, 0, sqrt_info / d_metric_j^2]
            Eigen::Matrix<double, 1, 3> d_res_d_Pcam_j;
            double inv_depth_sq = 1.0 / (depth_metric_j * depth_metric_j);
            d_res_d_Pcam_j << 0, 0, sqrt_info * inv_depth_sq;

            // 5.2 计算各参数块的雅可比
            // 这些推导基于链式法则，参考 ProjectionFactor 的实现

            // ---------- Jacobian w.r.t. Pose_i (首次观测帧位姿) ----------
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                // P_cam_j 对 Pose_i 的导数 (通过链式法则)
                // d(P_cam_j)/d(P_i) = R_ic^T * R_j^T
                // d(P_cam_j)/d(R_i) = R_ic^T * R_j^T * R_i * (-[P_imu_i]_x)
                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
                jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri *
                                        (-Utility::skewSymmetric(pts_imu_i));

                jacobian_pose_i.leftCols<6>() = d_res_d_Pcam_j * jaco_i;
                jacobian_pose_i.rightCols<1>().setZero();  // 四元数的第4维（w分量）导数为0
            }

            // ---------- Jacobian w.r.t. Pose_j (当前观测帧位姿) ----------
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

                // P_cam_j 对 Pose_j 的导数
                // d(P_cam_j)/d(P_j) = -R_ic^T * R_j^T
                // d(P_cam_j)/d(R_j) = R_ic^T * [P_imu_j]_x
                Eigen::Matrix<double, 3, 6> jaco_j;
                jaco_j.leftCols<3>() = ric.transpose() * (-Rj.transpose());
                jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

                jacobian_pose_j.leftCols<6>() = d_res_d_Pcam_j * jaco_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }

            // ---------- Jacobian w.r.t. Ex_Pose (外参) ----------
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

                // P_cam_j 对外参的导数（较复杂，涉及多个变换）
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

            // ---------- Jacobian w.r.t. Feature (特征点逆深度 lambda_k) ----------
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 1>> jacobian_feature(jacobians[3]);

                // P_cam_j 对 lambda_k 的导数
                // d(P_cam_i)/d(lambda) = -(1/lambda^2) * pts_i
                // 然后沿着变换链传播
                Eigen::Vector3d d_Pcam_i_d_lambda = (-1.0 / (inv_dep_i * inv_dep_i)) * pts_i;
                Eigen::Vector3d d_Pcam_j_d_lambda = ric.transpose() * Rj.transpose() * Ri * ric * d_Pcam_i_d_lambda;

                jacobian_feature = d_res_d_Pcam_j * d_Pcam_j_d_lambda;
            }

            // ---------- Jacobian w.r.t. ScaleShift (深度仿射变换参数 [a, b]) ----------
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> jacobian_scale_shift(jacobians[4]);

                // 残差对 a 的导数: d(r)/d(a) = sqrt_info * d_nn
                jacobian_scale_shift(0, 0) = sqrt_info * predicted_inv_depth;

                // 残差对 b 的导数: d(r)/d(b) = sqrt_info * 1.0
                jacobian_scale_shift(0, 1) = sqrt_info * 1.0;
            }
        }

        return true;
    }

private:
    // 深度图中该点的预测逆深度值（归一化的，无度量尺度）
    const double predicted_inv_depth;

    // 特征点在首次观测帧 i 中的归一化坐标 [x, y, 1]^T
    const Eigen::Vector3d pts_i;

    // 信息矩阵的平方根（权重因子）
    double sqrt_info;
};
