#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string OUTPUT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_TUM_OPEN_PATH;
extern std::string VINS_TUM_CLOSED_PATH;
extern std::string IMU_TOPIC;
extern std::string IMAGE_TOPIC; // <-- 新增
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

extern std::string DEPTH_MODEL_PATH;
extern int USE_FAST_INIT;

// --- 深度传感器因子约束参数 (Backend Depth Constraint) ---
// 是否在后端优化中启用深度约束: 0=关闭, 1=开启
extern int ESTIMATE_DEPTH_SCALE_SHIFT;
// 深度尺度因子初始值 (affine transformation: 1/d_metric = a * d_nn + b)
extern double DEPTH_SCALE_A;
// 深度偏移因子初始值
extern double DEPTH_SHIFT_B;
// 权重模式: 0=固定权重, 1=自适应权重
extern int DEPTH_WEIGHT_MODE;
// 深度因子的权重 (影响约束强度) - 仅在 weight_mode=0 时使用
extern double DEPTH_FACTOR_WEIGHT;
// 深度因子的Huber鲁棒核阈值 (用于抑制异常值) - 仅在 weight_mode=0 时使用
extern double DEPTH_FACTOR_HUBER_THRESHOLD;
// 深度融合预热帧数 (Warm-up frames for depth fusion)
extern int DEPTH_FUSION_WARMUP_FRAMES;
// 深度参数随机游走过程噪声 (Random Walk Process Noise)
extern double DEPTH_A_RANDOM_WALK;
extern double DEPTH_B_RANDOM_WALK;

// ========================================================================
// Depth Weight Tuning Parameter (深度权重调参系数)
// ========================================================================
// K 系数用于深度因子权重公式: W = 1.0 / (K × σ + ε)²
// K 越大 → 权重越小 → 深度约束越弱
// K 越小 → 权重越大 → 深度约束越强
// 建议范围: 0.5 ~ 5.0, 默认: 2.0
extern double DEPTH_WEIGHT_K;

// ========================================================================
// Physics-Aware Adaptive Depth Fusion Parameters (Multi-Factor Version)
// ========================================================================
// 基于IMU陀螺仪和加速度计的多因子运动感知自适应权重
extern double DEPTH_WEIGHT_STATIC;      // 静止/慢速运动时的最大权重 (高置信度)
extern double DEPTH_WEIGHT_DYNAMIC;     // 快速运动时的最小权重 (低置信度)
// 物理不变量：Depth Anything V2 网络的特征误差下限 (逆深度单位: 1/m)
extern double PHYSICAL_ERROR_THRESHOLD;
// 综合不稳定性评分阈值 (combined_score = gyro_norm + acc_disturbance)
extern double INSTABILITY_THRESHOLD_LOW;   // 低于此值视为"稳定"
extern double INSTABILITY_THRESHOLD_HIGH;  // 高于此值视为"不稳定"
// 加速度扰动权重因子 (用于平衡陀螺仪和加速度的贡献)
extern double ACC_DISTURBANCE_WEIGHT;

// Fast Init configurable parameters (read from YAML with defaults)
extern int FAST_INIT_MIN_FEATURES;
extern double FAST_INIT_MIN_ACC_VAR;
extern int FAST_INIT_RANSAC_MIN_MEASUREMENTS;
extern int FAST_INIT_RANSAC_MAX_ITERATIONS;
extern double FAST_INIT_RANSAC_THRESHOLD_SQ;
extern int FAST_INIT_RANSAC_MIN_INLIERS;
extern double FAST_INIT_SVD_MIN_SIGMA;
extern double FAST_INIT_COND_THRESHOLD;
extern double FAST_INIT_DEPTH_INV_MIN;
extern double FAST_INIT_DEPTH_INV_MAX;
extern double FAST_INIT_DEPTH_Z_MIN;
extern double FAST_INIT_DEPTH_Z_MAX;
extern int FAST_INIT_MIN_VALID_DEPTH_FEATURES;
extern int FAST_INIT_IRLS_ITERS;
extern double FAST_INIT_IRLS_HUBER_DELTA;
extern double FAST_INIT_REG_LAMBDA_A;
extern double FAST_INIT_REG_LAMBDA_B;
extern double FAST_INIT_REG_LAMBDA_V;
extern double FAST_INIT_EARLY_EXIT_INLIER_RATIO;


void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

// ========================================
// Depth Estimation Module Constants
// ========================================

namespace DepthConstants {
    // Online depth provider settings
    constexpr int ONLINE_DEPTH_QUEUE_SIZE = 5;
    constexpr double DEPTH_RESULT_CACHE_DURATION = 1.0;  // seconds

    // Depth alignment thresholds
    constexpr double MIN_ALIGNMENT_INLIERS = 10.0;
    constexpr double MAX_ALIGNMENT_ERROR = 0.5;

    // Depth computation timeouts
    constexpr int DEPTH_MODEL_READY_TIMEOUT_MS = 5000;
}

