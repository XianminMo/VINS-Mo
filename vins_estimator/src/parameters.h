#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 20;
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
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern std::string IMAGE_TOPIC; // <-- 新增
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

extern std::string DEPTH_MODEL_PATH;
extern int USE_FAST_INIT;

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
