#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string VINS_TUM_RESULT_PATH;
std::string IMU_TOPIC;
std::string IMAGE_TOPIC; // <-- 新增
double ROW, COL;
double TD, TR;
std::string DEPTH_MODEL_PATH;
int USE_FAST_INIT; // <-- 添加这行定义

// Fast Init parameters (defaults will be overridden by YAML if provided)
int FAST_INIT_MIN_FEATURES;
double FAST_INIT_MIN_ACC_VAR;
int FAST_INIT_RANSAC_MIN_MEASUREMENTS;
int FAST_INIT_RANSAC_MAX_ITERATIONS;
double FAST_INIT_RANSAC_THRESHOLD_SQ;
int FAST_INIT_RANSAC_MIN_INLIERS;
double FAST_INIT_SVD_MIN_SIGMA;
double FAST_INIT_COND_THRESHOLD;
double FAST_INIT_DEPTH_INV_MIN;
double FAST_INIT_DEPTH_INV_MAX;
double FAST_INIT_DEPTH_Z_MIN;
double FAST_INIT_DEPTH_Z_MAX;
int FAST_INIT_MIN_VALID_DEPTH_FEATURES;
int FAST_INIT_IRLS_ITERS;
double FAST_INIT_IRLS_HUBER_DELTA;
double FAST_INIT_REG_LAMBDA_A;
double FAST_INIT_REG_LAMBDA_B;
double FAST_INIT_REG_LAMBDA_V;
double FAST_INIT_EARLY_EXIT_INLIER_RATIO;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["image_topic"] >> IMAGE_TOPIC; // <-- 新增

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    VINS_TUM_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.tum";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
    std::ofstream fout_tum(VINS_TUM_RESULT_PATH, std::ios::out);
    fout_tum.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }

    fsSettings["use_fast_init"] >> USE_FAST_INIT;
    fsSettings["depth_model_path"] >> DEPTH_MODEL_PATH;
    if (USE_FAST_INIT)
    {
        ROS_INFO("Fast Monocular Initialization ENABLED.");
        if (DEPTH_MODEL_PATH.empty())
        {
            ROS_FATAL("Fast Init is enabled, but 'deep_model' path is not set in config file!");
        }
        else
        {
            std::ifstream f(DEPTH_MODEL_PATH.c_str());
            if (!f.good())
            {
                ROS_FATAL("Fast Init enabled, but deep_model file not found at: %s", DEPTH_MODEL_PATH.c_str());
            }
            else
            {
                ROS_INFO("Fast Init will use model: %s", DEPTH_MODEL_PATH.c_str());
            }
        }
    }
    else
    {
        ROS_INFO("Using standard VINS-Mono SFM Initialization.");
    }
    
    // Helper to read a value with default
    auto readOr = [&](const std::string &key, auto def_val)
    {
        cv::FileNode node = fsSettings[key];
        if (!node.empty())
        {
            decltype(def_val) v; node >> v; return v;
        }
        return def_val;
    };

    // Load Fast Init parameters with sensible defaults
    FAST_INIT_MIN_FEATURES = readOr("fast_init.min_features", 50);
    FAST_INIT_MIN_ACC_VAR = readOr("fast_init.min_acc_var", 0.2);
    FAST_INIT_RANSAC_MIN_MEASUREMENTS = readOr("fast_init.ransac.min_measurements", 6);
    FAST_INIT_RANSAC_MAX_ITERATIONS = readOr("fast_init.ransac.max_iterations", 250);
    // residual threshold is pixel residual; store squared as used by code
    double ransac_residual_thresh_px = readOr("fast_init.ransac.residual_thresh_px", 0.01);
    FAST_INIT_RANSAC_THRESHOLD_SQ = ransac_residual_thresh_px * ransac_residual_thresh_px;
    FAST_INIT_RANSAC_MIN_INLIERS = readOr("fast_init.ransac.min_inliers", 20);
    FAST_INIT_SVD_MIN_SIGMA = readOr("fast_init.svd.min_sigma", 1e-8);
    FAST_INIT_COND_THRESHOLD = readOr("fast_init.cond.threshold", 3e5);
    FAST_INIT_DEPTH_INV_MIN = readOr("fast_init.depth.inv_min", 1e-6);
    FAST_INIT_DEPTH_INV_MAX = readOr("fast_init.depth.inv_max", 10.0);
    FAST_INIT_DEPTH_Z_MIN = readOr("fast_init.depth.z_min", 0.1);
    FAST_INIT_DEPTH_Z_MAX = readOr("fast_init.depth.z_max", 50.0);
    FAST_INIT_MIN_VALID_DEPTH_FEATURES = readOr("fast_init.depth.min_valid_features", 10);
    FAST_INIT_IRLS_ITERS = readOr("fast_init.irls.iters", 3);
    FAST_INIT_IRLS_HUBER_DELTA = readOr("fast_init.irls.huber_delta", 1.5e-2);
    FAST_INIT_REG_LAMBDA_A = readOr("fast_init.reg.lambda_a", 1e-2);
    FAST_INIT_REG_LAMBDA_B = readOr("fast_init.reg.lambda_b", 1e-2);
    FAST_INIT_REG_LAMBDA_V = readOr("fast_init.reg.lambda_v", 1e-3);
    FAST_INIT_EARLY_EXIT_INLIER_RATIO = readOr("fast_init.ransac.early_exit_inlier_ratio", 0.7);

    ROS_INFO("FastInit Params: min_features=%d, min_acc_var=%.3f, ransac[min_meas=%d, max_iter=%d, thr_px=%.4f, min_inliers=%d], svd_min=%.1e, cond_thr=%.1e",
             FAST_INIT_MIN_FEATURES, FAST_INIT_MIN_ACC_VAR, FAST_INIT_RANSAC_MIN_MEASUREMENTS,
             FAST_INIT_RANSAC_MAX_ITERATIONS, ransac_residual_thresh_px, FAST_INIT_RANSAC_MIN_INLIERS,
             FAST_INIT_SVD_MIN_SIGMA, FAST_INIT_COND_THRESHOLD);
    ROS_INFO("FastInit Depth: inv[%g,%g], z[%g,%g], min_valid=%d",
             FAST_INIT_DEPTH_INV_MIN, FAST_INIT_DEPTH_INV_MAX, FAST_INIT_DEPTH_Z_MIN, FAST_INIT_DEPTH_Z_MAX,
             FAST_INIT_MIN_VALID_DEPTH_FEATURES);
    ROS_INFO("FastInit IRLS: iters=%d, huber=%.3e, reg[a=%.2e, b=%.2e, v=%.2e], early_exit_ratio=%.2f",
             FAST_INIT_IRLS_ITERS, FAST_INIT_IRLS_HUBER_DELTA, FAST_INIT_REG_LAMBDA_A,
             FAST_INIT_REG_LAMBDA_B, FAST_INIT_REG_LAMBDA_V, FAST_INIT_EARLY_EXIT_INLIER_RATIO);

    fsSettings.release();
}
