#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/depth_factor.h"  // Hierarchical decoupling depth factor
#include "factor/random_walk_factor.h"  // Random walk for para_a temporal consistency
#include "factor/depth_scale_shift_random_walk_factor.h"  // Legacy (to be removed)

#include <unordered_map>
#include <queue>
#include <fstream>
#include <opencv2/core/eigen.hpp>

// --- MODIFICATION START: Break circular dependency ---
#include "initial/init_depth_provider.h"
#include "initial/initial_fast_mono.h"
#include "depth_estimation/online_depth_provider.h"  // NEW: Modular async depth provider
#include "depth_estimation/depth_model_onnx.h"  // NEW: Shared depth model
#include <memory> // for std::unique_ptr and std::shared_ptr

// Forward declaration for visualization structure (to avoid circular dependency)
struct DepthConstraintDebugInfo;


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, const cv::Mat &raw_image_input);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal - initialization
    void clearState();
    void initDepthEstimator();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    // internal - main processing
    void solveOdometry();
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();

    // internal - depth processing (clean & modular)
    /**
     * @brief 尝试将在线深度提供者的TTA结果存储到ImageFrame（异步非阻塞）
     */
    void tryStoreOnlineDepthToImageFrame();

    /**
     * @brief 为滑动窗口中缺失深度图的帧计算深度（同步推理补充）
     */
    void computeBackendDepthMaps();

    /**
     * @brief 执行前端深度对齐（从在线深度提供者获取结果并对齐）
     */
    void performFrontendDepthAlignment();

    /**
     * @brief 推送原始图像到在线深度提供者（异步处理）
     */
    void pushImageToOnlineDepthProvider(double timestamp, const cv::Mat& raw_image);
    void vector2double();
    void double2vector();
    bool failureDetection();

    bool isDepthEstimatorReady() const {
        return mp_init_depth_provider && mp_init_depth_provider->isReady();
    }

    /**
     * @brief 获取最新帧的深度图（用于可视化）
     * @return 深度图的拷贝，如果不存在则返回空Mat
     */
    cv::Mat getLatestDepthMap() const {
        if (all_image_frame.empty()) {
            return cv::Mat();
        }
        // 获取最新帧
        auto latest_frame = all_image_frame.rbegin();
        if (latest_frame->second.depth_map_computed && !latest_frame->second.predicted_depth_map.empty()) {
            return latest_frame->second.predicted_depth_map.clone();
        }
        return cv::Mat();
    }

    // ========================================================================
    // ⚠️ DEPRECATED FUNCTIONS - 废弃函数声明（仅保留用于参考）
    // ========================================================================
    // 以下函数在 Hierarchical Decoupling Depth Fusion 架构中已废弃
    // 实现已在 estimator.cpp 中用 #if 0 注释
    // 废弃时间：2025-12-14
    // ========================================================================

#if 0  // 废弃函数声明
    /**
     * @brief 【已废弃】在线估计深度尺度偏移参数（初始化完成后调用）
     *
     * ⚠️ **废弃原因**：新架构使用每帧独立的 para_a_global[i]，不再需要全局 a,b 初值
     *
     * 使用当前滑动窗口中的特征点，通过线性回归计算最优的 a, b 参数。
     * 对齐公式：depth_vins = a * depth_net + b
     *
     * 这个函数会在VIO初始化成功后被调用，为后端优化提供更好的初始值。
     * 无论使用快速初始化还是标准SFM初始化，都会执行这个对齐过程。
     */
    void estimateDepthScaleShift();

    /**
     * @brief 【已废弃】确保至少有一帧深度图可用于参数对齐
     * @return true 如果成功计算或已存在深度图
     *
     * ⚠️ **废弃原因**：新架构由 computeBackendDepthMaps() 按需计算深度图
     *
     * 这个函数在初始化完成后调用，确保有足够的深度图数据用于估计a,b参数。
     * 对于快速初始化，第一帧深度图已存在，直接返回true。
     * 对于传统SFM初始化，需要计算一帧深度图（选择特征最多的帧）。
     */
    bool ensureDepthMapForAlignment();
#endif  // 废弃函数声明结束

    // Fast initialization helper methods
    /**
     * @brief 尝试计算窗口第一帧的深度图
     * @return true 如果深度图计算成功或已存在
     *
     * 检查是否需要重新计算深度图（窗口第一帧变化或未计算过），
     * 如果需要则等待深度估计模型就绪并计算深度图。
     */
     bool tryComputeFirstFrameDepth();

     /**
      * @brief 自适应回填策略：遍历滑窗所有帧，尝试对齐未处理的深度
      *
      * **核心思想**:
      * - 不再使用固定200ms滞后查询，而是遍历滑窗中所有未对齐的帧
      * - 对每帧检查其时间戳，查询深度推理结果队列
      * - 如果查到结果：立即执行 SolveLinearAlignment 并标记为已对齐
      * - 如果未查到：跳过，等待下次回填（给推理更多时间）
      *
      * **优势**:
      * - 自适应延迟：无论推理快（30ms）还是慢（300ms），只要帧还在滑窗里就能被利用
      * - 最大化利用率：不会因为固定延迟而浪费早到的结果，也不会丢弃晚到的结果
      * - 鲁棒性：适应不同硬件和负载下的推理速度波动
      *
      * @return 本次回填成功对齐的帧数量
      */
     int backfillDepthAlignment();

     /**
      * @brief 检查快速初始化的条件是否满足
      * @param rot_sum_out 输出累计旋转角度（弧度）
      * @return true 如果满足所有初始化条件
      *
      * 检查特征数量是否足够，以及累计旋转激励是否充分。
      */
     bool checkFastInitConditions(double& rot_sum_out);
 
     /**
      * @brief 执行快速单目初始化
      * @param Ps_init 输出：初始化后的位置（map: frame_id -> position）
      * @param Vs_init 输出：初始化后的速度（map: frame_id -> velocity）
      * @param Rs_init 输出：初始化后的旋转（map: frame_id -> rotation）
      * @return true 如果初始化成功
      * 
      * 调用 FastInitializer 进行初始化，利用深度学习深度图恢复尺度、重力和速度。
      */
     bool performFastInitialization(std::map<int, Eigen::Vector3d>& Ps_init,
                                    std::map<int, Eigen::Vector3d>& Vs_init,
                                    std::map<int, Eigen::Quaterniond>& Rs_init);
 
     /**
      * @brief 将快速初始化的结果更新到估计器状态
      * @param Ps_init 初始化后的位置
      * @param Vs_init 初始化后的速度
      * @param Rs_init 初始化后的旋转
      * 
      * 将 FastInitializer 返回的初始化结果复制到 Estimator 的状态变量中，
      * 并将所有帧标记为关键帧。
      */
     void updateEstimatorStateFromFastInit(const std::map<int, Eigen::Vector3d>& Ps_init,
                                          const std::map<int, Eigen::Vector3d>& Vs_init,
                                          const std::map<int, Eigen::Quaterniond>& Rs_init);

     /**
      * @brief 从快速初始化结果为特征点分配estimated_depth
      *
      * 快速初始化内部计算了参数将网络逆深度转为正深度，这里利用这些参数
      * 和第一帧深度图为滑动窗口中的特征点设置estimated_depth，使得后续的
      * 在线参数估计能够工作。
      */
     void assignEstimatedDepthFromFastInit();

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    // --- Adaptive Backfill: Track which frames have been depth-aligned ---
    bool depth_aligned[(WINDOW_SIZE + 1)];  // 标记每帧是否已完成深度对齐

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // ========================================================================
    // Frame Count Variables (清晰的语义说明)
    // ========================================================================

    /**
     * @brief 滑动窗口内的帧计数
     *
     * **语义说明**:
     * - 初始化阶段(solver_flag == INITIAL): 从0递增到WINDOW_SIZE,表示"当前累积的帧数"
     * - 窗口满后(frame_count == WINDOW_SIZE): 固定为WINDOW_SIZE,不再变化
     *
     * **使用场景**:
     * - 初始化前: 判断窗口是否已满 (if frame_count == WINDOW_SIZE)
     * - 窗口满后: 作为"当前帧在窗口中的索引" (永远等于WINDOW_SIZE)
     *
     * ⚠️ 注意: 窗口满后,frame_count 实际上等价于 WINDOW_SIZE 常量
     */
    int frame_count;

    int global_frame_count;       // 全局帧计数器(用于预热策略等,持续增长)
    int depth_fusion_frame_count; // 深度融合专用帧计数器(从VINS初始化成功后开始计数)

    /**
     * @brief 获取当前帧在滑动窗口中的索引
     *
     * 这个辅助函数使代码意图更清晰:
     * - 窗口未满时: 返回 frame_count (当前累积帧数,即新帧索引)
     * - 窗口满后: 返回 WINDOW_SIZE (新帧总在最后一个位置)
     *
     * 等价于: (frame_count < WINDOW_SIZE) ? frame_count : WINDOW_SIZE
     * 实际上就是: frame_count (因为窗口满后 frame_count == WINDOW_SIZE)
     *
     * @return 当前帧在窗口中的索引 (0 到 WINDOW_SIZE)
     */
    inline int getCurrentFrameIndex() const {
        return frame_count;
    }
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    // --- Dual-Parameter Depth Fusion Parameters ---
    // Per-frame scale and shift parameters for inverse depth space refinement
    // Residual: r = lambda_vins - (a * lambda_aligned + b)
    // where lambda = 1/depth (inverse depth space)
    double para_depth_scale[WINDOW_SIZE + 1][1];  // Per-frame scale refinement (a)
    double para_depth_shift[WINDOW_SIZE + 1][1];  // Per-frame shift refinement (b)

    // Legacy depth parameters (DELETED - no longer used)
    // double para_a_global[WINDOW_SIZE + 1][1];  // OLD: single parameter
    // double para_DepthScaleShift[1][2];  // OLD: global [a, b]
    // double last_depth_a;  // OLD
    // double last_depth_b;  // OLD

    bool has_last_depth_params;  // 标记是否有上一次的参数值
    bool is_first_depth_optimization;  // 标记是否是第一次深度优化（用于放松随机游走约束）

    // --- Signal Filtering for Depth Fusion Stability ---
    // 存储平滑后的运动不稳定性评分（低通滤波后）
    double smoothed_instability_score;  // EMA-filtered motion score (gyro + acc)
    bool is_score_initialized;          // 标记评分是否已初始化（首帧处理）

    // --- Dynamic Warmup: IMU Bias Convergence Check ---
    bool is_depth_fusion_ready;  // 标记深度融合是否就绪（一旦为true就锁定）

    /**
     * @brief 检查系统稳定性（IMU偏置是否收敛）
     * @return true 如果加速度偏置标准差 < 0.1
     *
     * 在滑动窗口中计算 Bas（加速度偏置）的标准差，
     * 如果 std_dev(Bas) < 0.1 则认为系统已稳定，可以启用深度融合。
     * 该状态一旦为true就会锁定（is_depth_fusion_ready标志位），不会再变回false。
     */
    bool checkSystemStability();

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info = nullptr;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration = nullptr;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;

    // deep estimation module
    // Depth estimation modules (shared architecture)
    std::shared_ptr<depth_estimation::DepthModelONNX> mp_depth_model;  // Shared ONNX model
    std::unique_ptr<InitDepthProvider> mp_init_depth_provider;  // Init depth provider
    std::unique_ptr<depth_estimation::OnlineDepthProvider> mp_online_depth_provider;  // Online depth provider
    cv::Mat m_first_frame_depth_map; // 存储第一帧的归一化逆深度图
    bool m_first_frame_depth_computed;
    std::mutex m_depth_mutex;

    std::unique_ptr<FastInitializer> mp_fast_initializer;
    int m_depth_window_start_id = -1;
    std::atomic<bool> m_depth_estimator_ready{false};

    // Depth fusion logging
    std::ofstream depth_fusion_log_file;
    int log_frame_counter = 0;
    void logDepthFusionMetrics(int frame_id, double gyro_norm, double acc_disturbance,
                               double raw_score, double smoothed_score, double weight,
                               double huber_threshold, double scale_a, double shift_b);

    // para_a_global logging (for plotting)
    std::ofstream para_a_log_file;
    void logParaAGlobal();  // Log all para_a_global values in current window

    // Balance ratio logging (for tuning K parameter)
    std::ofstream balance_ratio_log_file;
    void logBalanceRatio(double balance_ratio, double avg_visual_cost, double avg_depth_cost,
                        int visual_count, int depth_count, double current_K);

    // Depth constraint visualization
    std::vector<DepthConstraintDebugInfo> depth_constraint_debug_info;  // Store debug info for visualization
};
