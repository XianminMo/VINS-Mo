#pragma once

#include <vector>
#include <map>
#include <random> // 用于 RANSAC 采样

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp> // 用于 Eigen 和 OpenCV 转换 (如果需要)

// 包含 VINS-Mono 的核心头文件，我们需要访问 Estimator 内部数据
#include "../parameters.h"
#include "../feature_manager.h"
#include "../factor/integration_base.h" // IMU 预积分
#include "../utility/utility.h"       // skewSymmetric, g2R 等工具函数
#include "initial_alignment.h"        // 为了 ImageFrame 的定义

/**
 * @brief RANSAC 算法的相关参数
 */
namespace fast_mono
{
    const int RANSAC_MIN_MEASUREMENTS = 4; 
    const int RANSAC_MAX_ITERATIONS = 500;  // 增加迭代次数，从250改为500
    const double RANSAC_THRESHOLD_SQ = 0.01 * 0.01; // 从1.0*1.0改为0.01*0.01 (更严格)
    const int RANSAC_MIN_INLIERS = 20;   // 从10改为20 (需要更多内点，提高鲁棒性)
} // namespace fast_mono


/**
 * @class FastInitializer
 * @brief 实现了《Fast Monocular Visual-Inertial Initialization Leveraging Learned Single-View Depth》
 * 论文中提出的快速初始化方法的核心逻辑。
 */
class FastInitializer
{
public:
    /**
     * @brief 构造函数
     * @param f_manager_ptr 指向 Estimator 中 FeatureManager 实例的指针
     */
    FastInitializer(FeatureManager* f_manager_ptr);

    /**
     * @brief 尝试使用快速单目初始化方法初始化 VIO 系统。
     * @param image_frames [in] 包含滑动窗口内所有图像帧信息的 map (时间戳 -> ImageFrame)。
     * @param first_frame_norm_inv_depth [in] 第一帧的归一化逆深度图 (CV_32F, 范围 [0, 1])。
     * @param G_gravity_world [out] 如果初始化成功，将被设置为估计出的重力向量 (在 W' 系，即重力对齐系)。
     * @param Ps_out [out] 如果初始化成功，将被填充为窗口内各帧的初始位置 (在 W' 系)。
     * @param Vs_out [out] 如果初始化成功，将被填充为窗口内各帧的初始速度 (在 W' 系)。
     * @param Rs_out [out] 如果初始化成功，将被填充为窗口内各帧的初始姿态 (从 W' 系到 Body 系的旋转)。
     * @return true 如果初始化成功，false 如果失败。
     */
    bool initialize(const std::map<double, ImageFrame>& image_frames,
                    const cv::Mat& first_frame_norm_inv_depth,
                    Eigen::Vector3d& G_gravity_world,
                    std::map<int, Eigen::Vector3d>& Ps_out,
                    std::map<int, Eigen::Vector3d>& Vs_out,
                    std::map<int, Eigen::Quaterniond>& Rs_out);


private:
    // 计算滑窗起始帧ID：返回当前窗口内所有特征 start_frame 的最小值（若为空返回0）
    int computeWindowStartFrameId() const;

    /**
     * @struct ObservationData
     * @brief 用于存储一次 (特征点 i, 图像帧 k) 观测所需的所有数据，方便 RANSAC 处理。
     */
    struct ObservationData
    {
        int feature_id;                 ///< 特征点 ID
        int frame_k_index;              ///< 观测帧在窗口内的索引 (0 到 WINDOW_SIZE)
        IntegrationBase* pre_integration_k; ///< 从第一帧(I0)到当前帧(Ik)的 IMU 预积分对象 $\Delta_{I_0}^{I_k}$
        Eigen::Vector3d z_i0;           ///< 特征点 i 在第一帧 C0 中的归一化坐标 \bar{z}_{i,0} (单位向量 [x, y, 1])
        Eigen::Vector3d z_ik;           ///< 特征点 i 在第 k 帧 Ck 中的归一化坐标 \bar{z}_{i,k} (单位向量 [x, y, 1])
        double d_hat_i;                 ///< 特征点 i 在第一帧的归一化逆深度值 $\hat{d}_i$ (来自深度网络)
    };

    /**
     * @brief 根据论文公式 (23)，为一个 (i, k) 观测构建线性系统 A'x' = b' 的两行。
     * @details M1*a + M2*b + Tv*v_I0 + Tg*g + Tc = 0  (注意论文公式有误，Tc项前应为正号)
     * 重排为 [M1, M2, Tv, Tg] * [a, b, v_I0, g]^T = -Tc
     * @param pre_int_k [in] IMU 预积分 $\Delta_{I_0}^{I_k}$
     * @param z_i0 [in] 特征在 C0 中的归一化坐标
     * @param z_ik [in] 特征在 Ck 中的归一化坐标
     * @param d_hat_i [in] 特征的归一化逆深度
     * @param A_row [out] 填充的 A' 矩阵的 2x8 行
     * @param b_row [out] 填充的 b' 向量的 2x1 行
     */
    void buildLinearSystemRow(const IntegrationBase* pre_int_k,
                              const Eigen::Vector3d& z_i0,
                              const Eigen::Vector3d& z_ik,
                              double d_hat_i,
                              Eigen::Matrix<double, 2, 8>& A_row,
                              Eigen::Vector2d& b_row);

    /**
     * @brief 实现论文中的 Algorithm 1，使用 RANSAC 鲁棒地求解线性系统 A'x' = b'。
     * @param all_observations [in] 包含窗口内所有有效观测的向量。
     * @param best_x [out] 如果 RANSAC 成功，将被设置为最优解 x' = [a, b, v_I0, g_I0]^T (在 I0 系)。
     * @return true 如果 RANSAC 找到一个足够好的解，false 如果失败。
     */
    bool solveRANSAC(const std::vector<ObservationData>& all_observations,
                     Eigen::Matrix<double, 8, 1>& best_x);

    /**
     * @brief 使用给定的一组观测（通常是 RANSAC 的内点）求解线性系统 A'x' = b'。
     * @details 采用 "先无约束求解，再修正重力" 的策略。
     * 1. 使用 SVD 求解 A'x' = b' 得到 x'_svd。
     * 2. 提取重力 g_svd，将其归一化到 G (e.g., 9.81) 的大小得到 g_normed。
     * 3. 固定 g_normed，重新求解 A_y * y = b' - A_g * g_normed，得到 y = [a, b, v_I0]^T。
     * @param observations [in] 用于求解的观测数据。
     * @param x_out [out] 计算得到的解 x' = [a, b, v_I0, g_normed]^T。
     * @return true 如果求解成功 (例如 SVD 稳定)，false 如果失败。
     */
    bool solveLinearSystem(const std::vector<ObservationData>& observations,
                           Eigen::Matrix<double, 8, 1>& x_out);
    
    /**
    * @brief 验证归一化逆深度值的有效性
    * @param d_hat_inv 归一化逆深度值
    * @return true 如果深度值有效，false 否则
    */
    static bool isValidDepth(double d_hat_inv);
    
    /**
    * @brief 验证像素坐标是否在图像范围内
    * @param u 像素坐标 x
    * @param v 像素坐标 y
    * @param rows 图像行数
    * @param cols 图像列数
    * @return true 如果坐标有效，false 否则
    */
    static bool isValidPixelCoord(int u, int v, int rows, int cols);
    
    /**
    * @brief 验证归一化特征坐标的有效性
    * @param z 归一化坐标向量 [x, y, 1]
    * @return true 如果坐标有效，false 否则
    */
    static bool isValidNormalizedCoord(const Eigen::Vector3d& z);
    
    /**
    * @brief 从深度图获取深度值并验证
    * @param depth_map 归一化逆深度图
    * @param u 像素坐标 x
    * @param v 像素坐标 y
    * @param d_out [out] 输出的深度值
    * @return true 如果成功获取有效深度值，false 否则
    */
    static bool getValidDepthFromMap(const cv::Mat& depth_map, int u, int v, double& d_out);
    
    /**
    * @brief 计算复合 IMU 预积分（从第一帧 I0 到当前帧 Ik）
    * @param image_frames 所有图像帧的映射
    * @param pre_integrations_out [out] 输出的复合预积分向量，索引对应窗口内帧索引
    * @return true 如果成功，false 如果失败（如缺少预积分数据）
    */
    bool computeCompoundPreIntegrations(
        const std::map<double, ImageFrame>& image_frames,
        std::vector<IntegrationBase*>& pre_integrations_out);
    
    /**
    * @brief 收集所有有效的特征观测数据
    * @param depth_map 第一帧的归一化逆深度图
    * @param window_start_frame_id 窗口起始帧ID
    * @param pre_integrations_compound 复合预积分向量
    * @param observations_out [out] 输出的观测数据向量
    * @return 收集到的有效观测数量
    */
    int collectValidObservations(
        const cv::Mat& depth_map,
        int window_start_frame_id,
        const std::vector<IntegrationBase*>& pre_integrations_compound,
        std::vector<ObservationData>& observations_out);
    
    /**
    * @brief 验证 RANSAC 解的基本物理合理性
    * @param x 解向量 [a, b, v_I0, g_I0]^T
    * @return true 如果解合理，false 否则
    */
    static bool isValidSolution(const Eigen::Matrix<double, 8, 1>& x);
    
    /**
    * @brief 计算所有特征点的深度统计信息
    * @param depth_map 归一化逆深度图
    * @param a 深度尺度因子
    * @param b 深度偏移
    * @param window_start_frame_id 窗口起始帧ID
    * @param stats_out [out] 输出统计信息结构体
    * @return true 如果统计成功
    */
    struct DepthStatistics {
        int total_count = 0;      // 总特征点数
        int valid_count = 0;       // 有效深度特征点数（0.1~50m范围内）
        double min_depth = std::numeric_limits<double>::infinity();
        double max_depth = -std::numeric_limits<double>::infinity();
        double mean_depth = 0.0;
    };
    bool computeDepthStatistics(
        const cv::Mat& depth_map,
        double a, double b,
        int window_start_frame_id,
        DepthStatistics& stats_out);
    
    /**
    * @brief 执行坐标系对齐：从 I0 系转换到 W' 重力对齐系
    * @param g_in_I0 重力向量在 I0 系下的表示
    * @param v_I0_in_I0 速度向量在 I0 系下的表示
    * @param G_gravity_world [out] 重力向量在 W' 系下的表示
    * @param R_I0_to_W_prime [out] 从 I0 到 W' 的旋转
    * @param v_I0_in_W_prime [out] 速度向量在 W' 系下的表示
    */
    void alignCoordinateSystem(
        const Eigen::Vector3d& g_in_I0,
        const Eigen::Vector3d& v_I0_in_I0,
        Eigen::Vector3d& G_gravity_world,
        Eigen::Quaterniond& R_I0_to_W_prime,
        Eigen::Vector3d& v_I0_in_W_prime);
    
    /**
    * @brief 使用 IMU 预积分前向传播状态到所有帧
    * @param image_frames 所有图像帧的映射
    * @param R_I0_to_W_prime 第一帧的旋转（I0 到 W'）
    * @param p_I0_in_W_prime 第一帧的位置（在 W' 系下）
    * @param v_I0_in_W_prime 第一帧的速度（在 W' 系下）
    * @param G_gravity_world 重力向量（在 W' 系下）
    * @param Ps_out [out] 输出的位置向量
    * @param Vs_out [out] 输出的速度向量
    * @param Rs_out [out] 输出的旋转向量
    * @return true 如果成功，false 如果失败
    */
    bool propagateStatesToAllFrames(
        const std::map<double, ImageFrame>& image_frames,
        const Eigen::Quaterniond& R_I0_to_W_prime,
        const Eigen::Vector3d& p_I0_in_W_prime,
        const Eigen::Vector3d& v_I0_in_W_prime,
        const Eigen::Vector3d& G_gravity_world,
        std::map<int, Eigen::Vector3d>& Ps_out,
        std::map<int, Eigen::Vector3d>& Vs_out,
        std::map<int, Eigen::Quaterniond>& Rs_out);
    
    
    
    // 成员变量 
    FeatureManager* m_feature_manager; ///< 指向 Estimator 中 FeatureManager 的指针，用于访问特征信息
    std::mt19937 m_random_generator;   ///< 用于 RANSAC 的随机数生成器
};