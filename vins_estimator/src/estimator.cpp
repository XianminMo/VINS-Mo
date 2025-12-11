#include "estimator.h"
#include "initial/initial_fast_mono.h" // Include full definition here

// Estimator类的构造函数
Estimator::Estimator(): f_manager{Rs}
{
    // 打印初始化开始的日志信息
    ROS_INFO("init begins");
    // 调用clearState()函数，重置所有状态变量和参数
    clearState();
    // 初始化标志，表示第一帧的深度图尚未计算
    m_first_frame_depth_computed = false;
    // 初始化日志计数器
    log_frame_counter = 0;
}

/**
 * @brief 初始化深度估计器（如果启用）
 * 这个函数应该在 readParameters() 之后被调用
 */
void Estimator::initDepthEstimator()
{
    // 如果启用了快速初始化或者启用了深度约束后端，则需要初始化深度估计器
    if (USE_FAST_INIT || ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        ROS_INFO("Initializing DepthEstimator asynchronously...");

        // 创建一个深度估计器的智能指针实例
        mp_depth_estimator = std::make_unique<DepthEstimator>();

        // 使用配置文件中指定的模型路径初始化深度估计器
        if (!mp_depth_estimator->initAsync(DEPTH_MODEL_PATH))
        {
            // 如果初始化失败
            // 打印致命错误日志，提示用户检查模型路径和相关配置（如ONNX, CUDA）
            ROS_FATAL("DepthEstimator initialization failed! Please check the model path and ONNX/CUDA configuration.");
            // 可以在此处添加 ros::shutdown() 来终止节点，防止进一步错误
            ros::shutdown();
        }

        // 注意：这里不等待模型加载完成，让它在后台进行
    }

    // 快速初始化器只在USE_FAST_INIT为true时初始化
    if (USE_FAST_INIT)
    {
        ROS_INFO("Initializing FastInitializer...");
        mp_fast_initializer = std::make_unique<FastInitializer>(&f_manager);
    }
}


/**
 * @brief 设置VINS系统的参数
 * 
 * 从全局参数（通常在parameters.h中定义，由配置文件加载）中读取
 * IMU与相机之间的外参、特征投影的误差信息矩阵以及IMU与相机的时间延迟。
 */
void Estimator::setParameter()
{
    // 遍历所有相机
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 从全局变量 TIC 和 RIC 读取相机到IMU的外参（平移和旋转）
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    // 将外参旋转矩阵设置给特征管理器，用于后续的特征点三角化和坐标变换
    f_manager.setRic(ric);
    // 设置视觉重投影误差的协方差矩阵的逆（信息矩阵）的平方根
    // FOCAL_LENGTH / 1.5 是一个经验值，用于调节视觉测量在优化中的权重
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    // 如果考虑时间延迟td，同样设置其投影因子的信息矩阵
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    // 设置IMU和相机之间的时间戳延迟
    td = TD;

    // 初始化深度融合日志文件
    if (ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        std::string log_path = VINS_RESULT_PATH + "/depth_fusion_metrics.csv";
        depth_fusion_log_file.open(log_path, std::ios::out);
        if (depth_fusion_log_file.is_open())
        {
            // 写入CSV头部
            depth_fusion_log_file << "frame_id,gyro_norm,acc_disturbance,raw_score,smoothed_score,"
                                  << "weight,huber_threshold,scale_a,shift_b,weight_mode\n";
            ROS_INFO("Depth fusion metrics log file opened: %s", log_path.c_str());
        }
        else
        {
            ROS_WARN("Failed to open depth fusion metrics log file: %s", log_path.c_str());
        }
    }
}

/**
 * @brief 重置或清空整个VIO系统的状态
 * 
 * 将所有状态变量（位姿、速度、偏置）、缓存、标志位等恢复到初始状态。
 * 在系统启动或发生故障重启时调用。
 */
void Estimator::clearState()
{
    // 遍历滑动窗口中的所有帧（包括当前帧和历史帧）
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 姿态旋转矩阵初始化为单位矩阵
        Rs[i].setIdentity();
        // 位置向量初始化为零
        Ps[i].setZero();
        // 速度向量初始化为零
        Vs[i].setZero();
        // 加速度计偏置初始化为零
        Bas[i].setZero();
        // 陀螺仪偏置初始化为零
        Bgs[i].setZero();
        // 清空IMU数据缓存
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        // 如果预积分对象存在，则释放内存
        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    // 重置相机到IMU的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    // 遍历并清空所有历史图像帧信息
    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    // 重置系统状态标志
    solver_flag = INITIAL; // 求解器状态设置为初始化阶段
    first_imu = false,     // 标记尚未收到第一帧IMU数据
    sum_of_back = 0;       // 边缘化旧帧的计数器
    sum_of_front = 0;      // 边缘化新帧的计数器
    frame_count = 0;       // 滑动窗口中的当前帧计数
    global_frame_count = 0; // 全局帧计数器（持续增长）
    depth_fusion_frame_count = 0; // 深度融合帧计数器（从VINS初始化成功后开始计数）
    initial_timestamp = 0; // 初始化时间戳
    all_image_frame.clear(); // 清空所有图像帧数据
    td = TD;               // 重置IMU和相机之间的时间偏移

    // 释放临时预积分对象和边缘化信息
    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    // 清空特征管理器中的所有特征点状态
    f_manager.clearState();

    // --- 深度传感器因子约束状态初始化 (Backend Depth Constraint) ---
    // 从全局配置参数初始化深度仿射变换参数
    para_DepthScaleShift[0][0] = DEPTH_SCALE_A;  // 尺度因子 a
    para_DepthScaleShift[0][1] = DEPTH_SHIFT_B;  // 偏移因子 b

    // 初始化随机游走相关变量
    last_depth_a = DEPTH_SCALE_A;
    last_depth_b = DEPTH_SHIFT_B;
    has_last_depth_params = false;  // 系统刚启动，没有"上一次"的值
    is_first_depth_optimization = true;  // 标记为第一次优化，允许大幅跳转

    // 初始化信号滤波状态（运动评分平滑）
    smoothed_instability_score = 0.0;
    is_score_initialized = false;

    // 添加对我们新成员变量的重置
    {
        // 使用互斥锁保护深度图相关成员变量的访问，确保线程安全
        std::lock_guard<std::mutex> lock(m_depth_mutex);
        m_first_frame_depth_computed = false; // 标记第一帧深度图未计算
        m_first_frame_depth_map.release();    // 释放深度图内存
    }

    // 关闭日志文件（如果打开）
    if (depth_fusion_log_file.is_open())
    {
        depth_fusion_log_file.flush();
        depth_fusion_log_file.close();
        ROS_INFO("Depth fusion metrics log file closed");
    }
    log_frame_counter = 0;

    failure_occur = 0; // 失败标志位清零
    relocalization_info = 0; // 重定位信息标志位清零

    // 漂移校正参数初始化
    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief 确保至少有一帧深度图可用于参数对齐
 * @return true 如果成功计算或已存在深度图
 *
 * 这个函数在初始化完成后调用，确保有足够的深度图数据用于估计a,b参数。
 *
 * **工作流程**：
 * 1. 检查是否启用了深度约束，如果未启用则直接返回false
 * 2. 检查滑动窗口中是否已有深度图（通过遍历all_image_frame）
 * 3. 如果已有深度图，直接返回true（快速初始化路径会走到这里）
 * 4. 如果没有深度图（传统SFM初始化路径），则计算一帧：
 *    - 选择当前帧（WINDOW_SIZE）或特征最多的帧
 *    - 等待深度估计模型就绪
 *    - 推理深度图并存储
 * 5. 返回是否成功
 *
 * **设计理由**：
 * - 快速初始化已经计算了第一帧深度图，无需重复计算
 * - 传统SFM初始化没有深度图，需要计算一帧
 * - 只计算一帧（~50-100ms），避免推理多帧的高延迟
 * - 一帧的特征点（~150个）足以求解2个参数（a, b）
 */
bool Estimator::ensureDepthMapForAlignment()
{
    // 1. 检查是否启用深度约束
    if (!ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        return false;
    }

    // 2. 检查深度估计器是否可用
    if (!mp_depth_estimator || !mp_depth_estimator->isReady())
    {
        ROS_WARN("[Depth Alignment] Depth estimator not ready, skipping depth map computation.");
        return false;
    }

    // 3. 检查滑动窗口中是否已有深度图
    int frames_with_depth = 0;
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        double timestamp = Headers[i].stamp.toSec();
        auto frame_it = all_image_frame.find(timestamp);
        if (frame_it != all_image_frame.end() && frame_it->second.depth_map_computed)
        {
            frames_with_depth++;
        }
    }

    if (frames_with_depth > 0)
    {
        ROS_INFO("[Depth Alignment] Found %d frames with depth maps, no need to compute.", frames_with_depth);
        return true;  // 已有深度图（快速初始化路径）
    }

    // 4. 没有深度图，需要计算一帧（传统SFM初始化路径）
    ROS_INFO("[Depth Alignment] No depth maps found, computing one frame for parameter alignment...");

    // 选择当前帧（WINDOW_SIZE）- 通常特征较多
    int selected_frame_id = WINDOW_SIZE;
    double timestamp = Headers[selected_frame_id].stamp.toSec();
    auto frame_it = all_image_frame.find(timestamp);

    if (frame_it == all_image_frame.end())
    {
        ROS_WARN("[Depth Alignment] Cannot find frame %d in all_image_frame.", selected_frame_id);
        return false;
    }

    auto& frame = frame_it->second;

    // 检查原始图像是否可用
    if (frame.raw_image.empty())
    {
        ROS_WARN("[Depth Alignment] Frame %d has no raw image, cannot compute depth.", selected_frame_id);
        return false;
    }

    // 计算深度图
    TicToc t_depth;
    cv::Mat depth_map;
    if (!mp_depth_estimator->predict(frame.raw_image, depth_map))
    {
        ROS_WARN("[Depth Alignment] Depth prediction failed for frame %d.", selected_frame_id);
        return false;
    }

    // 存储深度图
    frame.predicted_depth_map = depth_map;
    frame.depth_map_computed = true;

    ROS_INFO("[Depth Alignment] Computed depth map for frame %d (%.2f ms).",
             selected_frame_id, t_depth.toc());

    return true;
}

/**
 * @brief 在线估计深度尺度偏移参数（初始化完成后调用）
 *
 * 通过线性回归对齐VINS三角化深度和网络预测深度：
 * depth_vins = a * depth_net + b
 *
 * 算法流程：
 * 1. 遍历滑动窗口中所有已成功三角化的特征点
 * 2. 对每个特征点，找到对应帧的网络预测深度值
 * 3. 构建线性最小二乘系统并求解
 * 4. 更新全局参数 para_DepthScaleShift 和随机游走历史值
 */
void Estimator::estimateDepthScaleShift()
{
    if (!ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        ROS_WARN("[Depth Init] Depth constraint is disabled, skipping online alignment.");
        return;
    }

    // 数据收集：配对 (depth_net, depth_vins)
    std::vector<double> depth_net_vec;
    std::vector<double> depth_vins_vec;

    int features_checked = 0;
    int features_with_depth = 0;

    // 诊断计数器
    int features_triangulated = 0;
    int features_valid_depth_vins = 0;
    int features_with_observations = 0;
    int obs_frames_checked = 0;
    int obs_frames_with_depth_map = 0;
    int obs_out_of_bounds = 0;
    int obs_invalid_depth_net = 0;

    // 遍历特征管理器中的所有特征点
    for (auto &it_per_id : f_manager.feature)
    {
        features_checked++;

        // 使用有有效深度的特征点（不依赖solve_flag）
        // 三角化后estimated_depth会被设置，但solve_flag可能仍为0
        if (it_per_id.estimated_depth <= 0.0 || !std::isfinite(it_per_id.estimated_depth))
            continue;

        features_triangulated++;

        // 获取VINS三角化的深度，转换为逆深度
        double depth_vins = it_per_id.estimated_depth;

        // 过滤不合理的深度值
        if (depth_vins < 0.01 || depth_vins > 100.0 || !std::isfinite(depth_vins))
        {
            static int depth_filter_count = 0;
            if (depth_filter_count < 10) {
                ROS_INFO("[Depth Init] Filtered feature with depth_vins=%.3f (out of range [0.01, 100.0])", depth_vins);
                depth_filter_count++;
            }
            continue;
        }

        // 转换为逆深度：这是我们要拟合的VINS逆深度
        double inv_depth_vins = 1.0 / depth_vins;

        features_valid_depth_vins++;

        // 遍历特征点的所有观测帧，查找有深度图的帧
        // 修复：不只检查首次观测帧，而是遍历所有观测帧
        bool found_valid_depth = false;
        int feature_obs_count = it_per_id.feature_per_frame.size();

        if (feature_obs_count > 0)
            features_with_observations++;

        for (int obs_idx = 0; obs_idx < feature_obs_count && !found_valid_depth; obs_idx++)
        {
            obs_frames_checked++;

            int frame_id = it_per_id.start_frame + obs_idx;
            if (frame_id < 0 || frame_id >= WINDOW_SIZE + 1)
                continue;

            // 获取特征点在该观测帧中的像素坐标
            const auto& feature_per_frame = it_per_id.feature_per_frame[obs_idx];
            Eigen::Vector2d uv = feature_per_frame.uv;

            // 找到对应的ImageFrame
            double timestamp = Headers[frame_id].stamp.toSec();
            auto frame_it = all_image_frame.find(timestamp);

            if (frame_it == all_image_frame.end())
                continue;

            const auto& image_frame = frame_it->second;

            // 检查该帧是否有深度图
            if (!image_frame.depth_map_computed || image_frame.predicted_depth_map.empty())
                continue;

            obs_frames_with_depth_map++;

            const cv::Mat& depth_map = image_frame.predicted_depth_map;

            // 边界检查
            int u = static_cast<int>(uv.x() + 0.5);
            int v = static_cast<int>(uv.y() + 0.5);

            if (v < 0 || v >= depth_map.rows || u < 0 || u >= depth_map.cols)
            {
                obs_out_of_bounds++;
                continue;
            }

            // 读取网络预测的逆深度（这是网络的直接输出）
            double inv_depth_net = static_cast<double>(depth_map.at<float>(v, u));

            // 检查逆深度值有效性
            if (!std::isfinite(inv_depth_net) || inv_depth_net < 1e-6 || inv_depth_net > 100.0)
            {
                obs_invalid_depth_net++;
                continue;
            }

            // 找到有效的逆深度配对，收集数据
            // 拟合关系：inv_depth_vins = a * inv_depth_net + b
            depth_net_vec.push_back(inv_depth_net);  // 网络预测的逆深度
            depth_vins_vec.push_back(inv_depth_vins); // VINS的逆深度
            features_with_depth++;
            found_valid_depth = true;  // 标记已找到，跳出循环
        }
    }

    // 打印详细的诊断信息
    ROS_INFO("[Depth Init] Diagnostic info:");
    ROS_INFO("[Depth Init]   Features checked: %d", features_checked);
    ROS_INFO("[Depth Init]   Features triangulated (solve_flag==1): %d", features_triangulated);
    ROS_INFO("[Depth Init]   Features with valid depth_vins: %d", features_valid_depth_vins);
    ROS_INFO("[Depth Init]   Features with observations: %d", features_with_observations);
    ROS_INFO("[Depth Init]   Observation frames checked: %d", obs_frames_checked);
    ROS_INFO("[Depth Init]   Observation frames with depth map: %d", obs_frames_with_depth_map);
    ROS_INFO("[Depth Init]   Observations out of bounds: %d", obs_out_of_bounds);
    ROS_INFO("[Depth Init]   Observations with invalid depth_net: %d", obs_invalid_depth_net);
    ROS_INFO("[Depth Init]   Final valid pairing points: %d", features_with_depth);

    // 检查数据点数量是否足够
    const int min_points = 20;
    if (features_with_depth < min_points)
    {
        ROS_WARN("[Depth Init] Alignment failed: only %d valid points (need >= %d). Using config values.",
                 features_with_depth, min_points);
        return;
    }

    // 构建线性最小二乘系统并求解，使用增强的鲁棒性策略
    ROS_INFO("[Depth Init] Building linear system with %d valid points...", features_with_depth);

    // 1. 数据预处理：计算统计信息用于异常值检测
    double mean_inv_depth_vins = 0.0, mean_inv_depth_net = 0.0;
    for (size_t i = 0; i < depth_net_vec.size(); ++i) {
        mean_inv_depth_vins += depth_vins_vec[i];
        mean_inv_depth_net += depth_net_vec[i];
    }
    mean_inv_depth_vins /= depth_net_vec.size();
    mean_inv_depth_net /= depth_net_vec.size();

    double std_inv_depth_vins = 0.0, std_inv_depth_net = 0.0;
    for (size_t i = 0; i < depth_net_vec.size(); ++i) {
        double diff_vins = depth_vins_vec[i] - mean_inv_depth_vins;
        double diff_net = depth_net_vec[i] - mean_inv_depth_net;
        std_inv_depth_vins += diff_vins * diff_vins;
        std_inv_depth_net += diff_net * diff_net;
    }
    std_inv_depth_vins = std::sqrt(std_inv_depth_vins / depth_net_vec.size());
    std_inv_depth_net = std::sqrt(std_inv_depth_net / depth_net_vec.size());

    ROS_INFO("[Depth Init] Data statistics:");
    ROS_INFO("[Depth Init]   VINS inv_depth: mean=%.4f, std=%.4f", mean_inv_depth_vins, std_inv_depth_vins);
    ROS_INFO("[Depth Init]   NET inv_depth:  mean=%.4f, std=%.4f", mean_inv_depth_net, std_inv_depth_net);
    ROS_INFO("[Depth Init]   Scale ratio (VINS/NET): %.4f", mean_inv_depth_vins / (mean_inv_depth_net + 1e-10));

    // Print a few sample data points for debugging
    ROS_INFO("[Depth Init] Sample data points (first 5):");
    for (size_t i = 0; i < std::min(size_t(5), depth_net_vec.size()); ++i) {
        // Convert back to depth for readability
        double depth_vins_m = 1.0 / depth_vins_vec[i];
        double depth_net_equiv = 1.0 / depth_net_vec[i];
        ROS_INFO("[Depth Init]     [%zu] VINS: inv_d=%.4f (d=%.2fm), NET: inv_d=%.4f (d=%.2fm)",
                 i, depth_vins_vec[i], depth_vins_m, depth_net_vec[i], depth_net_equiv);
    }

    // 2. 异常值过滤（3-sigma rule）
    // NOTE: We do NOT modify the depth data here. The linear regression will find the correct
    // transformation parameters (a, b) that map network inverse depths to VINS inverse depths.
    // For traditional initialization: depth_vins is real metric depth from SFM
    // For fast initialization: depth_vins is from network with internal a,b parameters
    // The transformation inv_depth_vins = a * inv_depth_net + b handles both cases correctly.
    std::vector<double> filtered_depth_net_vec, filtered_depth_vins_vec;
    int outliers_removed = 0;
    for (size_t i = 0; i < depth_net_vec.size(); ++i) {
        double z_score_vins = std::abs(depth_vins_vec[i] - mean_inv_depth_vins) / (std_inv_depth_vins + 1e-6);
        double z_score_net = std::abs(depth_net_vec[i] - mean_inv_depth_net) / (std_inv_depth_net + 1e-6);

        // 移除3-sigma以外的异常值
        if (z_score_vins < 3.0 && z_score_net < 3.0) {
            filtered_depth_net_vec.push_back(depth_net_vec[i]);
            filtered_depth_vins_vec.push_back(depth_vins_vec[i]);
        } else {
            outliers_removed++;
        }
    }

    ROS_INFO("[Depth Init] Outlier filtering: removed %d points, kept %zu points",
             outliers_removed, filtered_depth_net_vec.size());

    // 检查过滤后数据是否足够
    if (filtered_depth_net_vec.size() < min_points) {
        ROS_WARN("[Depth Init] Too few points after outlier removal (%zu < %d). Using config values.",
                 filtered_depth_net_vec.size(), min_points);
        return;
    }

    // 3. 构建加权最小二乘系统（使用Huber权重降低异常值影响）
    // 先进行一次普通最小二乘估计，然后基于残差计算权重
    double sum_dn_dn = 0.0, sum_dn = 0.0, sum_dn_dv = 0.0, sum_dv = 0.0;
    double N = static_cast<double>(filtered_depth_net_vec.size());

    for (size_t i = 0; i < filtered_depth_net_vec.size(); ++i) {
        double dn = filtered_depth_net_vec[i];
        double dv = filtered_depth_vins_vec[i];
        sum_dn_dn += dn * dn;
        sum_dn += dn;
        sum_dn_dv += dn * dv;
        sum_dv += dv;
    }

    // 4. 检查条件数，确保数值稳定性
    Eigen::Matrix2d A_initial;
    A_initial(0, 0) = sum_dn_dn;
    A_initial(0, 1) = sum_dn;
    A_initial(1, 0) = sum_dn;
    A_initial(1, 1) = N;

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(A_initial);
    double condition_number = svd.singularValues()(0) / svd.singularValues()(1);
    ROS_INFO("[Depth Init] System condition number: %.2e", condition_number);

    if (condition_number > 1e6) {
        ROS_WARN("[Depth Init] Ill-conditioned system (cond=%.2e > 1e6). Adding regularization.", condition_number);

        // 添加Tikhonov正则化
        double lambda = 1e-6;
        A_initial(0, 0) += lambda;
        A_initial(1, 1) += lambda;
    }

    // 5. 使用SVD求解（比LDLT更稳定）
    Eigen::Vector2d b_initial;
    b_initial(0) = sum_dn_dv;
    b_initial(1) = sum_dv;

    Eigen::JacobiSVD<Eigen::Matrix2d> solver(A_initial, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector2d x_solution = solver.solve(b_initial);

    double estimated_a = x_solution(0);
    double estimated_b = x_solution(1);

    ROS_INFO("[Depth Init] Raw estimates from linear regression: a=%.6f, b=%.6f", estimated_a, estimated_b);

    // ============ Statistical Quality Check: Pearson Correlation Coefficient ============
    // Calculate correlation to assess if the relationship is statistically meaningful
    // Formula: r = Σ(x_i - x̄)(y_i - ȳ) / √(Σ(x_i - x̄)² Σ(y_i - ȳ)²)

    double mean_x = 0.0, mean_y = 0.0;
    for (size_t i = 0; i < filtered_depth_net_vec.size(); ++i) {
        mean_x += filtered_depth_net_vec[i];
        mean_y += filtered_depth_vins_vec[i];
    }
    mean_x /= filtered_depth_net_vec.size();
    mean_y /= filtered_depth_vins_vec.size();

    double numerator = 0.0;
    double denom_x = 0.0;
    double denom_y = 0.0;

    for (size_t i = 0; i < filtered_depth_net_vec.size(); ++i) {
        double dx = filtered_depth_net_vec[i] - mean_x;
        double dy = filtered_depth_vins_vec[i] - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    double correlation = numerator / (std::sqrt(denom_x * denom_y) + 1e-10);
    ROS_INFO("[Depth Init] Pearson correlation coefficient: r = %.4f", correlation);

    // Correlation threshold: reject if correlation is too weak
    const double correlation_threshold = 0.6;
    if (std::abs(correlation) < correlation_threshold)
    {
        ROS_WARN("[Depth Init]   Alignment REJECTED: Correlation too low (r=%.4f < %.2f).",
                 correlation, correlation_threshold);
        ROS_WARN("[Depth Init]   This indicates VINS initialization may be noisy or scale ambiguous.");
        ROS_WARN("[Depth Init]   Fallback to default config values: a=%.6f, b=%.6f",
                 DEPTH_SCALE_A, DEPTH_SHIFT_B);
        return;
    }

    // ============ Physical Validity Check: Correlation-Aware Parameter Bounds ============
    // Strategy: Trust high-correlation fits more than hardcoded ranges
    // - High correlation (r > 0.90): Allow wider bounds (scene-dependent scale)
    // - Low correlation (r <= 0.90): Use strict bounds (filter noise)
    //
    // Rationale: Depth Anything V2 can output values close to metric scale (a ≈ 1.0)
    // depending on scene content. High correlation indicates reliable fit.

    const double a_min = 0.01;   // Lower bound: always enforced
    double a_max = 0.35;         // Upper bound: will be relaxed for high correlation
    const double b_min = -1.0;   // Shift bounds
    const double b_max = 1.0;

    // Correlation-aware upper bound relaxation
    const double high_correlation_threshold = 0.90;
    if (correlation > high_correlation_threshold)
    {
        // High correlation: Trust the fit, allow up to a=5.0
        a_max = 5.0;
        ROS_INFO("[Depth Init] High correlation detected (r=%.4f > %.2f). Relaxing upper bound to a_max=%.2f",
                 correlation, high_correlation_threshold, a_max);
    }
    else
    {
        // Low/medium correlation: Use conservative bounds
        ROS_INFO("[Depth Init] Standard correlation (r=%.4f <= %.2f). Using strict bound a_max=%.2f",
                 correlation, high_correlation_threshold, a_max);
    }

    if (estimated_a < a_min || estimated_a > a_max ||
        estimated_b < b_min || estimated_b > b_max ||
        !std::isfinite(estimated_a) || !std::isfinite(estimated_b))
    {
        ROS_WARN("[Depth Init] ✗ Alignment REJECTED: Parameter out of safe range.");
        ROS_WARN("[Depth Init]   Calculated: a=%.6f (valid: [%.2f, %.2f]), b=%.6f (valid: [%.2f, %.2f])",
                 estimated_a, a_min, a_max, estimated_b, b_min, b_max);
        ROS_WARN("[Depth Init]   Correlation: r=%.4f (high_threshold: %.2f)",
                 correlation, high_correlation_threshold);
        ROS_WARN("[Depth Init]   Fallback to default config values: a=%.6f, b=%.6f",
                 DEPTH_SCALE_A, DEPTH_SHIFT_B);
        return;
    }

    // ============ Physical Sanity Check: Mean Depth Ratio ============
    // Check if the implied mean depth makes physical sense
    // Mean depth (VINS) = 1 / mean_inv_depth_vins
    // Mean depth (NET after transform) = 1 / (a * mean_inv_depth_net + b)

    double mean_depth_vins = 1.0 / (mean_inv_depth_vins + 1e-10);
    double mean_depth_net_transformed = 1.0 / (estimated_a * mean_inv_depth_net + estimated_b + 1e-10);
    double depth_ratio = mean_depth_vins / (mean_depth_net_transformed + 1e-10);

    ROS_INFO("[Depth Init] Mean depth check: VINS=%.2fm, NET(transformed)=%.2fm, ratio=%.3f",
             mean_depth_vins, mean_depth_net_transformed, depth_ratio);

    // Expect ratio close to 1.0; large deviations suggest bad fit
    const double depth_ratio_min = 0.5;
    const double depth_ratio_max = 2.0;

    if (depth_ratio < depth_ratio_min || depth_ratio > depth_ratio_max)
    {
        ROS_WARN("[Depth Init] ✗ Alignment REJECTED: Mean depth ratio unreasonable (%.3f not in [%.1f, %.1f]).",
                 depth_ratio, depth_ratio_min, depth_ratio_max);
        ROS_WARN("[Depth Init]   Fallback to default config values: a=%.6f, b=%.6f",
                 DEPTH_SCALE_A, DEPTH_SHIFT_B);
        return;
    }

    // 计算残差（用于评估拟合质量）
    double residual_sum = 0.0;
    for (size_t i = 0; i < filtered_depth_net_vec.size(); ++i)
    {
        double predicted = estimated_a * filtered_depth_net_vec[i] + estimated_b;
        double error = filtered_depth_vins_vec[i] - predicted;
        residual_sum += error * error;
    }
    double rmse = std::sqrt(residual_sum / filtered_depth_net_vec.size());

    // ============================================================================
    // Physics-Aware Online Initialization with Quality Gates
    // ============================================================================
    // Strategy: "Safety Net" - Accept calculated values ONLY if quality is high
    // - IF correlation > 0.6 AND 0.05 < a < 0.35: Accept (good for V2_03)
    // - ELSE: Fallback to Config Default (safe for MH_05)
    // - Warm-up mechanism will refine either case
    // ============================================================================

    // All quality checks passed - accept calculated values
    ROS_WARN("[Depth Init] ═══════════════════════════════════════════════════");
    ROS_WARN("[Depth Init] ✓ Online Alignment ACCEPTED - All quality gates passed:");
    ROS_WARN("[Depth Init]   Data quality:");
    ROS_WARN("[Depth Init]     - Points used: %zu / %d checked", filtered_depth_net_vec.size(), features_checked);
    ROS_WARN("[Depth Init]     - Correlation: r = %.4f (threshold: %.2f) ✓", correlation, correlation_threshold);

    // Explain acceptance reasoning
    if (correlation > high_correlation_threshold && estimated_a > 0.35)
    {
        ROS_WARN("[Depth Init]     - SPECIAL: High correlation (r=%.4f > %.2f) enabled relaxed bounds",
                 correlation, high_correlation_threshold);
        ROS_WARN("[Depth Init]     - Accepted a=%.6f despite exceeding standard limit (0.35)", estimated_a);
        ROS_WARN("[Depth Init]     - Reason: Strong linear relationship indicates reliable scale estimation");
    }

    ROS_WARN("[Depth Init]   Parameter estimates:");
    ROS_WARN("[Depth Init]     - a = %.6f (range: [%.2f, %.2f]) ✓", estimated_a, a_min, a_max);
    ROS_WARN("[Depth Init]     - b = %.6f (range: [%.2f, %.2f]) ✓", estimated_b, b_min, b_max);
    ROS_WARN("[Depth Init]   Fitting quality:");
    ROS_WARN("[Depth Init]     - RMSE: %.4f", rmse);
    ROS_WARN("[Depth Init]     - Condition number: %.2e", condition_number);
    ROS_WARN("[Depth Init]     - Mean depth ratio: %.3f (range: [%.1f, %.1f]) ✓",
             depth_ratio, depth_ratio_min, depth_ratio_max);
    ROS_WARN("[Depth Init] ───────────────────────────────────────────────────");
    ROS_WARN("[Depth Init] Applying calculated parameters to system:");
    ROS_WARN("[Depth Init]   Old: a=%.6f, b=%.6f (config defaults)", DEPTH_SCALE_A, DEPTH_SHIFT_B);
    ROS_WARN("[Depth Init]   New: a=%.6f, b=%.6f (online estimation)", estimated_a, estimated_b);
    ROS_WARN("[Depth Init] ═══════════════════════════════════════════════════");

    // Update system parameters
    DEPTH_SCALE_A = estimated_a;
    DEPTH_SHIFT_B = estimated_b;

    // Store for random walk constraint
    last_depth_a = estimated_a;
    last_depth_b = estimated_b;
    has_last_depth_params = true;

    // Update parameter block (para_DepthScaleShift is double[1][2])
    para_DepthScaleShift[0][0] = DEPTH_SCALE_A;
    para_DepthScaleShift[0][1] = DEPTH_SHIFT_B;
}

/**
 * @brief 处理收到的IMU数据
 * 
 * @param dt 时间间隔
 * @param linear_acceleration IMU测得的线加速度
 * @param angular_velocity IMU测得的角速度
 * 
 * 该函数负责IMU预积分。它将IMU数据累积到预积分对象中，并用于预测下一时刻的位姿。
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 如果是第一帧IMU数据，记录初始值
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 如果当前帧的预积分对象不存在，则创建一个新的
    if (!pre_integrations[frame_count])
    {
        // 使用上一帧的偏置作为初始偏置来创建预积分对象
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; 
    }
    
    // 从第二帧开始，将IMU数据加入预积分
    if (frame_count != 0)
    {
        // 将IMU数据推入当前帧的预积分器
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // 同时推入一个临时的预积分器，用于视觉帧之间
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 缓存IMU数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // --- 简单的状态传播，用于提供一个粗略的位姿估计 ---
        int j = frame_count;
        // 计算去除重力和偏置后的加速度（在世界坐标系下）
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        // 计算去除偏置后的角速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 使用角速度更新姿态
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // 计算当前时刻去除重力和偏置后的加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 使用梯形积分计算平均加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 更新位置和速度
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    
    // 更新上一时刻的IMU测量值
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief 处理新的图像帧和其中的特征点
 * 
 * @param image 图像数据，包含特征点ID及其在图像中的位置 image: feature_id -> (camera_id, [x,y,z,u,v,velocity_x,velocity_y])
 * @param header ROS消息头，包含时间戳等信息
 * 
 * 这是VIO系统的核心驱动函数之一。它负责决定当前帧是否为关键帧，
 * 触发VIO初始化、后端优化和滑动窗口操作。
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, const cv::Mat &raw_image)
{
    // image数据结构: feature_id -> (camera_id, [x,y,z,u,v,velocity_x,velocity_y])
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 递增全局帧计数器（每次处理新图像时递增，持续增长）
    global_frame_count++;

    // 1. 检查特征点视差，决定当前帧是否为关键帧
    // addFeatureCheckParallax会检查当前帧与之前关键帧的平均视差
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD; // 视差足够大，是关键帧，后续需要边缘化最老的帧
    else
        marginalization_flag = MARGIN_SECOND_NEW; // 视差不足，是非关键帧，后续需要边缘化次新帧（即丢弃当前帧）

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    
    // 存储当前帧的ROS消息头
    Headers[frame_count] = header;

    // 创建图像帧对象，并与时间戳关联, 将两帧之间的IMU预积分数据关联到当前图像帧
    ImageFrame imageframe(image, header.stamp.toSec(), tmp_pre_integration, raw_image);

    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    
    // 为下一帧创建新的临时预积分对象
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; 

    // 2. 如果需要在线标定外参
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 获取前后两帧之间的匹配特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 尝试使用这些匹配点和IMU预积分的旋转来标定外参旋转 ric
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; // 标定成功后，切换到优化模式
            }
        }
    }

    // 3. 根据求解器状态执行不同操作
    if (solver_flag == INITIAL) // 如果系统处于初始化阶段
    {
        if (frame_count == WINDOW_SIZE) // 如果滑动窗口已满（收集了足够的数据）
        {
            bool is_init_success = false; // 标记初始化是否成功

            if (USE_FAST_INIT) // 如果启用了快速初始化
            {
                // --- 分支 1: 执行新的快速初始化流程 ---
                ROS_INFO_THROTTLE(1.0, "Attempting Fast Monocular Initialization...");

                // 步骤 1: 尝试计算窗口第一帧的深度图
                if (!tryComputeFirstFrameDepth()) {
                    ROS_WARN_THROTTLE(1.0, "Fast-Init: Waiting for depth map computation...");
                    is_init_success = false;
                }
                else {
                    // 步骤 2: 检查初始化条件并执行初始化
                    TicToc t_fast_init;
                    double rot_sum = 0.0;
                    
                    if (!checkFastInitConditions(rot_sum)) {
                        // 初始化条件不满足，跳过本次初始化
                        is_init_success = false;
                    }
                    else {
                        // 步骤 3: 执行快速初始化
                        std::map<int, Eigen::Vector3d> Ps_init;
                        std::map<int, Eigen::Vector3d> Vs_init;
                        std::map<int, Eigen::Quaterniond> Rs_init;
                        
                        if (performFastInitialization(Ps_init, Vs_init, Rs_init)) {
                            // 步骤 4: 更新估计器状态
                            updateEstimatorStateFromFastInit(Ps_init, Vs_init, Rs_init);
                            ROS_INFO("Fast Monocular Init Succeeded! (%.2f ms, rotation: %.3f rad)", 
                                t_fast_init.toc(), rot_sum);
                            is_init_success = true;
                        }
                        else {
                            ROS_WARN("Fast Monocular Init Failed.");
                            // 初始化失败，重置深度图状态
                            std::lock_guard<std::mutex> lock(m_depth_mutex);
                            m_first_frame_depth_computed = false;
                            m_depth_window_start_id = -1;
                            is_init_success = false;
                        }
                    }
                }
            }
            else
            {
                // --- 分支 2: 执行 VINS-Mono 原版初始化流程 ---
                if(ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
                {
                    // 步骤 1: 纯视觉SFM（Structure from Motion）
                    TicToc t_init; // 记录初始化开始时间
                    if (initialStructure())
                    {
                        // 步骤 2: 视觉-IMU对齐
                        if (visualInitialAlign())
                        {
                            is_init_success = true; // 初始化成功
                            double t_init_cost = t_init.toc();
                            ROS_INFO("VINS-Mono Init Succeeded! Took %.2f ms.", t_init_cost);
                        }
                    }
                   initial_timestamp = header.stamp.toSec();
                }
            }
            
            // --- 统一处理初始化结果 ---
            if(is_init_success)
            {
                // *** 确保至少有一帧深度图可用于参数对齐 ***
                // 快速初始化路径：第一帧深度图已存在，直接返回true
                // 传统SFM初始化路径：计算一帧深度图（当前帧 WINDOW_SIZE）
                ensureDepthMapForAlignment();

                solver_flag = NON_LINEAR; // 切换到非线性优化模式

                // *** 在线估计深度尺度偏移参数 ***
                // 现在特征点已经三角化，可以进行参数对齐
                estimateDepthScaleShift();

                solveOdometry();          // 立即进行一次后端优化
                slideWindow();            // 滑动窗口
                f_manager.removeFailures(); // 移除失败的特征点
                ROS_INFO("Initialization finish!");
                // 记录当前位姿，用于失败检测
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
            {
                // 初始化失败，滑动窗口，丢弃最老的帧，继续尝试
                slideWindow();
            }
        }
        else
            frame_count++; // 如果窗口未满，继续累积帧
    }
    else // 如果系统已完成初始化，进入正常的非线性优化流程
    {
        // --- 深度传感器因子约束：计算滑动窗口帧的深度图 (Backend Depth Constraint) ---
        // 为当前滑动窗口中的所有帧计算深度图（如果尚未计算）
        if (ESTIMATE_DEPTH_SCALE_SHIFT && mp_depth_estimator && mp_depth_estimator->isReady())
        {
            TicToc t_depth_backend;
            int depth_computed_count = 0;

            // 遍历滑动窗口中的所有帧
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                double timestamp = Headers[i].stamp.toSec();
                auto frame_it = all_image_frame.find(timestamp);

                // 检查帧是否存在且有原始图像
                if (frame_it != all_image_frame.end() && !frame_it->second.raw_image.empty())
                {
                    auto& frame = frame_it->second;

                    // 如果该帧的深度图尚未计算，则计算之
                    if (!frame.depth_map_computed)
                    {
                        if (mp_depth_estimator->predict(frame.raw_image, frame.predicted_depth_map))
                        {
                            frame.depth_map_computed = true;
                            depth_computed_count++;
                            ROS_DEBUG("Backend: computed depth for frame %d (t=%.3f)", i, timestamp);
                        }
                        else
                        {
                            ROS_DEBUG("Backend: depth prediction failed for frame %d", i);
                        }
                    }
                }
            }

            if (depth_computed_count > 0)
            {
                // ROS_INFO("[Backend] Computed depth maps for %d frames (%.1f ms)",
                //          depth_computed_count, t_depth_backend.toc());
            }
        }

        TicToc t_solve;
        // 核心步骤：执行后端优化
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 检查系统是否出现故障
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();   // 重置系统状态
            setParameter(); // 重新设置参数
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        // 滑动窗口
        slideWindow();
        // 移除跟踪失败的特征点
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        
        // 准备VINS的输出
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        // 更新用于失败检测的上一帧位姿
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief 尝试计算窗口第一帧的深度图
 */
 bool Estimator::tryComputeFirstFrameDepth()
 {
     if (all_image_frame.empty()) {
         return false;
     }
 
     auto first_frame_it = all_image_frame.begin();
     double first_frame_stamp = first_frame_it->first;
     
     // 判断是否需要重新计算：未计算过，或窗口第一帧发生变化
     bool need_recompute = !m_first_frame_depth_computed;
     if (m_first_frame_depth_computed && m_depth_window_start_id >= 0) {
         double stored_stamp = Headers[m_depth_window_start_id].stamp.toSec();
         need_recompute = (std::abs(first_frame_stamp - stored_stamp) > 1e-6);
     }
     
     if (!need_recompute) {
         return true; // 深度图已存在且仍然有效
     }
     
     // 等待深度估计模型就绪
     if (!mp_depth_estimator->isReady()) {
         ROS_WARN_THROTTLE(1.0, "Fast-Init: Depth model still loading, waiting...");
         if (!mp_depth_estimator->waitForReady(5000)) {
             ROS_WARN_THROTTLE(1.0, "Fast-Init: Depth model not ready after timeout, skipping depth computation.");
             return false;
         }
     }
     
     // 检查原始图像是否可用
     if (first_frame_it->second.raw_image.empty()) {
         ROS_WARN_THROTTLE(1.0, "Fast-Init: Waiting for the raw image of the first frame...");
         return false;
     }
     
     // 计算深度图
     ROS_INFO("Fast-Init: Calculating depth for the first frame...");
     TicToc t_depth;
     cv::Mat depth_map;
     
     if (!mp_depth_estimator->predict(first_frame_it->second.raw_image, depth_map)) {
         return false;
     }
     
     // 存储深度图并查找对应的 frame_id
     std::lock_guard<std::mutex> lock(m_depth_mutex);
     m_first_frame_depth_map = depth_map;
     m_first_frame_depth_computed = true;

     // *** FIX: 同时更新 ImageFrame 的深度图数据 ***
     // 这样 ensureDepthMapForAlignment() 才能检测到深度图已存在，避免重复推理
     first_frame_it->second.predicted_depth_map = depth_map;
     first_frame_it->second.depth_map_computed = true;

     // 查找对应的 frame_id 并记录
     for (int i = 0; i <= WINDOW_SIZE; i++) {
         if (std::abs(Headers[i].stamp.toSec() - first_frame_stamp) < 1e-6) {
             m_depth_window_start_id = i;
             break;
         }
     }
     
     ROS_INFO("Fast-Init: Depth prediction succeeded (%.2f ms).", t_depth.toc());
     return true;
 }
 
 /**
  * @brief 检查快速初始化的条件是否满足
  */
  bool Estimator::checkFastInitConditions(double& rot_sum_out)
  {
      rot_sum_out = 0.0;
      
      // 1. 特征数门槛：可以比原版更低（原版需要80+，快速初始化可以降至50-60）
      //    因为深度图提供了额外的约束，不依赖大量特征点的三角化
      int feature_count = f_manager.getFeatureCount();
      if (feature_count < FAST_INIT_MIN_FEATURES) {
          ROS_WARN_THROTTLE(1.0, "Fast-Init gate: too few features (%d < %d).", 
                           feature_count, FAST_INIT_MIN_FEATURES);
          return false;
      }
      
      // 2. IMU激励检查：使用原版逻辑，但可以适当降低阈值
      //    因为有深度先验，对运动激励的要求可以稍微放宽
      double acc_var = 0.0;
      {
          map<double, ImageFrame>::iterator frame_it;
          Vector3d sum_g;
          int valid_count = 0;
          
          // 遍历所有帧（从第二帧开始），计算平均加速度
          for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
          {
              if (!frame_it->second.pre_integration) {
                  continue;
              }
              double dt = frame_it->second.pre_integration->sum_dt;
              if (dt < 1e-6) {
                  continue;
              }
              Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
              sum_g += tmp_g;
              valid_count++;
          }
          
          if (valid_count <= 0) {
              ROS_WARN_THROTTLE(1.0, "Fast-Init gate: insufficient frames for IMU excitation check.");
              return false;
          }
          
          Vector3d aver_g = sum_g / valid_count;
          
          // 计算加速度方差
          for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
          {
              if (!frame_it->second.pre_integration) {
                  continue;
              }
              double dt = frame_it->second.pre_integration->sum_dt;
              if (dt < 1e-6) {
                  continue;
              }
              Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
              acc_var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
          }
          acc_var = sqrt(acc_var / valid_count);
      }
      
      // 快速初始化的优势：可以接受更低的IMU激励（原版0.25，这里可以0.15-0.2）
      // 因为深度图提供了尺度约束，对IMU激励的依赖降低
      if (acc_var < FAST_INIT_MIN_ACC_VAR) {
          ROS_WARN_THROTTLE(1.0, "Fast-Init gate: insufficient IMU excitation (acc variance: %.4f < %.2f).", 
                           acc_var, FAST_INIT_MIN_ACC_VAR);
          return false;
      }
      
      // 3. 可选：计算旋转激励用于日志记录（用于对比分析，但不作为硬性要求）
      for (int k = 1; k <= WINDOW_SIZE; ++k) {
          if (pre_integrations[k]) {
              Eigen::AngleAxisd aa(pre_integrations[k]->delta_q);
              rot_sum_out += std::abs(aa.angle());
          }
      }
      
      // 记录初始化条件（用于性能分析和证明优势）
      ROS_INFO_THROTTLE(1.0, "Fast-Init conditions: features=%d, acc_var=%.4f, rot_sum=%.3f rad", 
                       feature_count, acc_var, rot_sum_out);
      
      return true;
  }
 
 /**
  * @brief 执行快速单目初始化
  */
 bool Estimator::performFastInitialization(std::map<int, Eigen::Vector3d>& Ps_init,
                                          std::map<int, Eigen::Vector3d>& Vs_init,
                                          std::map<int, Eigen::Quaterniond>& Rs_init)
 {
     std::lock_guard<std::mutex> lock(m_depth_mutex);
     return mp_fast_initializer->initialize(all_image_frame,
                                           m_first_frame_depth_map,
                                           g,
                                           Ps_init, Vs_init, Rs_init);
 }
 
 /**
  * @brief 将快速初始化的结果更新到估计器状态
  */
 void Estimator::updateEstimatorStateFromFastInit(const std::map<int, Eigen::Vector3d>& Ps_init,
                                                  const std::map<int, Eigen::Vector3d>& Vs_init,
                                                  const std::map<int, Eigen::Quaterniond>& Rs_init)
 {
    // 将初始化结果复制到 Estimator 的状态变量
    for (int k = 0; k <= WINDOW_SIZE; ++k) {
        if (Ps_init.count(k) && Vs_init.count(k) && Rs_init.count(k)) {
            Ps[k] = Ps_init.at(k);
            Vs[k] = Vs_init.at(k);
            Rs[k] = Rs_init.at(k).toRotationMatrix();
    
            double ts = Headers[k].stamp.toSec();
            auto it = all_image_frame.find(ts);
            if (it != all_image_frame.end()) {
                it->second.R = Rs[k];
                it->second.T = Ps[k];
                it->second.is_key_frame = true;
            } else {
                // 最近邻回退，容忍小的时间误差
                const double tol = 1e-3; // 秒，按你的时间戳精度可调
                auto it2 = all_image_frame.lower_bound(ts);
                auto choose = it2;
                if (it2 != all_image_frame.begin()) {
                    auto it_prev = std::prev(it2);
                    if (it2 == all_image_frame.end() ||
                        std::abs(it_prev->first - ts) < std::abs(it2->first - ts)) {
                        choose = it_prev;
                    }
                }
                if (choose != all_image_frame.end() && std::abs(choose->first - ts) < tol) {
                    choose->second.R = Rs[k];
                    choose->second.T = Ps[k];
                    choose->second.is_key_frame = true;
                } else {
                    ROS_WARN_STREAM("Fast-Init: cannot sync pose to all_image_frame at ts=" << std::fixed << ts);
                }
            }
        }
    }

    // 追加：仅估计陀螺仪零偏并在成功时重积分（重力已由 fast-init 求得）
    bool bias_updated = solveGyroscopeBiasNew(all_image_frame, Bgs);
    if (bias_updated)
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            if (pre_integrations[i])
                pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        ROS_INFO("Fast-Init post bias: gyro bias refined and pre-integrations repropagated.");
    }
    else
    {
        ROS_INFO("Fast-Init post bias: skipped (no valid bias update).");
    }

    // *** 新增：为特征点设置estimated_depth，使得后续在线参数估计可用 ***
    // 快速初始化内部计算了 a_fast, b_fast，将网络逆深度转为正深度
    // 这里我们需要访问快速初始化的参数结果，并据此给特征点赋予estimated_depth
    assignEstimatedDepthFromFastInit();
 }

/**
 * @brief 纯视觉的结构恢复（Structure from Motion, SFM）
 * @return true 如果SFM成功
 * 
 * 这个函数是VIO初始化的第一步。它尝试仅通过视觉信息恢复出滑动窗口内
 * 各个关键帧的相对位姿以及地图点的三维坐标。
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // 1. 检查IMU的可观性（激励是否充分）
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 遍历所有帧，计算平均加速度，以估计重力方向
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            // delta_v / dt 是这段时间内的平均加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        // 计算加速度的方差，如果方差太小，说明IMU运动不剧烈，可能导致重力方向不可观
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            // return false; // 在某些情况下，即使激励不足也可能继续尝试
        }
    }
    
    // 2. 全局SFM
    // 一共有frame_count + 1帧，窗口内是满的，再加上新来的一帧
    Quaterniond Q[frame_count + 1]; // 存储每帧的姿态（四元数）
    Vector3d T[frame_count + 1];    // 存储每帧的位置
    map<int, Vector3d> sfm_tracked_points; // 存储恢复出的3D地图点
    vector<SFMFeature> sfm_f; // 存储用于SFM的特征点数据结构
    
    // 将特征管理器中的数据转换为SFM所需格式
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false; // 初始状态为未三角化
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point; // 归一化相机坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    
    Matrix3d relative_R;
    Vector3d relative_T;
    int l; // 用于恢复相对位姿的参考帧索引
    
    // 1.1. 找到一个与最新帧有足够视差和共视点的历史帧 l
    // 1.2. 利用本质矩阵来求解两帧之间的相对位姿和相对平移
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    
    // 2. 使用 GlobalSFM 类来恢复所有帧的位姿和地图点
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD; // SFM失败，标记边缘化老帧，以便系统可以滑动窗口并重试
        return false;
    }

    // 3. 对所有非关键帧进行PnP求解
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 从相机到w的旋转 变成 imu到w的旋转 （w 设定的是第l帧的相机系）
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        
        // 如果是非关键帧，准备PnP求解
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];

        cv::eigen2cv(R_inital, rvec);
        cv::Rodrigues(rvec, rvec); // 转为旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector; // 3D点
        vector<cv::Point2f> pts_2_vector; // 对应的2D像素点
        // 收集该帧观测到的、并且已经被SFM恢复出3D坐标的地图点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
            it = sfm_tracked_points.find(feature_id);
            if(it != sfm_tracked_points.end())
            {
                Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
            }
        }
        
        // 使用单位矩阵作为相机内参，因为我们使用的是归一化坐标
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 执行PnP求解
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, cv::Mat(), rvec, t, true))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Mat r;
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    
    // 4. 进行视觉-IMU对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

    }
}

/**
 * @brief 视觉-IMU对齐
 * @return true 如果对齐成功
 * 
 * 这是VIO初始化的第二步。它利用SFM的结果和IMU预积分信息，
 * 来求解尺度、重力向量、速度以及陀螺仪偏置。
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x; // 状态向量，包含 [v0, v1, ..., g, s]
    
    // 1. 求解尺度、重力和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // 2. 更新滑动窗口内的所有状态
    // 将对齐后得到的位姿赋给滑动窗口中的状态变量
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 3. 更新特征点的深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1; // 暂时标记为-1，后续会重新三角化
    f_manager.clearDepth(dep);

    // 重新三角化所有特征点
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero(); // 三角化时暂时不考虑IMU和cam之间的平移
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    // 4. 恢复真实的尺度
    double s = (x.tail<1>())(0); // 从对齐结果中获取尺度因子
    
    // 用新的陀螺仪偏置重新进行IMU预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 将尺度 s 应用到所有位置和速度上，将原点设置为第一帧IMU
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // 从对齐结果中恢复速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 将尺度 s 应用到所有特征点的深度上
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 5. 将世界坐标系对齐到重力方向
    // 将z轴与重力方向对齐，消除yaw角的漂移
    Matrix3d R0 = Utility::g2R(g); // 计算一个从世界坐标系到重力对齐坐标系的旋转
    double yaw = Utility::R2ypr(R0 * Rs[0]).x(); // 计算第一帧在重力对齐坐标系下的yaw角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 构造一个只消除yaw角的旋转
    g = R0 * g; // 更新重力向量
    
    Matrix3d rot_diff = R0;
    // 将这个旋转应用到滑动窗口内所有的位姿和速度上
    // 最终的姿态参考坐标系为：以第一帧 IMU 为原点、Z 轴对齐重力、且第一帧 yaw 为 0 的规范世界坐标系
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/**
 * @brief 寻找用于SFM的相对位姿
 * @param[out] relative_R 相对旋转
 * @param[out] relative_T 相对平移
 * @param[out] l 找到的参考帧的索引
 * @return true 如果找到合适的帧
 * 
 * 该函数在滑动窗口中从前往后遍历，找到一个与最新帧(WINDOW_SIZE)
 * 有足够视差和共视点的帧 l，并计算它们之间的相对位姿。
 * 这个相对位姿将作为全局SFM的起点。
 * 滑窗内一共有WINDOW_SIZE + 1帧，从0到WINDOW_SIZE
 * 从0到WINDOW_SIZE-1帧，依次与最新帧(WINDOW_SIZE)进行匹配
 * 
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // 从滑窗内第一帧开始向后遍历
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 获取帧 i 和最新帧之间的匹配点
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20) // 如果匹配点足够多
        {
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            
            // 如果平均视差（乘以焦距后，近似为像素距离）足够大 && 能成功求解相对位姿（通过8点法等），则认为找到了合适的帧
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // 记录该帧的索引
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief 求解里程计，即后端优化
 * 
 * 在系统完成初始化后，每来一帧新的关键帧，此函数就会被调用。
 * 它负责三角化新的地图点，并构建和求解一个非线性优化问题。
 */
void Estimator::solveOdometry()
{
    // 必须在滑动窗口满了之后才能进行
    if (frame_count < WINDOW_SIZE)
        return;
    // 必须在非线性优化模式下
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        // 1. 三角化新的特征点
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        // 2. 执行后端优化
        optimization();
    }
}

/**
 * @brief 将状态变量从Eigen向量格式转换为Ceres求解器所需的double数组格式
 */
void Estimator::vector2double()
{
    // 转换位姿(Ps, Rs)和速度/偏置(Vs, Bas, Bgs)
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    // 转换外参(tic, ric)
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 转换特征点的逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 转换时间延迟td
    if (ESTIMATE_TD)
        para_Td[0][0] = td;

    // --- 深度传感器因子约束状态转换 (Backend Depth Constraint) ---
    // 注意：para_DepthScaleShift 在Ceres优化中是持久的，
    // 它会在clearState时初始化，在优化过程中被Ceres修改，
    // 在double2vector中被同步回全局变量。
    // 这里不需要每次都从全局变量重新赋值。
}

/**
 * @brief 将Ceres求解器优化后的double数组结果转换回Eigen向量格式
 */
void Estimator::double2vector()
{
    // 记录优化前的第一帧位姿，用于保持全局坐标系的一致性
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    // 如果发生过故障，则以上一次的位姿为基准
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    
    // 优化后的第一帧位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    // 计算yaw角的变化量
    double y_diff = origin_R0.x() - origin_R00.x();
    
    // 构造一个只包含yaw角变化的旋转矩阵，用于对齐整个轨迹，防止yaw角漂移
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    // 处理欧拉角奇异点的情况
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    // 将优化结果转换回状态变量，并应用yaw角对齐
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    // 转换外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    // 转换特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    // 转换时间延迟
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // --- 深度传感器因子约束状态转换 (Backend Depth Constraint) ---
    // 将Ceres优化后的深度仿射变换参数同步回全局变量
    if (ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        double old_a = DEPTH_SCALE_A;
        double old_b = DEPTH_SHIFT_B;

        DEPTH_SCALE_A = para_DepthScaleShift[0][0];  // 尺度因子 a
        DEPTH_SHIFT_B = para_DepthScaleShift[0][1];  // 偏移因子 b

        // 计算参数变化量
        double delta_a = DEPTH_SCALE_A - old_a;
        double delta_b = DEPTH_SHIFT_B - old_b;

        // 更新随机游走模型的"上一次"值
        last_depth_a = DEPTH_SCALE_A;
        last_depth_b = DEPTH_SHIFT_B;
        has_last_depth_params = true;

        // 在第一次非线性优化后重置标志，之后使用正常的随机游走约束
        if (solver_flag == NON_LINEAR && is_first_depth_optimization)
        {
            is_first_depth_optimization = false;
            ROS_INFO("[Depth Opt] First optimization completed. Future optimizations will use normal random walk constraint.");
        }

        // 定期打印深度参数的变化轨迹（用于验证随机游走效果）
        static int frame_counter = 0;
        frame_counter++;

        // 降低打印频率，每 10 帧打印一次
        if (frame_counter % 10 == 0)
        {
            ROS_INFO("[DepthParams] Frame %d: a=%.6f (delta %.6f), b=%.6f (delta %.6f)",
                     frame_counter, DEPTH_SCALE_A, delta_a, DEPTH_SHIFT_B, delta_b);
        }
    }

    // 如果有重定位信息，计算漂移修正
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        // 得到重定位帧在当前优化后的位姿
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        
        // 计算当前估计的位姿与重定位提供的真值位姿之间的漂移
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;  
        ROS_WARN("Drift updated: |t|=%.4f m, yaw=%.3f deg",
            drift_correct_t.norm(),
            Utility::R2ypr(drift_correct_r).x()); 
        
        // 计算重定位帧与当前帧的相对位姿
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        
        relocalization_info = 0; // 处理完毕，重置标志
    }
}

/**
 * @brief VIO系统故障检测
 * @return true 如果检测到故障
 * 
 * 通过一些启发式规则来判断系统是否运行异常。
 */
bool Estimator::failureDetection()
{
    // 1. 跟踪的特征点太少
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    // 2. IMU偏置估计值过大
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}


/**
 * @brief 后端非线性优化
 * 
 * 构建一个包含IMU约束、视觉重投影约束和先验约束的图优化问题，
 * 并使用Ceres Solver进行求解。
 */
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // 使用柯西核函数来减小外点的影响
    loss_function = new ceres::CauchyLoss(1.0);
    
    // 1. 添加参数块
    // 添加滑动窗口中所有帧的位姿(Pose)和速度/偏置(SpeedBias)作为待优化变量
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 位姿参数块，使用自定义的 PoseLocalParameterization 进行李群流形上的更新
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 添加相机到IMU的外参作为待优化变量
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        // 如果不在线估计外参，则将其固定
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 如果估计时间延迟，则添加为待优化变量
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
    }

    TicToc t_whole, t_prepare;
    // 将状态变量转换为Ceres所需的数组格式
    vector2double();

    // 2. 添加残差块（约束）
    // a. 添加边缘化的先验信息
    if (last_marginalization_info)
    {
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // b. 添加IMU预积分约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    
    // c. 添加视觉重投影约束
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
    
        ++feature_index;
        
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        
        // 方案1：第一帧作为参考，与其他所有帧建立约束（保证全局一致性）
        for (int idx = 1; idx < it_per_id.feature_per_frame.size(); idx++)
        {
            int imu_j = it_per_id.start_frame + idx;
            Vector3d pts_j = it_per_id.feature_per_frame[idx].point;
            
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(
                    pts_i, pts_j, 
                    it_per_id.feature_per_frame[0].velocity, 
                    it_per_id.feature_per_frame[idx].velocity,
                    it_per_id.feature_per_frame[0].cur_td, 
                    it_per_id.feature_per_frame[idx].cur_td,
                    it_per_id.feature_per_frame[0].uv.y(), 
                    it_per_id.feature_per_frame[idx].uv.y());
                problem.AddResidualBlock(f_td, loss_function, 
                    para_Pose[imu_i], para_Pose[imu_j], 
                    para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, 
                    para_Pose[imu_i], para_Pose[imu_j], 
                    para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
        
        // 方案2：相邻帧对之间建立约束（提高局部精度）
        for (int idx = 0; idx < it_per_id.feature_per_frame.size() - 1; idx++)
        {
            int imu_i_adj = it_per_id.start_frame + idx;
            int imu_j_adj = it_per_id.start_frame + idx + 1;
            Vector3d pts_i_adj = it_per_id.feature_per_frame[idx].point;
            Vector3d pts_j_adj = it_per_id.feature_per_frame[idx + 1].point;
            
            // 跳过第一帧参考的约束（避免重复）
            if (idx == 0) continue;  // 因为 (0,1) 已经在方案1中建立了
            
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(
                    pts_i_adj, pts_j_adj, 
                    it_per_id.feature_per_frame[idx].velocity, 
                    it_per_id.feature_per_frame[idx + 1].velocity,
                    it_per_id.feature_per_frame[idx].cur_td, 
                    it_per_id.feature_per_frame[idx + 1].cur_td,
                    it_per_id.feature_per_frame[idx].uv.y(), 
                    it_per_id.feature_per_frame[idx + 1].uv.y());
                problem.AddResidualBlock(f_td, loss_function, 
                    para_Pose[imu_i_adj], para_Pose[imu_j_adj], 
                    para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i_adj, pts_j_adj);
                problem.AddResidualBlock(f, loss_function, 
                    para_Pose[imu_i_adj], para_Pose[imu_j_adj], 
                    para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_INFO_THROTTLE(20.0, "[Backend] Visual factors: %d", f_m_cnt);

    // --- 深度传感器因子约束：添加深度约束残差块 (Backend Depth Constraint) ---
    if (ESTIMATE_DEPTH_SCALE_SHIFT)
    {
        // 首先添加 para_DepthScaleShift 参数块
        problem.AddParameterBlock(para_DepthScaleShift[0], 2);

        TicToc t_depth_factor;
        int depth_factor_cnt = 0;
        int features_checked = 0;
        int frames_with_depth = 0;

        // ========================================================================
        // Depth Fusion Weight and Huber Threshold Calculation
        // ========================================================================
        // 支持两种模式：
        // - Mode 0 (Fixed): 使用固定权重和Huber阈值
        // - Mode 1 (Adaptive): 基于IMU运动状态动态调整权重和阈值
        // ========================================================================

        double final_weight;
        double final_huber_threshold;

        // 声明日志需要的变量（在两种模式外部声明，以便日志函数可以访问）
        double current_gyro_norm = 0.0;
        double current_acc_disturbance = 0.0;
        double raw_instability_score = 0.0;
        double smoothed_instability_score = 0.0;

        if (DEPTH_WEIGHT_MODE == 0)
        {
            // ========================================================================
            // Fixed Weight Mode (weight_mode = 0)
            // ========================================================================
            final_weight = DEPTH_FACTOR_WEIGHT;
            final_huber_threshold = DEPTH_FACTOR_HUBER_THRESHOLD;

            ROS_INFO_THROTTLE(10.0, "[Fixed Weight] Frame %d: W=%.2f, Huber=%.3f | a=%.4f",
                             global_frame_count, final_weight, final_huber_threshold, DEPTH_SCALE_A);
        }
        else
        {
            // ========================================================================
            // Physics-Aware Adaptive Depth Fusion (Multi-Factor Version)
            // ========================================================================
            // 理论基础：单目深度预测质量与运动模糊成反比
            // - 运动模糊由相机快速旋转(角速度)和平移振动(线加速度)共同引起
            // - IMU陀螺仪 + 加速度计提供更全面的运动感知
            // - 物理不变量：Depth Anything V2 特征误差下限 ≈ 0.25 (1/m)
            //
            // 多因子自适应策略：
            // - 综合不稳定性评分 = gyro_norm + acc_disturbance_weight × acc_disturbance
            // - 稳定 (score < 0.3):  高权重 (W=3.0), 严格阈值 (δ=0.75)
            // - 不稳定 (score > 1.5):  低权重 (W=1.0), 宽松阈值 (δ=0.25)
            // - 中间状态: 线性插值
            // ========================================================================

            // Step A1: 计算角速度强度 (从IMU陀螺仪数据)
            int valid_gyro_count = 0;

        // 遍历滑动窗口中的所有预积分，计算平均陀螺仪幅值
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            if (pre_integrations[i] != nullptr && !pre_integrations[i]->gyr_buf.empty())
            {
                // 计算该预积分中所有陀螺仪测量的平均范数
                for (const auto& gyr : pre_integrations[i]->gyr_buf)
                {
                    current_gyro_norm += gyr.norm();
                    valid_gyro_count++;
                }
            }
        }

        // 计算平均值
        if (valid_gyro_count > 0)
        {
            current_gyro_norm /= valid_gyro_count;
        }
        else
        {
            // 如果没有有效陀螺仪数据，假设为静止状态
            current_gyro_norm = 0.0;
        }

            // Step A2: 计算加速度扰动 (从IMU加速度计数据)
            int valid_acc_count = 0;
        const double GRAVITY_NOMINAL = 9.81; // 标称重力加速度 (m/s^2)

        // 遍历滑动窗口中的所有预积分，计算平均加速度扰动
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            if (pre_integrations[i] != nullptr && !pre_integrations[i]->acc_buf.empty())
            {
                // 计算该预积分中所有加速度测量与重力的偏差
                for (const auto& acc : pre_integrations[i]->acc_buf)
                {
                    // 加速度扰动 = |测量幅值 - 重力幅值|
                    // 这捕捉到平移加速度和振动，即使旋转很小时也能检测到
                    double acc_norm = acc.norm();
                    current_acc_disturbance += std::abs(acc_norm - GRAVITY_NOMINAL);
                    valid_acc_count++;
                }
            }
        }

        // 计算平均值
        if (valid_acc_count > 0)
        {
            current_acc_disturbance /= valid_acc_count;
        }
        else
        {
            // 如果没有有效加速度数据，假设无扰动
            current_acc_disturbance = 0.0;
        }

            // ========================================================================
            // Step B: Low-Pass Filter for Motion Score (Optimization B)
            // ========================================================================
            // 计算原始运动评分 (使用调整后的加速度权重 0.5)
            raw_instability_score = current_gyro_norm + 0.5 * current_acc_disturbance;

        // 应用指数移动平均（EMA）滤波器，消除加速度计高频噪声
        // α = 0.2: 新测量权重20%，历史平滑值权重80%
        // 时间常数 τ ≈ 5帧 @ 20Hz (约0.25秒响应时间)
        if (!is_score_initialized)
        {
            // 首帧：直接使用原始值初始化
            smoothed_instability_score = raw_instability_score;
            is_score_initialized = true;
            ROS_INFO("[Signal Filter] Initialized smoothed_instability_score = %.3f", smoothed_instability_score);
        }
        else
        {
            // 后续帧：EMA滤波
            const double ALPHA = 0.2;  // 平滑因子 (0 = 完全平滑, 1 = 无滤波)
            smoothed_instability_score = (1.0 - ALPHA) * smoothed_instability_score + ALPHA * raw_instability_score;
        }

        // Step C: 计算自适应权重 W (基于平滑后的评分线性插值)
        // *** TUNED: Relaxed thresholds to allow higher weights during normal flight ***
        // Old: THRESHOLD_LOW = 0.3, THRESHOLD_HIGH = 1.5
        // New: THRESHOLD_LOW = 0.8, THRESHOLD_HIGH = 2.5
        // Rationale: V2_03 has score ~1.2 (normal drone vibration), should get W >= 2.5
        const double RELAXED_THRESHOLD_LOW = 0.8;   // Increased from 0.3
        const double RELAXED_THRESHOLD_HIGH = 2.5;  // Increased from 1.5

        double adaptive_weight;
        if (smoothed_instability_score < RELAXED_THRESHOLD_LOW)
        {
            // 稳定状态：最大权重
            adaptive_weight = DEPTH_WEIGHT_STATIC;
        }
        else if (smoothed_instability_score > RELAXED_THRESHOLD_HIGH)
        {
            // 不稳定状态：最小权重
            adaptive_weight = DEPTH_WEIGHT_DYNAMIC;
        }
        else
        {
            // 中间状态：线性插值
            double instability_ratio = (smoothed_instability_score - RELAXED_THRESHOLD_LOW) /
                                      (RELAXED_THRESHOLD_HIGH - RELAXED_THRESHOLD_LOW);
            adaptive_weight = DEPTH_WEIGHT_STATIC - instability_ratio * (DEPTH_WEIGHT_STATIC - DEPTH_WEIGHT_DYNAMIC);
        }

        // Step D: 计算自适应Huber阈值 (用于稳态，作为线性衰减的目标值)
        // 物理不变量：无论权重如何变化，保持相同的物理误差下限
        double steady_state_huber_threshold = adaptive_weight * PHYSICAL_ERROR_THRESHOLD;

            // ========================================================================
            // Step E: Linear Decay for Huber Threshold (Optimization A - 3-Phase Ramp)
            // ========================================================================
            // 递增深度融合帧计数器（每次执行深度融合时递增）
            depth_fusion_frame_count++;

            ceres::LossFunction *depth_loss_function = nullptr;
            double current_huber_threshold;

            // Phase 1: Aggressive Convergence (Frame 1-30)
            // 使用极大阈值（近似L2损失），允许大梯度快速修正初值偏差
            if (depth_fusion_frame_count <= 30)
            {
                current_huber_threshold = 999.0;  // 实质上是L2损失（所有残差 < 999都是二次惩罚）
                depth_loss_function = new ceres::HuberLoss(current_huber_threshold);

                ROS_INFO_THROTTLE(5.0, "[3-Phase] Phase 1 (Aggressive): Frame %d/30, "
                                 "Huber=%.1f (≈L2), W=%.2f, smoothed_score=%.3f (raw=%.3f)",
                                 depth_fusion_frame_count, current_huber_threshold,
                                 adaptive_weight, smoothed_instability_score, raw_instability_score);
            }
            // Phase 2: Smooth Exit (Frame 31-100)
            // 线性衰减从5.0降至1.0，逐步"关闭大门"，避免梯度突变
            else if (depth_fusion_frame_count <= 100)
            {
                // 计算衰减进度 [0.0, 1.0]
                double decay_progress = static_cast<double>(depth_fusion_frame_count - 30) / 70.0;

                // 线性插值: 5.0 → steady_state_huber_threshold
                current_huber_threshold = 5.0 * (1.0 - decay_progress) + steady_state_huber_threshold * decay_progress;
                depth_loss_function = new ceres::HuberLoss(current_huber_threshold);

                // 首次进入Phase 2时打印完整说明
                static bool phase2_entry_printed = false;
                if (!phase2_entry_printed)
                {
                    ROS_WARN("[3-Phase] Entering Phase 2 (Smooth Exit) at Frame %d", depth_fusion_frame_count);
                    ROS_WARN("[3-Phase] Linear decay: Huber threshold 5.0 → %.3f over 70 frames",
                             steady_state_huber_threshold);
                    phase2_entry_printed = true;
                }

                ROS_INFO_THROTTLE(5.0, "[3-Phase] Phase 2 (Decay): Frame %d/100, "
                                 "Huber=%.3f (progress=%.1f%%), W=%.2f, smoothed_score=%.3f",
                                 depth_fusion_frame_count, current_huber_threshold,
                                 decay_progress * 100.0, adaptive_weight, smoothed_instability_score);
            }
            // Phase 3: Steady State (Frame 101+)
            // 使用自适应Huber阈值，基于运动评分动态调整
            else
            {
                current_huber_threshold = steady_state_huber_threshold;
                depth_loss_function = new ceres::HuberLoss(current_huber_threshold);

                // 首次进入Phase 3时打印完整说明
                static bool phase3_entry_printed = false;
                if (!phase3_entry_printed)
                {
                    ROS_WARN("[3-Phase] Entering Phase 3 (Steady State) at Frame %d", depth_fusion_frame_count);
                    ROS_WARN("[3-Phase] Now using adaptive Huber threshold (physics-aware)");
                    phase3_entry_printed = true;
                }

                ROS_INFO_THROTTLE(10.0, "[3-Phase] Phase 3 (Steady): Frame %d, "
                                 "Huber=%.3f, W=%.2f, smoothed_score=%.3f",
                                 depth_fusion_frame_count, current_huber_threshold,
                                 adaptive_weight, smoothed_instability_score);
            }

            // 包装自适应权重缩放
            // Note: Both warm-up Huber and normal Huber need weight scaling
            if (depth_loss_function != nullptr)
            {
                // Wrap the Huber loss with adaptive weight
                depth_loss_function = new ceres::ScaledLoss(depth_loss_function, adaptive_weight, ceres::TAKE_OWNERSHIP);
            }
            else
            {
                // This branch should never be reached now (warm-up uses Huber, not L2)
                // Keep for safety: L2 loss with weight scaling
                depth_loss_function = new ceres::ScaledLoss(nullptr, adaptive_weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            }

            // 每10帧打印一次自适应参数（更新为使用平滑后的评分）
            static int last_log_frame = -10;
            if (global_frame_count - last_log_frame >= 10 || global_frame_count < 5)
            {
                ROS_INFO("[Filtered Motion] Frame %d: Gyro=%.3f, AccDist=%.3f, "
                         "Raw_Score=%.3f, Smoothed_Score=%.3f, W=%.2f, Huber=%.3f | a=%.4f",
                         global_frame_count,
                         current_gyro_norm,
                         current_acc_disturbance,
                         raw_instability_score,
                         smoothed_instability_score,
                         adaptive_weight,
                         current_huber_threshold,
                         DEPTH_SCALE_A);
                last_log_frame = global_frame_count;
            }

            // 将自适应模式计算的结果赋值给最终变量
            final_weight = adaptive_weight;
            final_huber_threshold = current_huber_threshold;
        } // end of adaptive mode

        // ========================================================================
        // Create Loss Function for Fixed Mode
        // ========================================================================
        ceres::LossFunction *depth_loss_function = nullptr;
        if (DEPTH_WEIGHT_MODE == 0)
        {
            // Fixed mode: simple Huber loss with fixed threshold
            depth_loss_function = new ceres::HuberLoss(final_huber_threshold);
            depth_loss_function = new ceres::ScaledLoss(depth_loss_function, final_weight, ceres::TAKE_OWNERSHIP);
        }
        // Note: adaptive mode already created depth_loss_function in the else block above

        // ========================================================================
        // Log Depth Fusion Metrics (every 10 frames)
        // ========================================================================
        // 对于固定模式，IMU数据设为0（因为不使用）
        double log_gyro = (DEPTH_WEIGHT_MODE == 1) ? current_gyro_norm : 0.0;
        double log_acc = (DEPTH_WEIGHT_MODE == 1) ? current_acc_disturbance : 0.0;
        double log_raw_score = (DEPTH_WEIGHT_MODE == 1) ? raw_instability_score : 0.0;
        double log_smoothed_score = (DEPTH_WEIGHT_MODE == 1) ? smoothed_instability_score : 0.0;

        logDepthFusionMetrics(global_frame_count, log_gyro, log_acc,
                             log_raw_score, log_smoothed_score,
                             final_weight, final_huber_threshold,
                             DEPTH_SCALE_A, DEPTH_SHIFT_B);

        // 重新遍历所有特征点，为有深度图的观测添加深度约束
        feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            // 使用与视觉约束相同的筛选条件
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;
            features_checked++;

            int first_obs_frame_id = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 特征在首次观测帧的归一化坐标

            // 遍历该特征点的所有观测
            for (int idx = 0; idx < it_per_id.feature_per_frame.size(); idx++)
            {
                int current_obs_frame_id = it_per_id.start_frame + idx;

                // 避免重复参数块：如果首次观测帧和当前观测帧是同一帧，跳过
                if (first_obs_frame_id == current_obs_frame_id)
                    continue;

                // 找到对应的 ImageFrame
                double timestamp = Headers[current_obs_frame_id].stamp.toSec();
                auto frame_it = all_image_frame.find(timestamp);

                // 检查深度图是否已计算
                if (frame_it == all_image_frame.end() || !frame_it->second.depth_map_computed)
                    continue;  // 没有可用的深度图

                frames_with_depth++;
                auto& frame = frame_it->second;
                const cv::Mat& depth_map = frame.predicted_depth_map;

                // 获取特征点的像素坐标
                Vector2d uv = it_per_id.feature_per_frame[idx].uv;

                // 边界检查
                int u = static_cast<int>(uv.x() + 0.5);
                int v = static_cast<int>(uv.y() + 0.5);
                if (v < 0 || v >= depth_map.rows || u < 0 || u >= depth_map.cols)
                    continue;

                // 查找测量值：从深度图中提取预测的逆深度
                // 假设 MiDaS 输出是 CV_32F 格式的归一化逆深度
                double d_nn = static_cast<double>(depth_map.at<float>(v, u));

                // 检查测量值有效性
                const double min_valid_inv_depth = 1e-6;
                const double max_valid_inv_depth = 100.0;
                if (d_nn <= min_valid_inv_depth || d_nn > max_valid_inv_depth)
                    continue;  // 无效测量

                // 创建深度因子
                DepthFactor* depth_factor = new DepthFactor(d_nn, pts_i);

                // 添加残差块：连接 Pose_i, Pose_j, Ex_Pose, Feature, ScaleShift
                problem.AddResidualBlock(depth_factor, depth_loss_function,
                    para_Pose[first_obs_frame_id],     // 首次观测帧位姿
                    para_Pose[current_obs_frame_id],   // 当前观测帧位姿
                    para_Ex_Pose[0],                   // 外参
                    para_Feature[feature_index],       // 特征点逆深度
                    para_DepthScaleShift[0]);          // 尺度偏移参数

                depth_factor_cnt++;
            }
        }

        ROS_DEBUG("depth factor count: %d, time: %.2f ms", depth_factor_cnt, t_depth_factor.toc());

        // 输出深度约束信息（添加更多诊断信息）
        if (depth_factor_cnt > 0)
        {
            ROS_INFO_THROTTLE(10.0,"[Backend] Depth factors: %d (checked %d features, %d frames with depth) | a=%.6f, b=%.6f",
                     depth_factor_cnt, features_checked, frames_with_depth, DEPTH_SCALE_A, DEPTH_SHIFT_B);
        }
        else
        {
            static int no_depth_count = 0;
            no_depth_count++;
            if (no_depth_count % 5 == 1)  // 每5帧打印一次
            {
                ROS_WARN_THROTTLE(5.0,
                         "[Backend] No depth factors! (checked %d features, %d frames with depth maps)",
                         features_checked, frames_with_depth);
            }
        }

        // --- 添加随机游走先验因子 (Random Walk Prior) ---
        // 如果有上一次的参数值，添加随机游走先验约束参数变化不要太大
        // 即使没有深度因子，这个先验也能提供约束，避免参数发散
        if (has_last_depth_params && solver_flag == NON_LINEAR)
        {
            // 根据是否是第一次优化来决定随机游走噪声的大小
            double current_rw_a = DEPTH_A_RANDOM_WALK;
            double current_rw_b = DEPTH_B_RANDOM_WALK;

            // 如果是第一次优化，放松约束 100 倍，允许参数从错误的初始化跳转到正确值
            // 这对于从糟糕的 SFM 初始化恢复非常重要
            if (is_first_depth_optimization)
            {
                current_rw_a *= 100.0;
                current_rw_b *= 100.0;
                ROS_WARN("[Depth Opt] Relaxing random walk constraint for FIRST optimization (sigma_a: %.6f -> %.6f, sigma_b: %.6f -> %.6f)",
                         DEPTH_A_RANDOM_WALK, current_rw_a, DEPTH_B_RANDOM_WALK, current_rw_b);
            }

            // 创建随机游走先验因子
            DepthScaleShiftRandomWalkFactor* random_walk_factor =
                new DepthScaleShiftRandomWalkFactor(
                    last_depth_a, last_depth_b,
                    current_rw_a, current_rw_b);

            // 添加残差块（不使用鲁棒核函数，因为这是一个软约束）
            problem.AddResidualBlock(random_walk_factor, nullptr, para_DepthScaleShift[0]);

            ROS_DEBUG("[Backend] Added random walk prior: last_a=%.6f, last_b=%.6f, sigma_a=%.6f, sigma_b=%.6f",
                     last_depth_a, last_depth_b, current_rw_a, current_rw_b);
        }
        else if (solver_flag == NON_LINEAR)
        {
            // 第一次优化，没有先验，但如果有深度因子就不需要固定参数
            if (depth_factor_cnt == 0)
            {
                // 没有深度因子也没有先验，必须固定参数避免欠约束
                ROS_DEBUG("No depth factors and no prior, fixing para_DepthScaleShift");
                problem.SetParameterBlockConstant(para_DepthScaleShift[0]);
            }
            else
            {
                ROS_INFO_THROTTLE(5.0, "[Backend] First optimization with depth factors, no random walk prior yet");
            }
        }
    }

    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    // d. 如果有重定位信息，添加闭环约束
    if(relocalization_info)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }

    // 3. 配置并运行Ceres求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
        
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    // 将优化结果转换回系统状态变量
    double2vector();

    // 4. 处理边缘化
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) // 边缘化最老的帧
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 如果有上一次的先验信息，将其传递给本次边缘化
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 将与最老帧相关的IMU约束和视觉约束添加到边缘化信息中
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        // --- 深度传感器因子约束：将与frame 0相关的深度因子添加到边缘化 (Backend Depth Constraint) ---
        if (ESTIMATE_DEPTH_SCALE_SHIFT)
        {
            // 创建鲁棒核函数（必须与optimization中使用的一致）
            ceres::LossFunction *depth_loss_function = new ceres::CauchyLoss(DEPTH_FACTOR_HUBER_THRESHOLD);

            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                // 使用与视觉约束相同的筛选条件
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int first_obs_frame_id = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历该特征点的所有观测
                for (int idx = 0; idx < it_per_id.feature_per_frame.size(); idx++)
                {
                    int current_obs_frame_id = it_per_id.start_frame + idx;

                    // 只添加与frame 0相关的因子到边缘化
                    // 条件：首次观测帧是0 或 当前观测帧是0
                    if (first_obs_frame_id != 0 && current_obs_frame_id != 0)
                        continue;

                    // 避免重复参数块：如果首次观测帧和当前观测帧是同一帧，跳过
                    if (first_obs_frame_id == current_obs_frame_id)
                        continue;

                    // 找到对应的 ImageFrame
                    double timestamp = Headers[current_obs_frame_id].stamp.toSec();
                    auto frame_it = all_image_frame.find(timestamp);

                    // 检查深度图是否已计算
                    if (frame_it == all_image_frame.end() || !frame_it->second.depth_map_computed)
                        continue;

                    auto& frame = frame_it->second;
                    const cv::Mat& depth_map = frame.predicted_depth_map;

                    // 获取特征点的像素坐标
                    Vector2d uv = it_per_id.feature_per_frame[idx].uv;

                    // 边界检查
                    int u = static_cast<int>(uv.x() + 0.5);
                    int v = static_cast<int>(uv.y() + 0.5);
                    if (v < 0 || v >= depth_map.rows || u < 0 || u >= depth_map.cols)
                        continue;

                    // 查找测量值
                    double d_nn = static_cast<double>(depth_map.at<float>(v, u));

                    // 检查测量值有效性
                    const double min_valid_inv_depth = 1e-6;
                    const double max_valid_inv_depth = 10.0;
                    if (d_nn <= min_valid_inv_depth || d_nn > max_valid_inv_depth)
                        continue;

                    // 创建深度因子
                    DepthFactor* depth_factor = new DepthFactor(d_nn, pts_i);

                    // 定义参数块
                    std::vector<double*> para_blocks = {
                        para_Pose[first_obs_frame_id],
                        para_Pose[current_obs_frame_id],
                        para_Ex_Pose[0],
                        para_Feature[feature_index],
                        para_DepthScaleShift[0]
                    };

                    // 定义哪些参数块需要被丢弃（边缘化）
                    std::vector<int> drop_set;
                    if (first_obs_frame_id == 0)
                        drop_set.push_back(0);  // para_Pose[0]
                    if (current_obs_frame_id == 0)
                        drop_set.push_back(1);  // para_Pose[0] (if first_obs != 0)
                    // 如果特征的起始帧是0，特征也会被边缘化
                    if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() == 1)
                        drop_set.push_back(3);  // para_Feature (只有一个观测的特征)

                    // 如果没有需要边缘化的参数块，跳过
                    if (drop_set.empty())
                    {
                        delete depth_factor;
                        continue;
                    }

                    // 添加到边缘化信息
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                        depth_factor, depth_loss_function, para_blocks, drop_set);

                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        // 执行边缘化，计算先验信息
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // 更新下一次优化所需的先验信息和参数块地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // --- 深度传感器因子约束：添加ScaleShift参数块地址映射 (Backend Depth Constraint) ---
        // para_DepthScaleShift是全局参数，在滑动窗口中保持不变
        if (ESTIMATE_DEPTH_SCALE_SHIFT)
        {
            addr_shift[reinterpret_cast<long>(para_DepthScaleShift[0])] = para_DepthScaleShift[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            // --- 深度传感器因子约束：添加ScaleShift参数块地址映射 (Backend Depth Constraint) ---
            if (ESTIMATE_DEPTH_SCALE_SHIFT)
            {
                addr_shift[reinterpret_cast<long>(para_DepthScaleShift[0])] = para_DepthScaleShift[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief 滑动窗口操作
 * 
 * 根据 marginalization_flag 的值，决定是移除最老的帧还是丢弃最新的帧，
 * 以维持滑动窗口的大小。
 */
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) // 移除最老的帧 (第0帧)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            // 如果在初始化阶段滑动窗口，说明初始化失败，需要重置深度图
            if (solver_flag == INITIAL)
            {
                if (m_first_frame_depth_computed)
                {
                     ROS_WARN("Fast-Init: Initialization failed, sliding oldest frame and resetting depth map.");
                    std::lock_guard<std::mutex> lock(m_depth_mutex);
                    m_first_frame_depth_computed = false;
                    m_first_frame_depth_map.release();
                }
            }

            // 将滑动窗口中的所有状态向前移动一格
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);
                std::swap(pre_integrations[i], pre_integrations[i + 1]);
                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            // 新的最后一帧状态暂时与前一帧相同
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            // 为新的最后一帧创建预积分对象
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
    
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else // 丢弃最新的帧 (WINDOW_SIZE)
    {
        if (frame_count == WINDOW_SIZE)
        {
            // 将最新帧的IMU数据合并到前一帧的预积分中
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            // 直接用最新帧的状态覆盖前一帧（因为前一帧被当作非关键帧丢弃了）
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // 为新的最后一帧创建预积分对象
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // 在特征管理器中移除最新帧
            slideWindowNew();
        }
    }
}

/**
 * @brief 在特征管理器中移除最新的帧（非关键帧）
 */
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

/**
 * @brief 在特征管理器中移除最老的帧（关键帧）
 */
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        // 如果系统已初始化，需要根据位姿变化来调整被移除帧中特征点的深度
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack(); // 如果未初始化，直接移除
}

/**
 * @brief 设置重定位帧的信息
 * @param _frame_stamp 重定位帧的时间戳
 * @param _frame_index 重定位帧的全局索引
 * @param _match_points 与重定位帧匹配的3D点
 * @param _relo_t 重定位帧的平移（真值）
 * @param _relo_r 重定位帧的旋转（真值）
 * 
 * 该函数由外部的重定位模块调用，用于传入闭环检测的结果。
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t; // 这是来自位姿图的位姿
    prev_relo_r = _relo_r;
    
    // 在当前滑动窗口中查找重定位帧
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        const double RELO_TIME_TOL = 0.01; // 10ms
        if (std::fabs(relo_frame_stamp - Headers[i].stamp.toSec()) < RELO_TIME_TOL)
        {
            relo_frame_local_index = i; // 找到它在滑动窗口中的局部索引
            relocalization_info = 1;    // 设置标志，以便在下一次优化中加入闭环约束
            // 记录下当前对该帧位姿的估计值
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

/**
 * @brief 从快速初始化结果为特征点分配estimated_depth
 *
 * 快速初始化内部计算了参数将网络逆深度转为正深度，这里利用这些参数
 * 和第一帧深度图为滑动窗口中的特征点设置estimated_depth，使得后续的
 * 在线参数估计能够工作。
 */
void Estimator::assignEstimatedDepthFromFastInit()
{
    if (!mp_fast_initializer) {
        ROS_WARN("assignEstimatedDepthFromFastInit: Fast initializer not available!");
        return;
    }

    // 1. 获取快速初始化计算的深度参数
    double a_fast, b_fast;
    if (!mp_fast_initializer->getLastDepthParams(a_fast, b_fast)) {
        ROS_WARN("assignEstimatedDepthFromFastInit: No valid depth parameters from fast init!");
        return;
    }

    // 2. 获取第一帧的深度图
    cv::Mat first_frame_depth_map;
    {
        std::lock_guard<std::mutex> lock(m_depth_mutex);
        if (!m_first_frame_depth_computed || m_first_frame_depth_map.empty()) {
            ROS_WARN("assignEstimatedDepthFromFastInit: First frame depth map not available!");
            return;
        }
        first_frame_depth_map = m_first_frame_depth_map.clone();
    }

    // 3. 遍历特征管理器中的所有特征点
    int assigned_count = 0;
    int total_features = 0;

    for (auto &it_per_id : f_manager.feature) {
        total_features++;

        // 跳过已经有有效estimated_depth的特征（避免重复赋值）
        if (it_per_id.estimated_depth > 0.0) {
            continue;
        }

        // 查找该特征在第一帧（滑动窗口的起始帧）的观测
        bool found_first_frame_obs = false;
        Eigen::Vector2d first_frame_uv;

        for (int obs_idx = 0; obs_idx < it_per_id.feature_per_frame.size(); obs_idx++) {
            int frame_id = it_per_id.start_frame + obs_idx;

            // 检查是否是第一帧（假设第一帧的frame_id为0）
            if (frame_id == 0) {
                first_frame_uv = it_per_id.feature_per_frame[obs_idx].uv;
                found_first_frame_obs = true;
                break;
            }
        }

        if (!found_first_frame_obs) {
            // 该特征在第一帧没有观测，跳过
            continue;
        }

        // 4. 从深度图获取网络预测的逆深度
        int u = static_cast<int>(first_frame_uv.x() + 0.5);
        int v = static_cast<int>(first_frame_uv.y() + 0.5);

        // 边界检查
        if (v < 0 || v >= first_frame_depth_map.rows ||
            u < 0 || u >= first_frame_depth_map.cols) {
            continue;
        }

        // 读取网络预测的归一化逆深度
        double d_net_inv = static_cast<double>(first_frame_depth_map.at<float>(v, u));

        // 检查深度值有效性
        if (!std::isfinite(d_net_inv) || d_net_inv <= 1e-6 || d_net_inv > 100.0) {
            continue;
        }

        // 5. 使用快速初始化的参数计算度量深度
        // 快速初始化内部关系: d_metric = a * 1/ d_net_inv + b
        double depth_metric = a_fast * 1 / d_net_inv + b_fast;


        // 检查最终深度合理性 (0.1m ~ 50m)
        if (depth_metric < 0.1 || depth_metric > 50.0) {
            continue;
        }

        // 6. 分配estimated_depth
        it_per_id.estimated_depth = depth_metric;
        assigned_count++;
    }

    ROS_INFO("assignEstimatedDepthFromFastInit: Assigned depth to %d / %d features using fast-init params (a=%.6f, b=%.6f)",
             assigned_count, total_features, a_fast, b_fast);

    if (assigned_count < 50) {
        ROS_WARN("assignEstimatedDepthFromFastInit: Low assignment success rate (%d features). Check depth map quality.",
                 assigned_count);
    }
}

/**
 * @brief 记录深度融合的关键指标到CSV文件
 *
 * 每10帧记录一次，包括IMU数据、运动评分、权重、Huber阈值和深度参数
 */
void Estimator::logDepthFusionMetrics(int frame_id, double gyro_norm, double acc_disturbance,
                                      double raw_score, double smoothed_score, double weight,
                                      double huber_threshold, double scale_a, double shift_b)
{
    // 每10帧记录一次
    log_frame_counter++;
    if (log_frame_counter % 10 != 0)
        return;

    if (!depth_fusion_log_file.is_open())
        return;

    // 写入CSV数据
    depth_fusion_log_file << frame_id << ","
                          << gyro_norm << ","
                          << acc_disturbance << ","
                          << raw_score << ","
                          << smoothed_score << ","
                          << weight << ","
                          << huber_threshold << ","
                          << scale_a << ","
                          << shift_b << ","
                          << DEPTH_WEIGHT_MODE << "\n";

    // 定期刷新缓冲区，确保数据写入磁盘
    if (log_frame_counter % 100 == 0)
    {
        depth_fusion_log_file.flush();
    }
}
