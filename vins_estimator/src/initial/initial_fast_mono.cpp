#include "initial_fast_mono.h"

// 构造函数
FastInitializer::FastInitializer(FeatureManager* f_manager_ptr)
    : m_feature_manager(f_manager_ptr),
      m_random_generator(std::random_device{}()) // 初始化随机数生成器
{
}

// 计算滑窗起始帧ID：返回当前窗口内所有特征 start_frame 的最小值（若为空返回0）
int FastInitializer::computeWindowStartFrameId() const
{
    int window_start_frame_id = -1;
    if (m_feature_manager && !m_feature_manager->feature.empty())
    {
        // 初始化为第一个特征的 start_frame
        window_start_frame_id = m_feature_manager->feature.front().start_frame;
        for (const auto& f : m_feature_manager->feature)
        {
            if (f.start_frame < window_start_frame_id)
                window_start_frame_id = f.start_frame;
        }
        if (window_start_frame_id < 0)
            window_start_frame_id = 0;
    }
    return (window_start_frame_id >= 0) ? window_start_frame_id : 0;
}

// 主初始化函数
bool FastInitializer::initialize(const std::map<double, ImageFrame>& image_frames,
                               const cv::Mat& first_frame_norm_inv_depth, // CV_32F, [1,2]
                               Eigen::Vector3d& G_gravity_world, // Global gravity (e.g., [0, 0, 9.8])
                               std::map<int, Eigen::Vector3d>& Ps_out,
                               std::map<int, Eigen::Vector3d>& Vs_out,
                               std::map<int, Eigen::Quaterniond>& Rs_out)
{
    // ========================================================================
    // 步骤 1: 数据准备与验证
    // ========================================================================
    ROS_INFO("FastInit: Collecting observations...");
    
    // 1.1 计算复合 IMU 预积分（I0 -> I1, I0 -> I2, ..., I0 -> Ik） Ik变换到I0的旋转矩阵
    std::vector<IntegrationBase*> pre_integrations_compound;
    if (!computeCompoundPreIntegrations(image_frames, pre_integrations_compound)) {
        ROS_WARN("FastInit: Failed to compute compound pre-integrations.");
        return false;
    }
    
    // 确保在函数退出时清理预积分内存（RAII风格的清理器）
    struct IntegrationBaseCleaner {
        std::vector<IntegrationBase*>& vec;
        ~IntegrationBaseCleaner() {
            for (IntegrationBase* p : vec) delete p;
        }
    } cleaner{pre_integrations_compound};
    
    // 1.2 收集所有有效的特征观测数据
    int window_start_frame_id = computeWindowStartFrameId();
    std::vector<ObservationData> all_observations;
    int num_obs = collectValidObservations(
        first_frame_norm_inv_depth,
        window_start_frame_id,
        pre_integrations_compound,
        all_observations);
    
    ROS_INFO("FastInit: Collected %zu observations from %d features.", 
             all_observations.size(), num_obs);
    
    // 检查是否有足够的观测来进行 RANSAC
    if (all_observations.size() < fast_mono::RANSAC_MIN_MEASUREMENTS) {
        ROS_WARN("FastInit: Not enough valid observations (%zu < %d) to initialize.", 
                 all_observations.size(), fast_mono::RANSAC_MIN_MEASUREMENTS);
        return false;
    }
    
    // ========================================================================
    // 步骤 2: RANSAC 鲁棒求解
    // ========================================================================
    ROS_INFO("FastInit: Starting RANSAC with %zu observations...", all_observations.size());
    Eigen::Matrix<double, 8, 1> x_best; // 最优解 [a, b, v_I0(3), g_I0(3)]^T
    if (!solveRANSAC(all_observations, x_best)) {
        ROS_WARN("FastInit: RANSAC failed to find a valid solution.");
        return false;
    }
    
    // ========================================================================
    // 步骤 3: 解析 RANSAC 解并验证
    // ========================================================================
    double a = x_best(0);                         // 深度尺度因子
    double b = x_best(1);                         // 深度偏移
    Eigen::Vector3d v_I0_in_I0 = x_best.segment<3>(2); // 第一帧速度（I0系）
    Eigen::Vector3d g_in_I0 = x_best.segment<3>(5);     // 重力向量（I0系）
    
    // 验证解的物理合理性
    if (!isValidSolution(x_best)) {
        ROS_WARN("FastInit: RANSAC solution failed physical validity check.");
        return false;
    }
    
    // ========================================================================
    // 步骤 4: 计算特征点深度统计并验证
    // ========================================================================
    DepthStatistics depth_stats;
    if (!computeDepthStatistics(first_frame_norm_inv_depth, a, b, 
                               window_start_frame_id, depth_stats)) {
        ROS_WARN("FastInit: Failed to compute depth statistics.");
        return false;
    }
    
    ROS_INFO("FastInit depth stats: a=%.6f b=%.6f | total=%d valid=%d | "
             "z[%.3f~%.3f] mean=%.3f", 
             a, b, depth_stats.total_count, depth_stats.valid_count,
             depth_stats.min_depth, depth_stats.max_depth, depth_stats.mean_depth);
    
    // 验证有效深度特征数量
    const int MIN_VALID_DEPTH_FEATURES = 10;
    if (depth_stats.valid_count < MIN_VALID_DEPTH_FEATURES) {
        ROS_WARN("FastInit: Not enough valid depth features (%d < %d).", 
                 depth_stats.valid_count, MIN_VALID_DEPTH_FEATURES);
        return false;
    }
    
    // ========================================================================
    // 步骤 5: 坐标系对齐（I0 系 -> W' 重力对齐系）
    // ========================================================================
    Eigen::Quaterniond R_I0_to_W_prime;
    Eigen::Vector3d v_I0_in_W_prime;
    alignCoordinateSystem(g_in_I0, v_I0_in_I0, G_gravity_world, 
                         R_I0_to_W_prime, v_I0_in_W_prime);
    
    // 第一帧在 W' 系下的位置设为原点
    Eigen::Vector3d p_I0_in_W_prime = Eigen::Vector3d::Zero();
    
    // ========================================================================
    // 步骤 6: 前向传播状态到所有帧
    // ========================================================================
    if (!propagateStatesToAllFrames(image_frames, R_I0_to_W_prime, 
                                    p_I0_in_W_prime, v_I0_in_W_prime,
                                    G_gravity_world, Ps_out, Vs_out, Rs_out)) {
        ROS_WARN("FastInit: Failed to propagate states to all frames.");
        return false;
    }
    
    // ========================================================================
    // 步骤 7: 输出成功信息
    // ========================================================================
    ROS_INFO("FastInit: Initialization successful!");
    ROS_INFO("  Depth model: a=%.3f, b=%.3f", a, b);
    ROS_INFO("  Initial velocity (W'): [%.3f, %.3f, %.3f] m/s", 
             v_I0_in_W_prime.x(), v_I0_in_W_prime.y(), v_I0_in_W_prime.z());
    ROS_INFO("  Gravity (W'): [%.3f, %.3f, %.3f] m/s² (norm: %.3f)", 
             G_gravity_world.x(), G_gravity_world.y(), G_gravity_world.z(), 
             G_gravity_world.norm());
    
    return true;
}


// 构建线性系统的一行 (对应论文 Eq. 23)
void FastInitializer::buildLinearSystemRow(const IntegrationBase* pre_int_k, // 输入: 从 I0 到 Ik 的 IMU 预积分对象 (包含 delta_q, delta_p, sum_dt)
                                         const Eigen::Vector3d& z_i0,      // 输入: 特征 i 在 C0 帧的归一化坐标 (单位向量 [x, y, 1], 对应论文 \bar{z}_{i,0})
                                         const Eigen::Vector3d& z_ik,      // 输入: 特征 i 在 Ck 帧的归一化坐标 (单位向量 [x, y, 1], 对应论文 \bar{z}_{i,k})
                                         double d_hat_i,                 // 输入: 特征 i 在 C0 帧的归一化逆深度 (来自深度网络, 对应论文 \hat{d}_i)
                                         Eigen::Matrix<double, 2, 8>& A_row, // 输出: 填充好的 A' 矩阵的 2 行
                                         Eigen::Vector2d& b_row)        // 输出: 填充好的 b' 向量的 2 行
{
    // --- 1. 提取外参 (VINS-Mono 标准: Camera-to-IMU) ---

    // R_c_i = R^C_I = {}_{C}^{I}R (Camera -> IMU 旋转)
    // 这就是 VINS-Mono 中 `RIC` 的定义
    const Eigen::Matrix3d& R_c_i = RIC[0]; 

    // T_c_i_in_I = {}^{I}p_C (Camera 光心在 IMU 系下的平移)
    // 这就是 VINS-Mono 中 `TIC` 的定义
    const Eigen::Vector3d& T_c_i_in_I = TIC[0]; 

    // --- 派生逆变换 (IMU-to-Camera) ---

    // R_i_c = R^I_C = {}_{I}^{C}R (IMU -> Camera 旋转)
    // 它是 R_c_i 的逆 (转置)
    Eigen::Matrix3d R_i_c = R_c_i.transpose(); 

    // T_i_c_in_C = {}^{C}p_I (IMU 原点在 Camera 系下的平移)
    //
    // 推导:
    // P_I = R_c_i * P_C + T_c_i_in_I
    // P_C = (R_c_i)^-1 * (P_I - T_c_i_in_I)
    // P_C = R_i_c * P_I - R_i_c * T_c_i_in_I
    //
    // 比较 P_C = R_i_c * P_I + T_i_c_in_C
    // 可得: T_i_c_in_C = -R_i_c * T_c_i_in_I
    Eigen::Vector3d T_i_c_in_C = -R_i_c * T_c_i_in_I;

    // --- 2. 提取 IMU 预积分信息 (I0 -> Ik) ---
    Eigen::Matrix3d R_Ik_I0 = pre_int_k->delta_q.toRotationMatrix();
    Eigen::Matrix3d R_I0_Ik = R_Ik_I0.transpose();
    Eigen::Vector3d I0_alpha_Ik = pre_int_k->delta_p;
    double delta_t = pre_int_k->sum_dt;
    double delta_t_sq = delta_t * delta_t;

    // --- 3. 计算论文中的关键中间变量 ---
    Eigen::Matrix3d z_ik_hat = Utility::skewSymmetric(z_ik); // Gamma_i,k
    // Upsilon = Gamma_i,k * R_i_c * R_I0_Ik
    Eigen::Matrix3d Upsilon_3x3 = z_ik_hat * R_i_c * R_I0_Ik;

    // d_i = 1 / d_hat_i (d_hat_i 为网络输出的“归一化逆深度”)
    double d_i = 1.0 / d_hat_i;

    // {}^{I_0}\theta_{C_0 \to f_i} = R_c_i * z_i0
    Eigen::Vector3d I0_theta_C0_fi = R_c_i * z_i0;

    // --- 4. 构建 A' 矩阵的系数 [M1, M2, Tv, Tg] ---
    Eigen::Vector3d M1_3d = Upsilon_3x3 * d_i * I0_theta_C0_fi;  // a
    Eigen::Vector3d M2_3d = Upsilon_3x3 * I0_theta_C0_fi;        // b
    Eigen::Matrix<double, 3, 3> Tv_3d = -Upsilon_3x3 * delta_t;  // v_I0
    Eigen::Matrix<double, 3, 3> Tg_3d =  Upsilon_3x3 * (0.5 * delta_t_sq); // g_I0

    // --- 5. 构建 b' 向量 (RHS) ---
    // b' = Upsilon * alpha - Upsilon * ^I p_C - Gamma * ^C p_I
    Eigen::Vector3d b_prime_3d = Upsilon_3x3 * I0_alpha_Ik
                               - Upsilon_3x3 * T_c_i_in_I
                               - z_ik_hat * T_i_c_in_C;

    // --- 6. 填充 A_row (2x8) 和 b_row (2x1) ---
    A_row.block<2, 1>(0, 0) = M1_3d.head<2>();           // a
    A_row.block<2, 1>(0, 1) = M2_3d.head<2>();           // b
    A_row.block<2, 3>(0, 2) = Tv_3d.block<2, 3>(0, 0);   // v_I0
    A_row.block<2, 3>(0, 5) = Tg_3d.block<2, 3>(0, 0);   // g_I0
    b_row = b_prime_3d.head<2>();

    // 函数结束时，A_row 和 b_row 就包含了由这一个 (i, k) 观测所贡献的两个线性方程。
    // RANSAC 和最终求解函数会调用这个函数多次，将得到的 A_row 和 b_row 堆叠起来，
    // 构成最终需要求解的 (超定) 线性系统 A'x' = b'。
}


// RANSAC 求解器
bool FastInitializer::solveRANSAC(const std::vector<ObservationData>& all_observations,
                                Eigen::Matrix<double, 8, 1>& best_x_out) // 输出最终拟合后的解
{
    int num_observations = all_observations.size();
    std::vector<int> best_inlier_indices;
    // 初始化 best_x_ransac 为无效值，存储 RANSAC 迭代中找到的最佳候选解
    Eigen::Matrix<double, 8, 1> best_x_ransac = Eigen::Matrix<double, 8, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
    double best_condition_number = std::numeric_limits<double>::max();

    // 检查是否有足够的观测数据来进行最小集采样
    // 建议 RANSAC_MIN_MEASUREMENTS 设回 4 或 5
    const int current_min_measurements = fast_mono::RANSAC_MIN_MEASUREMENTS;
    if (num_observations < current_min_measurements) {
        ROS_WARN("RANSAC Error: Not enough observations (%d) available to form a minimal set (%d required).",
                 num_observations, current_min_measurements);
        return false;
    }

    std::uniform_int_distribution<int> distribution(0, num_observations - 1);

    // --- RANSAC 主循环 ---
    for (int iter = 0; iter < fast_mono::RANSAC_MAX_ITERATIONS; ++iter)
    {
        // 1. 随机采样最小集
        std::set<int> used_indices;  // 使用set提高查找效率
        std::vector<ObservationData> minimal_set;
        while (minimal_set.size() < current_min_measurements)
        {
            int rand_idx = distribution(m_random_generator);
            // 检查是否已使用且数据有效
            if (used_indices.find(rand_idx) != used_indices.end()) {
                continue;  // 已使用，跳过
            }
            if (!all_observations[rand_idx].pre_integration_k) {
                continue;  // 数据无效，跳过
            }
            used_indices.insert(rand_idx);
            minimal_set.push_back(all_observations[rand_idx]);
        }

        // 2. 构建最小系统的 A_min 和 b_min
        int n_rows = current_min_measurements * 2;
        // 根据是否为方阵选择 Dynamic 或固定大小
        Eigen::Matrix<double, Eigen::Dynamic, 8> A_min(n_rows, 8);
        Eigen::Matrix<double, Eigen::Dynamic, 1> b_min(n_rows);
        
        // (构建 A_min, b_min 的循环, 包括检查 NaN)
        bool build_ok = true;
        for (int i = 0; i < current_min_measurements; ++i) {
            Eigen::Matrix<double, 2, 8> A_row;
            Eigen::Vector2d b_row;
            buildLinearSystemRow(minimal_set[i].pre_integration_k, minimal_set[i].z_i0,
                                 minimal_set[i].z_ik, minimal_set[i].d_hat_i, A_row, b_row);
            if (A_row.hasNaN() || b_row.hasNaN()) {
                build_ok = false; break;
            }
            A_min.block<2, 8>(i * 2, 0) = A_row;
            b_min.block<2, 1>(i * 2, 0) = b_row;
        }
        if (!build_ok) continue;

        // 3. 应用数值预处理 (缩放矩阵 S)
        Eigen::Matrix<double, 8, 8> S = Eigen::Matrix<double, 8, 8>::Identity();
        S(0, 0) = 0.1;   // a: 深度尺度因子，通常在0.01-1范围内，缩放到合理尺度
        S(1, 1) = 1.0;   // b: 深度偏移，通常较小，保持原尺度
        S(2, 2) = 10.0; S(3, 3) = 10.0; S(4, 4) = 10.0; // v (m/s)
        S(5, 5) = 100.0; S(6, 6) = 100.0; S(7, 7) = 100.0; // g (m/s²)
        Eigen::MatrixXd Am_S = A_min * S;

        // 4. 计算 SVD 和条件数
        Eigen::JacobiSVD<Eigen::MatrixXd> svd;
        if (n_rows == 8) {
             svd.compute(Am_S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        } else {
             svd.compute(Am_S, Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        double cond_num = std::numeric_limits<double>::infinity();
        const double min_singular_value_threshold = 1e-8;
        if (svd.singularValues().size() > 0 && svd.singularValues().minCoeff() > min_singular_value_threshold) {
            cond_num = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
        }

        // --- 条件数阈值检查 ---
        const double CONDITION_NUMBER_THRESHOLD = 1e8; // << 需要调整
        if (cond_num > CONDITION_NUMBER_THRESHOLD || std::isinf(cond_num) || std::isnan(cond_num))
        {
            // if (iter < 20) ROS_WARN("RANSAC Iter %d: Poor condition number (%.2e). Discarding sample.", iter, cond_num);
            continue; // 条件数太差，放弃本次采样 
        }

        // 5. SVD 求解
        Eigen::Matrix<double, 8, 1> x_scaled = svd.solve(b_min);
        if (x_scaled.hasNaN()) continue; // 跳过 NaN 解

        // 6. 恢复原始尺度的解 x_candidate
        Eigen::Matrix<double, 8, 1> x_candidate = S * x_scaled;

        // 7. 物理合理性检查 (g_norm 过滤器)
        double g_norm_cand = x_candidate.segment<3>(5).norm();
        if (g_norm_cand < 5.0 || g_norm_cand > 15.0)
        {
            // if (iter < 20) ROS_WARN("RANSAC Iter %d: Good cond (%.2e), bad g_norm (%.3f). Skipping.", iter, cond_num, g_norm_cand);
            continue; // 跳过物理上不可能的解
        }

        double a_cand = x_candidate(0);
        if (a_cand < 1e-3 || a_cand > 100.0) {
            continue; // a 必须在合理范围内
        }

        // --- 找到一个数值和物理都合理的候选解 ---

        // 8. 检验内点
        std::vector<int> current_inlier_indices;
        for (int i = 0; i < num_observations; ++i) {
            // ... (构建 A_row, b_row) ...
            Eigen::Matrix<double, 2, 8> A_row;
            Eigen::Vector2d b_row;
            buildLinearSystemRow(all_observations[i].pre_integration_k, all_observations[i].z_i0,
                                 all_observations[i].z_ik, all_observations[i].d_hat_i, A_row, b_row);

            double residual_sq_norm = (A_row * x_candidate - b_row).squaredNorm();
            if (std::isnan(residual_sq_norm) || std::isinf(residual_sq_norm)) continue;

            // 建议 RANSAC_THRESHOLD_SQ 设回较小值，如 0.1*0.1
            if (residual_sq_norm < fast_mono::RANSAC_THRESHOLD_SQ) {
                current_inlier_indices.push_back(i);
            }
        }

        // 9. 更新最优模型 (只关心内点数)
        if (current_inlier_indices.size() > best_inlier_indices.size()) {
            best_inlier_indices = current_inlier_indices;
            best_x_ransac = x_candidate; // 存储当前这个最好的候选解
            best_condition_number = cond_num;
            ROS_INFO("RANSAC Iter %d: Found new best model with %zu inliers (Cond Num=%.2e, g_norm=%.3f)",
                      iter, best_inlier_indices.size(), best_condition_number, g_norm_cand);
             // (可选) 提前退出
             // if (best_inlier_indices.size() > num_observations * 0.5) break;
        }

    } // --- RANSAC 主循环结束 ---

    ROS_INFO("RANSAC finished. Best model has %zu inliers.", best_inlier_indices.size());

    // 10. 检查是否找到足够内点
    if (best_inlier_indices.size() < fast_mono::RANSAC_MIN_INLIERS) {
        ROS_WARN("RANSAC failed: Not enough inliers found (%zu / %d required).",
                 best_inlier_indices.size(), fast_mono::RANSAC_MIN_INLIERS);
        return false;
    }

    // --- RANSAC 成功：我们找到了一个可靠的内点集 best_inlier_indices ---

    // 11. 使用所有内点进行最终的线性拟合 (调用 solveLinearSystem)
    ROS_INFO("RANSAC successful. Performing final fit using %zu inliers...", best_inlier_indices.size());
    std::vector<ObservationData> best_inliers_data;
    best_inliers_data.reserve(best_inlier_indices.size()); // 预分配内存
    for (int idx : best_inlier_indices) {
        // 安全检查，以防 all_observations 在 RANSAC 过程中被意外修改 (理论上不应发生)
        if (idx >= 0 && idx < all_observations.size()) {
             best_inliers_data.push_back(all_observations[idx]);
        } else {
             ROS_ERROR("RANSAC Error: Invalid inlier index %d found!", idx);
             return false; // 索引无效，严重错误
        }
    }

    // 调用最终拟合函数，并将结果存储在 best_x_out
    if (!solveLinearSystem(best_inliers_data, best_x_out)) {
        ROS_WARN("RANSAC succeeded in finding inliers, but the final linear system solve failed.");
        return false;
    }

    // 最终检查 solveLinearSystem 的结果是否合理
    double final_g_norm = best_x_out.segment<3>(5).norm();
    double final_a = best_x_out(0);
    double final_b = best_x_out(1);\
    
    if (final_g_norm < 5.0 || final_g_norm > 15.0 || 
        final_a < 1e-3 || final_a > 100.0 ||
        std::abs(final_b) > 1.0) 
    {
        ROS_WARN("Final solution after solveLinearSystem seems unreasonable (g_norm=%.3f, a=%.6f, b=%.6f). Initialization failed.", 
                 final_g_norm, final_a, final_b);
        return false;
    }

    ROS_INFO("Final solution obtained successfully after final fit.");
    return true; // RANSAC 和最终拟合都成功
}


// 使用所有内点求解线性系统 (带简化的重力约束)
bool FastInitializer::solveLinearSystem(const std::vector<ObservationData>& observations,
                                        Eigen::Matrix<double, 8, 1>& x_out)
{
    int num_inliers = observations.size();
    // 确保至少有足够的方程来求解 (理论上 4 个观测就够，但越多越好)
    if (num_inliers < 4) {
         ROS_WARN("solveLinearSystem Error: Too few observations (%d) for final fit.", num_inliers);
         return false;
    }

    // --- 1. 构建超定系统 Ax = b ---
    int n_rows = num_inliers * 2;
    Eigen::MatrixXd A(n_rows, 8); // 使用动态大小
    Eigen::VectorXd b(n_rows);

    bool build_ok = true;
    for (int i = 0; i < num_inliers; ++i)
    {
        // 安全检查
        if (!observations[i].pre_integration_k) {
            ROS_ERROR("solveLinearSystem Error: Observation %d has null pre_integration!", i);
            return false;
        }
        Eigen::Matrix<double, 2, 8> A_row;
        Eigen::Vector2d b_row;
        buildLinearSystemRow(observations[i].pre_integration_k, observations[i].z_i0,
                             observations[i].z_ik, observations[i].d_hat_i, A_row, b_row);

        if (A_row.hasNaN() || b_row.hasNaN()) {
             ROS_WARN("solveLinearSystem Warning: NaN detected while building row %d.", i);
             // 策略：可以选择跳过这个观测，或者直接返回失败
             // return false; // 更安全的选择
             build_ok = false; break; // 或者跳出循环
        }
        A.block<2, 8>(i * 2, 0) = A_row;
        b.block<2, 1>(i * 2, 0) = b_row;
    }
     if (!build_ok) {
        ROS_ERROR("solveLinearSystem Error: Failed to build the system due to NaN values.");
        return false;
     }

    // --- 2. 应用数值预处理 (缩放矩阵 S) ---
    Eigen::VectorXd col_scale(8);
    for (int j = 0; j < 8; ++j) {
        double s = A.col(j).norm() / std::sqrt(static_cast<double>(A.rows()));
        if (s < 1e-8) s = 1.0;
        col_scale(j) = s;
    }
    Eigen::Matrix<double,8,8> S = Eigen::Matrix<double,8,8>::Identity();
    for (int j = 0; j < 8; ++j) S(j,j) = 1.0 / col_scale(j);
    Eigen::MatrixXd A_S = A * S; // 缩放后的矩阵

    // --- 3. 求解无约束最小二乘问题 (SVD) ---
    // 关键: A_S 是 N x 8 (N >= 8)，使用 ThinU/V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_unconstrained(A_S, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // 检查 无约束 SVD 是否成功以及条件数
    double cond_num_unconstrained = std::numeric_limits<double>::infinity();
    const double min_singular_value_threshold = 1e-8;
    if (svd_unconstrained.singularValues().size() > 0 && svd_unconstrained.singularValues().minCoeff() > min_singular_value_threshold) {
        cond_num_unconstrained = svd_unconstrained.singularValues().maxCoeff() / svd_unconstrained.singularValues().minCoeff();
    }
    ROS_INFO("solveLinearSystem (Unconstrained): Condition number = %.2e", cond_num_unconstrained);


    Eigen::Matrix<double, 8, 1> x_scaled_svd = svd_unconstrained.solve(b);
    if (x_scaled_svd.hasNaN()) {
         ROS_ERROR("solveLinearSystem Error: Unconstrained SVD solve resulted in NaN!");
         return false;
    }
    Eigen::Matrix<double, 8, 1> x_svd = S * x_scaled_svd; // 恢复原始尺度

    // --- 4. 施加重力大小约束 ---
    Eigen::Vector3d g_svd = x_svd.segment<3>(5);
    double g_norm_svd = g_svd.norm();

    // 检查无约束解的有效性
    if (g_norm_svd < 1.0 || a_svd < 1e-3) { 
        ROS_WARN("solveLinearSystem Warning: Unconstrained solve resulted in invalid values (g_norm=%.3f, a=%.6f).", 
                 g_norm_svd, a_svd);
        // 如果无约束解已经不合理，直接返回失败
        if (a_svd < 1e-3) {
            ROS_ERROR("solveLinearSystem Error: Unconstrained solution has invalid a=%.6f. Failed.", a_svd);
            return false;
        }
        // 如果只是重力小，尝试使用无约束解
        x_out = x_svd;
        if (std::abs(x_out(0)) < 1e-4 || x_out.segment<3>(5).norm() < 1.0) {
            ROS_ERROR("solveLinearSystem Error: Unconstrained solution is also unreasonable. Failed.");
            return false;
        }
        return true;
    }

    // 使用无约束解的方向 和 全局参数 G.norm() 确定约束后的重力
    Eigen::Vector3d g_normed = g_svd.normalized() * G.norm();
    ROS_INFO("g_svd: [%.6f, %.6f, %.6f], norm: %.6f", 
        g_svd.x(), g_svd.y(), g_svd.z(), g_svd.norm());

    // --- 5. 固定 g_normed，重新求解 y = [a, b, v_I0]^T ---
    Eigen::MatrixXd A_y = A.leftCols<5>(); // 前 5 列
    Eigen::MatrixXd A_g = A.rightCols<3>(); // 后 3 列
    Eigen::VectorXd b_y = b - A_g * g_normed; // 新的 RHS

    // --- 6. 自适应列缩放（替代固定 S_y） ---
    Eigen::VectorXd col_scale_y(5);
    for (int j = 0; j < 5; ++j) {
        double s = A_y.col(j).norm() / std::sqrt(static_cast<double>(A_y.rows()));
        if (s < 1e-8) s = 1.0;
        col_scale_y(j) = s;
    }
    Eigen::Matrix<double,5,5> S_y = Eigen::Matrix<double,5,5>::Identity();
    for (int j = 0; j < 5; ++j) S_y(j,j) = 1.0 / col_scale_y(j);
    Eigen::MatrixXd Ay_Sy = A_y * S_y;

    // 7. IRLS 参数（Huber）与正则（Tikhonov），可配置
    const int irls_iters = 3;
    const double huber_delta = 1.5e-2; // 和像平面噪声量级匹配
    const double lambda_a = 1e-2;
    const double lambda_b = 5e-3;      // 稍强，抑制 b 漂移
    const double lambda_v = 1e-3;

    // 初始化 y（用一次最小二乘作为初值）
    Eigen::MatrixXd H0 = Ay_Sy.transpose() * Ay_Sy;
    Eigen::VectorXd g0 = Ay_Sy.transpose() * b_y;
    Eigen::Matrix<double,5,5> Reg = Eigen::Matrix<double,5,5>::Zero();
    Reg(0,0) = lambda_a;       // a
    Reg(1,1) = lambda_b;       // b
    Reg.block<3,3>(2,2).setIdentity();
    Reg.block<3,3>(2,2) *= lambda_v;

    Eigen::Matrix<double,5,1> y_scaled = (H0 + Reg).ldlt().solve(g0);

    // 8) IRLS 迭代
    for (int it = 0; it < irls_iters; ++it) {
        Eigen::VectorXd r = Ay_Sy * y_scaled - b_y; // N
        // 行权重（Huber）
        Eigen::VectorXd w = Eigen::VectorXd::Ones(r.size());
        for (int i = 0; i < r.size(); ++i) {
            double a = std::abs(r(i));
            if (a > huber_delta) w(i) = huber_delta / a;
        }
        // 构造加权 A、b
        Eigen::VectorXd sqrtw = w.array().sqrt();
        Eigen::MatrixXd Aw = Ay_Sy;
        for (int i = 0; i < Aw.rows(); ++i) Aw.row(i) *= sqrtw(i);
        Eigen::VectorXd bw = b_y.cwiseProduct(sqrtw);

        // 正则化加权正规方程
        Eigen::MatrixXd H = Aw.transpose() * Aw + Reg;
        Eigen::VectorXd gvec = Aw.transpose() * bw;
        y_scaled = H.ldlt().solve(gvec);
    }

    // 恢复原尺度并组装解
    Eigen::Matrix<double,5,1> y = S_y * y_scaled; // 这里 S_y 是 1/scale，对应逆回来
    x_out.head<5>() = y;
    x_out.tail<3>() = g_normed;

    // 记录加权系统的近似条件数
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_w(Ay_Sy, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double cond_w = svd_w.singularValues()(0) / svd_w.singularValues().tail(1)(0);
    ROS_INFO("solveLinearSystem (Constrained+IRLS): approx cond = %.2e", cond_w);

    ROS_INFO("solveLinearSystem finished successfully.");
    return true;
}


bool FastInitializer::isValidDepth(double d_hat_inv)
{
    // 深度值必须：有限、大于0、在合理范围内（归一化逆深度通常为[1,2]）
    return std::isfinite(d_hat_inv) && d_hat_inv > 0.0 && d_hat_inv <= 10.0;
}

bool FastInitializer::isValidPixelCoord(int u, int v, int rows, int cols)
{
    return (u >= 0 && u < cols && v >= 0 && v < rows);
}

bool FastInitializer::isValidNormalizedCoord(const Eigen::Vector3d& z)
{
    // 归一化坐标的第三个分量应该接近1，且所有分量有限
    return z.allFinite() && z.z() > 1e-6;
}

bool FastInitializer::getValidDepthFromMap(const cv::Mat& depth_map, int u, int v, double& d_out)
{
    if (!isValidPixelCoord(u, v, depth_map.rows, depth_map.cols)) {
        return false;
    }
    
    // 双线性插值读取深度
    int u0 = static_cast<int>(std::floor(u));
    int v0 = static_cast<int>(std::floor(v));
    int u1 = std::min(u0 + 1, depth_map.cols - 1);
    int v1 = std::min(v0 + 1, depth_map.rows - 1);
    double du = u - u0;
    double dv = v - v0;

    float d00 = depth_map.at<float>(v0, u0);
    float d01 = depth_map.at<float>(v0, u1);
    float d10 = depth_map.at<float>(v1, u0);
    float d11 = depth_map.at<float>(v1, u1);

    auto valid = [](float x){ return std::isfinite(x) && x > 0.f && x <= 10.f; };
    if (!valid(d00) || !valid(d01) || !valid(d10) || !valid(d11)) return false;

    double d0 = d00 * (1.0 - du) + d01 * du;
    double d1 = d10 * (1.0 - du) + d11 * du;
    d_out = d0 * (1.0 - dv) + d1 * dv;

    return isValidDepth(d_out);
}

bool FastInitializer::computeCompoundPreIntegrations(
    const std::map<double, ImageFrame>& image_frames,
    std::vector<IntegrationBase*>& pre_integrations_out)
{
    pre_integrations_out.clear();
    pre_integrations_out.reserve(image_frames.size());
    
    // 第一帧的预积分为单位预积分（从 I0 到 I0）
    IntegrationBase* I0_to_I0 = new IntegrationBase(
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    pre_integrations_out.push_back(I0_to_I0);
    
    // 遍历后续帧，计算复合预积分
    int k_index = 0;
    for (const auto& pair : image_frames) {
        if (k_index > 0) {  // 跳过第一帧
            IntegrationBase* prev_compound = pre_integrations_out.back();
            IntegrationBase* delta_integration = pair.second.pre_integration;
            
            if (!delta_integration) {
                ROS_ERROR("FastInit: Missing pre_integration for frame %d!", k_index);
                return false;
            }
            
            // 复合预积分公式：I0->Ik = (I0->I_{k-1}) * (I_{k-1}->Ik)
            IntegrationBase* new_compound = new IntegrationBase(
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            
            // 时间累积
            new_compound->sum_dt = prev_compound->sum_dt + delta_integration->sum_dt;
            
            // 旋转复合：R_Ik^I0 = R_{I_{k-1}}^I0 * R_Ik^{I_{k-1}}
            new_compound->delta_q = prev_compound->delta_q * delta_integration->delta_q;
            
            // 速度复合：v_Ik^I0 = v_{I_{k-1}}^I0 + R_{I_{k-1}}^I0 * delta_v
            new_compound->delta_v = prev_compound->delta_v + 
                                   prev_compound->delta_q * delta_integration->delta_v;
            
            // 位置复合：p_Ik^I0 = p_{I_{k-1}}^I0 + v_{I_{k-1}}^I0 * dt + R_{I_{k-1}}^I0 * delta_p
            new_compound->delta_p = prev_compound->delta_p
                                  + prev_compound->delta_v * delta_integration->sum_dt
                                  + prev_compound->delta_q * delta_integration->delta_p;
            
            pre_integrations_out.push_back(new_compound);
        }
        k_index++;
    }
    
    return true;
}

int FastInitializer::collectValidObservations(
    const cv::Mat& depth_map,
    int window_start_frame_id,
    const std::vector<IntegrationBase*>& pre_integrations_compound,
    std::vector<ObservationData>& observations_out)
{
    observations_out.clear();
    int feature_count = 0;
    int rows = depth_map.rows;
    int cols = depth_map.cols;
    
    for (const auto& feature : m_feature_manager->feature) {
        // 只处理在第一帧（窗口起始帧）被观测到的特征
        if (feature.start_frame != window_start_frame_id || 
            feature.feature_per_frame.empty()) {
            continue;
        }
        
        feature_count++;
        
        // 获取第一帧的观测
        const FeaturePerFrame& obs_frame_0 = feature.feature_per_frame[0];
        Eigen::Vector3d z_i0 = obs_frame_0.point;
        
        // 验证归一化坐标
        if (!isValidNormalizedCoord(z_i0)) {
            continue;
        }
        
        // 获取深度值
        int u = static_cast<int>(std::round(obs_frame_0.uv.x()));
        int v = static_cast<int>(std::round(obs_frame_0.uv.y()));
        double d_hat_i_inv;
        if (!getValidDepthFromMap(depth_map, u, v, d_hat_i_inv)) {
            continue;
        }
        
        // 遍历该特征在后续帧的观测
        for (size_t frame_idx = 1; frame_idx < feature.feature_per_frame.size(); ++frame_idx) {
            const FeaturePerFrame& obs_frame_k = feature.feature_per_frame[frame_idx];
            int frame_k_window_index = obs_frame_k.frame_id - window_start_frame_id;
            
            // 确保帧索引在有效范围内
            if (frame_k_window_index <= 0 || 
                frame_k_window_index > WINDOW_SIZE ||
                frame_k_window_index >= static_cast<int>(pre_integrations_compound.size())) {
                continue;
            }
            
            Eigen::Vector3d z_ik = obs_frame_k.point;
            if (!isValidNormalizedCoord(z_ik)) {
                continue;
            }
            
            // 构建观测数据
            ObservationData obs_data;
            obs_data.feature_id = feature.feature_id;
            obs_data.frame_k_index = frame_k_window_index;
            obs_data.pre_integration_k = pre_integrations_compound[frame_k_window_index];
            obs_data.z_i0 = z_i0;
            obs_data.z_ik = z_ik;
            obs_data.d_hat_i = d_hat_i_inv;
            
            observations_out.push_back(obs_data);
        }
    }
    
    return feature_count;
}

bool FastInitializer::isValidSolution(const Eigen::Matrix<double, 8, 1>& x)
{
    double a = x(0);
    double b = x(1);
    Eigen::Vector3d g_in_I0 = x.segment<3>(5);
    double g_norm = g_in_I0.norm();
    
    // 检查尺度因子 a 的合理性
    if (a < 1e-3 || a > 100.0) {
        ROS_WARN("FastInit: Invalid scale factor a=%.6f (should be in [1e-3, 100]).", a);
        return false;
    }
    
    // 检查重力大小的合理性（地球表面重力加速度范围）
    if (g_norm < 5.0 || g_norm > 15.0) {
        ROS_WARN("FastInit: Invalid gravity norm |g|=%.2f (should be in [5, 15] m/s²).", g_norm);
        return false;
    }
    
    // 检查深度偏移 b 的合理性（可选：严格模式下可拒绝）
    const double MAX_B_ABS = 1.0;
    if (std::abs(b) > MAX_B_ABS) {
        ROS_WARN("FastInit: Large depth offset b=%.6f (|b| > %.1f). "
                 "This may indicate poor depth model quality.", b, MAX_B_ABS);
        // 注意：这里只警告，不直接拒绝，因为某些场景下可能合理
    }
    
    return true;
}

bool FastInitializer::computeDepthStatistics(
    const cv::Mat& depth_map,
    double a, double b,
    int window_start_frame_id,
    DepthStatistics& stats_out)
{
    stats_out = DepthStatistics();
    int rows = depth_map.rows;
    int cols = depth_map.cols;
    
    for (const auto& feature : m_feature_manager->feature) {
        if (feature.start_frame != window_start_frame_id || 
            feature.feature_per_frame.empty()) {
            continue;
        }
        
        const FeaturePerFrame& obs0 = feature.feature_per_frame[0];
        int u = static_cast<int>(std::round(obs0.uv.x()));
        int v = static_cast<int>(std::round(obs0.uv.y()));
        
        double d_hat_inv;
        if (!getValidDepthFromMap(depth_map, u, v, d_hat_inv)) {
            continue;
        }
        
        // 计算深度：z = a * (1 / d_hat_inv) + b
        double z = a * (1.0 / d_hat_inv) + b;
        
        stats_out.total_count++;
        stats_out.min_depth = std::min(stats_out.min_depth, z);
        stats_out.max_depth = std::max(stats_out.max_depth, z);
        
        // 统计有效深度（在合理范围内：0.1 ~ 50 米）
        if (z > 0.1 && z < 50.0) {
            stats_out.valid_count++;
        }
    }
    
    // 计算平均深度（只计算有效深度的平均）
    double z_sum = 0.0;
    int valid_sum_count = 0;
    for (const auto& feature : m_feature_manager->feature) {
        if (feature.start_frame != window_start_frame_id || 
            feature.feature_per_frame.empty()) {
            continue;
        }
        
        const FeaturePerFrame& obs0 = feature.feature_per_frame[0];
        int u = static_cast<int>(std::round(obs0.uv.x()));
        int v = static_cast<int>(std::round(obs0.uv.y()));
        
        double d_hat_inv;
        if (!getValidDepthFromMap(depth_map, u, v, d_hat_inv)) {
            continue;
        }
        
        double z = a * (1.0 / d_hat_inv) + b;
        if (z > 0.1 && z < 50.0) {
            z_sum += z;
            valid_sum_count++;
        }
    }
    
    stats_out.mean_depth = (valid_sum_count > 0) ? (z_sum / valid_sum_count) : 0.0;
    
    return true;
}

void FastInitializer::alignCoordinateSystem(
    const Eigen::Vector3d& g_in_I0,
    const Eigen::Vector3d& v_I0_in_I0,
    Eigen::Vector3d& G_gravity_world,
    Eigen::Quaterniond& R_I0_to_W_prime,
    Eigen::Vector3d& v_I0_in_W_prime)
{
    // 计算目标重力方向（使用全局参数 G，通常是 [0, 0, 9.81]）
    Eigen::Vector3d g_target_normalized = ::G.normalized();
    Eigen::Vector3d g_I0_normalized = g_in_I0.normalized();
    
    // 计算从 W' 到 I0 的旋转（将目标重力方向旋转到估计的重力方向）
    // R_W'_I0 = FromTwoVectors(g_target, g_I0)
    Eigen::Quaterniond R_W_prime_to_I0 = 
        Eigen::Quaterniond::FromTwoVectors(g_target_normalized, g_I0_normalized);
    
    // 更新全局重力向量：保留目标方向，使用估计的大小
    G_gravity_world = g_target_normalized * g_in_I0.norm();
    
    // 计算从 I0 到 W' 的旋转（用于变换速度等）
    R_I0_to_W_prime = R_W_prime_to_I0.inverse();
    
    // 消除第一帧在 W' 系下的 yaw 角（使其与重力对齐，但不固定 yaw）
    Matrix3d R0_matrix = R_I0_to_W_prime.toRotationMatrix();
    double yaw_first = Utility::R2ypr(R0_matrix).x();
    Matrix3d R_yaw_correction = Utility::ypr2R(Eigen::Vector3d{-yaw_first, 0, 0});
    R0_matrix = R_yaw_correction * R0_matrix;
    R_I0_to_W_prime = Quaterniond(R0_matrix);
    
    // 将速度从 I0 系转换到 W' 系
    v_I0_in_W_prime = R_I0_to_W_prime * v_I0_in_I0;
}

bool FastInitializer::propagateStatesToAllFrames(
    const std::map<double, ImageFrame>& image_frames,
    const Eigen::Quaterniond& R_I0_to_W_prime,
    const Eigen::Vector3d& p_I0_in_W_prime,
    const Eigen::Vector3d& v_I0_in_W_prime,
    const Eigen::Vector3d& G_gravity_world,
    std::map<int, Eigen::Vector3d>& Ps_out,
    std::map<int, Eigen::Vector3d>& Vs_out,
    std::map<int, Eigen::Quaterniond>& Rs_out)
{
    Ps_out.clear();
    Vs_out.clear();
    Rs_out.clear();
    
    // 初始化第一帧的状态
    Eigen::Quaterniond R_prev = R_I0_to_W_prime;
    Eigen::Vector3d p_prev = p_I0_in_W_prime;
    Eigen::Vector3d v_prev = v_I0_in_W_prime;
    
    int k_index = 0;
    for (const auto& pair : image_frames) {
        if (k_index == 0) {
            // 第一帧：直接存储初始状态
            Rs_out[k_index] = R_prev;
            Ps_out[k_index] = p_prev;
            Vs_out[k_index] = v_prev;
        } else {
            // 后续帧：使用 IMU 预积分进行状态传播
            IntegrationBase* pre_int_delta = pair.second.pre_integration;
            if (!pre_int_delta) {
                ROS_ERROR("FastInit: Missing pre_integration for frame %d!", k_index);
                return false;
            }
            
            double dt = pre_int_delta->sum_dt;
            
            // 旋转传播：R_k^w = R_{k-1}^w * delta_R_{k-1}^k
            Eigen::Quaterniond R_curr = R_prev * pre_int_delta->delta_q;
            
            // 位置传播：p_k^w = p_{k-1}^w + v_{k-1}^w * dt - 0.5*g*dt² + R_{k-1}^w * delta_p
            Eigen::Vector3d p_curr = p_prev + v_prev * dt 
                                    - 0.5 * G_gravity_world * dt * dt
                                    + R_prev * pre_int_delta->delta_p;
            
            // 速度传播：v_k^w = v_{k-1}^w - g*dt + R_{k-1}^w * delta_v
            Eigen::Vector3d v_curr = v_prev - G_gravity_world * dt
                                    + R_prev * pre_int_delta->delta_v;
            
            // 存储当前帧状态
            Rs_out[k_index] = R_curr;
            Ps_out[k_index] = p_curr;
            Vs_out[k_index] = v_curr;
            
            // 更新为下一帧的"前一帧"状态
            R_prev = R_curr;
            p_prev = p_curr;
            v_prev = v_curr;
        }
        k_index++;
    }
    
    return true;
}