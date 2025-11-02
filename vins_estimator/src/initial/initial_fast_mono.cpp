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
    int window_start_frame_id = 0;
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
    return window_start_frame_id;
}

// 主初始化函数
bool FastInitializer::initialize(const std::map<double, ImageFrame>& image_frames,
                               const cv::Mat& first_frame_norm_inv_depth, // CV_32F, [1,2]
                               Eigen::Vector3d& G_gravity_world, // Global gravity (e.g., [0, 0, 9.8])
                               std::map<int, Eigen::Vector3d>& Ps_out,
                               std::map<int, Eigen::Vector3d>& Vs_out,
                               std::map<int, Eigen::Quaterniond>& Rs_out)
{
    // --- 1. 数据收集 ---
    ROS_INFO("FastInit: Collecting observations...");
    std::vector<ObservationData> all_observations;
    std::vector<IntegrationBase*> pre_integrations_compound; 

    // 1.1. 复合 IMU 预积分

    IntegrationBase* current_pre_integration = new IntegrationBase(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                                                   Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    pre_integrations_compound.push_back(current_pre_integration);

    int k_index = 0; // 帧在窗口内的索引
    for (const auto& pair : image_frames)
    {
        if (k_index > 0) // 跳过第一帧
        {
            IntegrationBase* prev_compound = pre_integrations_compound.back();
            IntegrationBase* delta_integration = pair.second.pre_integration;
            if (!delta_integration) {
                ROS_ERROR("FastInit: Missing pre_integration for frame %d!", k_index);
                // 清理已分配的内存
                for(IntegrationBase* p : pre_integrations_compound) delete p;
                return false;
            }

            // 复合 I0->Ik 的预积分（i=0, j=k-1, k）
            IntegrationBase* new_compound = new IntegrationBase(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                                                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            new_compound->sum_dt  = prev_compound->sum_dt + delta_integration->sum_dt;
            new_compound->delta_q = prev_compound->delta_q * delta_integration->delta_q;
            new_compound->delta_v = prev_compound->delta_v + prev_compound->delta_q * delta_integration->delta_v;
            new_compound->delta_p = prev_compound->delta_p
                                    + prev_compound->delta_v * delta_integration->sum_dt
                                    + prev_compound->delta_q * delta_integration->delta_p;

            pre_integrations_compound.push_back(new_compound);
        }
        k_index++;
    }

    // 1.2. 收集特征观测和深度
    int first_frame_rows = first_frame_norm_inv_depth.rows;
    int first_frame_cols = first_frame_norm_inv_depth.cols;

    // 统一计算窗口起始帧ID
    int window_start_frame_id = computeWindowStartFrameId();

    for (const auto& feature : m_feature_manager->feature)
    {
        // 确保特征在第一帧被观测到
        if (feature.start_frame == window_start_frame_id && feature.feature_per_frame.size() > 0)
        {
            int feature_id = feature.feature_id;
            const FeaturePerFrame& obs_frame_0 = feature.feature_per_frame[0];

            // 获取归一化坐标 z_i0
            Eigen::Vector3d z_i0 = obs_frame_0.point; // VINS-Mono 已存储为 [x, y, 1]

            // 获取归一化逆深度 d_hat_i
            // 这里的 uv 是原始图像坐标，且 first_frame_norm_inv_depth
            // 已经被 resize 到了原始图像尺寸 (在 DepthEstimator::predict 中完成)
            int u = static_cast<int>(round(obs_frame_0.uv.x()));
            int v = static_cast<int>(round(obs_frame_0.uv.y()));

            if (v >= 0 && v < first_frame_rows && u >= 0 && u < first_frame_cols)
            {
                double d_hat_i_inv = static_cast<double>(first_frame_norm_inv_depth.at<float>(v, u));

                // 遍历该特征在后续帧 k > 0 的观测
                for (size_t frame_idx = 1; frame_idx < feature.feature_per_frame.size(); ++frame_idx)
                {
                    const FeaturePerFrame& obs_frame_k = feature.feature_per_frame[frame_idx];
                    // 将全局 frame_id 转换为窗口内的局部索引
                    int frame_k_window_index = obs_frame_k.frame_id - window_start_frame_id;

                    if (frame_k_window_index > 0 && frame_k_window_index <= WINDOW_SIZE)
                    {
                        Eigen::Vector3d z_ik = obs_frame_k.point;

                        ObservationData obs_data;
                        obs_data.feature_id = feature_id;
                        obs_data.frame_k_index = frame_k_window_index;
                        obs_data.pre_integration_k = pre_integrations_compound[frame_k_window_index]; // $\Delta_{I_0}^{I_k}$
                        obs_data.z_i0 = z_i0;
                        obs_data.z_ik = z_ik;
                        obs_data.d_hat_i = d_hat_i_inv;
                        all_observations.push_back(obs_data);
                    }
                }
            }
        }
    }
    ROS_INFO("FastInit: Collected %zu observations.", all_observations.size());

    // 检查是否有足够的观测来进行 RANSAC 最小集求解
    if (all_observations.size() < fast_mono::RANSAC_MIN_MEASUREMENTS)
    {
        ROS_WARN("FastInit: Not enough valid observations (%zu) to initialize.", all_observations.size());
        // 清理复合预积分对象
        for(IntegrationBase* p : pre_integrations_compound) delete p;
        return false;
    }

    // --- 2. RANSAC 求解 ---
    ROS_INFO("FastInit: Starting RANSAC...");
    Eigen::Matrix<double, 8, 1> x_best; // 最优解 [a, b, v_I0(3), g_I0(3)]^T
    bool ransac_success = solveRANSAC(all_observations, x_best);

    // 清理复合预积分对象 (不再需要)
    for(IntegrationBase* p : pre_integrations_compound) delete p;

    if (!ransac_success)
    {
        ROS_WARN("FastInit: RANSAC failed to find a valid solution.");
        return false;
    }

    // --- 3. 结果解析与状态生成 ---
    // 从 RANSAC 返回的最优解向量 x_best (8x1) 中提取各个分量。
    // x_best = [a, b, v_I0_x, v_I0_y, v_I0_z, g_I0_x, g_I0_y, g_I0_z]^T
    
    // 深度线性模型的尺度因子 a
    double a = x_best(0); 
    // 深度线性模型的偏移量 b
    double b = x_best(1); 
    // 第一帧 IMU 在 I0 坐标系下的速度向量
    Eigen::Vector3d v_I0_in_I0 = x_best.segment<3>(2); 
    // 重力向量在 I0 坐标系下的表示
    Eigen::Vector3d g_in_I0    = x_best.segment<3>(5);    

    // --- 3.1 结果有效性检查 ---
    // 对 RANSAC 解进行基本的物理合理性检查。
    // 检查尺度因子 a 是否过小 (接近 0)。如果 a 太小，会导致深度 z = a/d_hat + b对 d_hat 的变化不敏感，或者深度值趋于无穷，这通常是不稳定或错误的解。
    // 1e-3 是一个经验阈值。
    // 检查估计出的重力向量模长 g_in_I0.norm() 是否在合理范围内 (例如 5 到 15 m/s^2)。
    // 远小于或远大于标准重力加速度 (约 9.8) 都表明解可能错误。
    if (a < 1e-3 || a > 100.0 || g_in_I0.norm() < 5.0 || g_in_I0.norm() > 15.0)
    {
        ROS_WARN("FastInit: RANSAC solution seems unreasonable (a=%.6f must be in [1e-3, 100], |g|=%.2f). Aborting.", 
                a, g_in_I0.norm());
        return false;
    }
    
    // b 是深度偏移，通常应该在合理范围内（不能太大，否则会导致深度异常）
    // 如果 b 的绝对值太大，说明模型估计有问题
    const double MAX_B_ABS = 1.0;  // b 的绝对值不应该超过1米
    if (std::abs(b) > MAX_B_ABS) {
        ROS_WARN("FastInit: Offset b=%.6f is too large (|b| > %.1f). This may indicate poor depth model quality.", 
                b, MAX_B_ABS);
        // 可以选择拒绝或警告
        // return false;  // 严格模式：直接拒绝
    }

    // --- 4. 计算特征点深度 (在 C0 系) ---
    // 利用 RANSAC 估计出的 a 和 b，为所有在第一帧 (C0) 观测到的特征点计算初始深度。
    // 这些深度值将被用于后续的非线性优化 (BA)。
    
    int N_total = 0, N_ok = 0;
    double z_min = std::numeric_limits<double>::infinity();
    double z_max = -std::numeric_limits<double>::infinity();
    double z_sum = 0.0;

    window_start_frame_id = computeWindowStartFrameId();

    for (const auto& feature : m_feature_manager->feature)
    {
        if (feature.start_frame == window_start_frame_id && !feature.feature_per_frame.empty())
        {
            const FeaturePerFrame& obs0 = feature.feature_per_frame[0];
            int u = static_cast<int>(std::round(obs0.uv.x()));
            int v = static_cast<int>(std::round(obs0.uv.y()));
            if (v >= 0 && v < first_frame_rows && u >= 0 && u < first_frame_cols)
            {
                float d_hat_inv = first_frame_norm_inv_depth.at<float>(v, u); // 在 [1,2]
                if (d_hat_inv > 1e-6f)
                {
                    double z = a * (1.0 / static_cast<double>(d_hat_inv)) + b; // 现有线性深度模型
                    N_total++;
                    z_min = std::min(z_min, z);
                    z_max = std::max(z_max, z);
                    z_sum += z;
                    if (z > 0.1 && z < 50.0) N_ok++;
                }
            }
        }
    }
    double z_mean = (N_total > 0) ? (z_sum / N_total) : 0.0;
    ROS_INFO("FastInit depth diag: a=%.6f b=%.6f | N_total=%d N_ok(0.1~50m)=%d | z[min=%.6f max=%.6f mean=%.6f]",
             a, b, N_total, N_ok, z_min, z_max, z_mean);

    if (N_ok < 10) {
        ROS_WARN("FastInit: Not enough valid depth features (%d) to initialize.", N_ok);
        return false;
    }
    
      
      // --- 5. 坐标系对齐 (从 I0 系 -> W' 重力对齐系) ---
    // 目标: 我们 RANSAC 求解得到的速度 v_I0 和重力 g_I0 都是在 I0 坐标系（即第一帧 IMU 坐标系）下表示的。
    //       为了方便后续优化和与其他模块（如可视化）对接，我们需要将所有状态统一到一个规范的世界坐标系 W' 中。
    //       我们选择 W' 的定义为：原点与 I0 重合，Z 轴与真实重力方向对齐（通常是竖直向上或向下）。
    
    // G_gravity_world (输入参数): 这是 VINS-Mono 系统定义的全局目标重力向量。
    // 它的大小由配置文件中的 g_norm 决定，方向通常是 [0, 0, +G.norm()] (VINS-Mono 约定)。
    // 我们用它来确定目标 W' 系的 Z 轴方向。
    
    // g_in_I0: RANSAC 估计出的重力向量在 I0 系下的表示。
    
    // 计算估计出的重力向量 g_in_I0 的单位方向向量。
    Eigen::Vector3d g_I0_normalized = g_in_I0.normalized();
    // --- MODIFICATION START: Use global G for target direction ---
    // 计算目标重力向量 ::G (来自 parameters.h) 的单位方向向量，以确定 W' 系的 Z 轴。
    // 不能使用传入的 G_gravity_world，因为它在此时是未初始化的。
    Eigen::Vector3d g_target_normalized = ::G.normalized(); 

    // 计算从 W' 系到 I0 系的旋转 R_W'_I0。
    // Eigen::Quaterniond::FromTwoVectors(a, b) 计算的是将向量 a 旋转到向量 b 所需的最小旋转。
    // 因此，FromTwoVectors(g_target_normalized, g_I0_normalized) 计算的是将 W' 系下的重力方向 ([0,0,1]) 
    // 旋转到我们在 I0 系下估计出的重力方向 g_I0_normalized 所需的旋转。
    // 这个旋转就是 R_W'_I0 (从 W' 旋转到 I0)。
    Eigen::Quaterniond R_W_prime_to_I0 = Eigen::Quaterniond::FromTwoVectors(g_target_normalized, g_I0_normalized);

    // 更新全局重力向量 G_gravity_world。
    // 我们保留 VINS 系统预设的重力 方向 (g_target_normalized)，
    // 但是使用我们 RANSAC 估计出的重力 大小 (g_in_I0.norm())。
    // 这允许系统自适应地估计当地的重力加速度大小。
    G_gravity_world = g_target_normalized * g_in_I0.norm();

    // 计算从 I0 系到 W' 系的旋转 R_I0_W'，即 R_W'_I0 的逆。
    // 这个旋转用于将 I0 系下的向量（如速度）转换到 W' 系下。
    Eigen::Quaterniond R_I0_to_W_prime = R_W_prime_to_I0.inverse();

    // 第一帧在I0坐标系下的姿态是单位旋转（第一帧就是I0）
    // 需要消除第一帧在W'坐标系下的yaw角
    Matrix3d R0_matrix = R_I0_to_W_prime.toRotationMatrix();
    double yaw_first = Utility::R2ypr(R0_matrix).x();  // 第一帧在W'系下的yaw角
    Matrix3d R_yaw_correction = Utility::ypr2R(Eigen::Vector3d{-yaw_first, 0, 0});
    R0_matrix = R_yaw_correction * R0_matrix;
    R_I0_to_W_prime = Quaterniond(R0_matrix);

    // 将第一帧的速度 v_I0_in_I0 从 I0 系变换到 W' 系。
    Eigen::Vector3d v_I0_in_W_prime = R_I0_to_W_prime * v_I0_in_I0;
    // 定义第一帧 IMU 在 W' 系下的位置为原点。
    Eigen::Vector3d p_I0_in_W_prime = Eigen::Vector3d::Zero(); 

    // --- 6. 前向传播状态到窗口内所有关键帧 ---
    // 目标: 我们已经有了第 0 帧 (I0) 在 W' 系下的姿态 R_I0_W' (即 R_wb_0), 速度 v_I0_in_W', 位置 p_I0_in_W'。
    // 现在，我们需要利用 IMU 预积分结果，将这个状态依次传播到窗口内的第 1, 2, ..., k 帧。
    
    // 清空输出的 map 容器，准备填充新的状态。
    Ps_out.clear();
    Vs_out.clear();
    Rs_out.clear();

    k_index = 0; // 窗口内帧的索引。
    
    // R_prev, p_prev, v_prev 存储的是上一帧 (k-1) 在 W 系下的状态。
    // 初始化为第 0 帧的状态。
    // 注意 VINS-Mono 的 Rs 数组存储的是从 Body 到 World 的旋转 R_{b}^{w}。
    // R_I0_to_W_prime 是从 I0 (Body) 到 W (World) 的旋转 R_{I0}^{w}。
    Eigen::Quaterniond R_prev = R_I0_to_W_prime; // R_wb_0
    Eigen::Vector3d p_prev = p_I0_in_W_prime;   // p_w_0
    Eigen::Vector3d v_prev = v_I0_in_W_prime;   // v_w_0

    // 遍历输入的所有图像帧信息 (image_frames 是按时间戳排序的 map)。
    for (const auto& pair : image_frames)
    {
        // --- 处理第 0 帧 ---
        if (k_index == 0)
        {
            // 直接存储第 0 帧的初始状态。
            Rs_out[k_index] = R_prev; // R_wb_0
            Ps_out[k_index] = p_prev; // p_w_0
            Vs_out[k_index] = v_prev; // v_w_0
        }
        // --- 处理第 k 帧 (k > 0) ---
        else
        {
            // 获取从 k-1 帧到 k 帧的 IMU 预积分结果 (delta_q, delta_p, delta_v)。
            // 注意：pair.second.pre_integration 存储的是 $\Delta_{I_{k-1}}^{I_k}$。
            IntegrationBase* pre_int_delta = pair.second.pre_integration;
            // 检查预积分对象是否存在 (理论上应该总是存在)。
            if (!pre_int_delta) {
                ROS_ERROR("FastInit: Missing pre_integration for state propagation at frame %d!", k_index);
                return false; 
            }
            // 获取 k-1 到 k 的时间差。
            double dt = pre_int_delta->sum_dt;

            // --- 状态传播公式 ---
            // R_{k}^{w} 表示I_{k}到w的旋转
            // R_{k}^{w} = R_{k-1}^{w} * delta_R_{k}^{k-1}
            Eigen::Quaterniond R_curr = R_prev * pre_int_delta->delta_q ;


            // p_k = p_{k-1} + v_{k-1} * dt - 0.5*g*dt^2 + R_{k-1}^{w} * delta_p
            // 所有向量都在 W' (World) 系下。
            // delta_p 是在 Body(k-1) 系下表示的位移。需要用 R_{k-1}^{w} (即 R_prev) 转换到世界系。
            Eigen::Vector3d p_curr = p_prev + v_prev * dt - 0.5 * G_gravity_world * dt * dt + R_prev * pre_int_delta->delta_p;

            // v_k = v_{k-1} - g * dt + R_{k-1}^{w} * delta_v
            // delta_v 是在 Body(k-1) 系下表示的速度变化。需要用 R_{k-1}^{w} (即 R_prev) 转换到世界系。
            Eigen::Vector3d v_curr = v_prev - G_gravity_world * dt + R_prev * pre_int_delta->delta_v;

            // 存储计算出的第 k 帧的状态。
            Rs_out[k_index] = R_curr;
            Ps_out[k_index] = p_curr;
            Vs_out[k_index] = v_curr;

            // 更新 R_prev, p_prev, v_prev 以供下一轮迭代使用。
            R_prev = R_curr;
            p_prev = p_curr;
            v_prev = v_curr;
        }
        k_index++; // 帧索引加一。
    } // 遍历所有帧结束。

    // --- 7. 输出最终信息并返回成功 ---
    // 打印成功信息和关键的估计结果。
    ROS_INFO("FastInit: Initialization successful!");
    ROS_INFO("  a=%.3f, b=%.3f", a, b);
    ROS_INFO("  v_I0_W' = [%.3f, %.3f, %.3f]", v_I0_in_W_prime.x(), v_I0_in_W_prime.y(), v_I0_in_W_prime.z());
    ROS_INFO("  g_W'    = [%.3f, %.3f, %.3f] (norm: %.3f)", G_gravity_world.x(), G_gravity_world.y(), G_gravity_world.z(), G_gravity_world.norm());

    // 返回 true 表示初始化成功。
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
        std::vector<int> current_indices;
        std::vector<ObservationData> minimal_set;
        while (minimal_set.size() < current_min_measurements)
        {
            int rand_idx = distribution(m_random_generator);
            bool found = false;
            for(int idx : current_indices) if(idx == rand_idx) found = true;
            if(!found) {
                if (!all_observations[rand_idx].pre_integration_k) continue; // 跳过无效数据
                current_indices.push_back(rand_idx);
                minimal_set.push_back(all_observations[rand_idx]);
            }
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
        const double CONDITION_NUMBER_THRESHOLD = 1e12; // << 需要调整
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
    Eigen::Matrix<double, 8, 8> S = Eigen::Matrix<double, 8, 8>::Identity();
    S(0, 0) = 0.1;   // a: 深度尺度因子，通常在0.01-1范围内，缩放到合理尺度
    S(1, 1) = 1.0;   // b: 深度偏移，通常较小，保持原尺度
    S(2, 2) = 10.0; S(3, 3) = 10.0; S(4, 4) = 10.0; // v (m/s)
    S(5, 5) = 100.0; S(6, 6) = 100.0; S(7, 7) = 100.0; // g (m/s²)
    Eigen::MatrixXd A_S = A * S; // 缩放后的矩阵

    // --- 3. 求解无约束最小二乘问题 (SVD) ---
    // **关键**: A_S 是 N x 8 (N >= 8)，使用 ThinU/V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_unconstrained(A_S, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // 检查 SVD 是否成功以及条件数
    double cond_num_unconstrained = std::numeric_limits<double>::infinity();
    const double min_singular_value_threshold = 1e-8;
    if (svd_unconstrained.singularValues().size() > 0 && svd_unconstrained.singularValues().minCoeff() > min_singular_value_threshold) {
        cond_num_unconstrained = svd_unconstrained.singularValues().maxCoeff() / svd_unconstrained.singularValues().minCoeff();
    }
    ROS_INFO("solveLinearSystem (Unconstrained): Condition number = %.2e", cond_num_unconstrained);

    // 可以选择性地基于条件数判断是否继续
    // const double FINAL_FIT_COND_THRESHOLD = 1e12; // 设定一个最终拟合的条件数阈值
    // if (cond_num_unconstrained > FINAL_FIT_COND_THRESHOLD) {
    //    ROS_WARN("solveLinearSystem Warning: Poor condition number (%.2e) in final unconstrained fit. Result may be unreliable.", cond_num_unconstrained);
    //    // return false; // 或者继续尝试施加约束
    // }

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
    if (g_norm_svd < 1.0) { // 如果无约束解的重力接近零，说明数据可能有问题
        ROS_WARN("solveLinearSystem Warning: Unconstrained solve resulted in near-zero gravity (%.3f). Using this result without constraint.", g_norm_svd);
        // 在这种情况下，强行施加约束可能更糟，直接返回无约束解
        x_out = x_svd;
        // 仍然需要检查这个解是否合理
         if (std::abs(x_out(0)) < 1e-4 || x_out.segment<3>(5).norm() < 1.0) { // 如果 a 或 g 仍然无效
              ROS_ERROR("solveLinearSystem Error: Unconstrained solution is also unreasonable. Failed.");
              return false;
         }
        return true; // 返回无约束但（可能）合理的解
    }

    // 使用无约束解的方向 和 全局参数 G.norm() 确定约束后的重力
    Eigen::Vector3d g_normed = g_svd.normalized() * G.norm();
    cout << "g_svd: " << g_svd.transpose() << endl;
    cout << "g_svd.norm(): " << g_svd.norm() << endl;

    // --- 5. 固定 g_normed，重新求解 y = [a, b, v_I0]^T ---
    Eigen::MatrixXd A_y = A.leftCols<5>(); // 前 5 列
    Eigen::MatrixXd A_g = A.rightCols<3>(); // 后 3 列
    Eigen::VectorXd b_y = b - A_g * g_normed; // 新的 RHS

    // --- 6. 应用数值预处理 (y 部分) ---
    Eigen::Matrix<double, 5, 5> S_y = S.block<5, 5>(0, 0); // S 的左上 5x5
    Eigen::MatrixXd Ay_Sy = A_y * S_y;

    // --- 7. SVD 求解 A_y * y = b_y ---
    // **关键**: Ay_Sy 是 N x 5 (N >= 10)，使用 ThinU/V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_constrained(Ay_Sy, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // 检查 SVD 是否成功以及条件数
    double cond_num_constrained = std::numeric_limits<double>::infinity();
     if (svd_constrained.singularValues().size() > 0 && svd_constrained.singularValues().minCoeff() > min_singular_value_threshold) {
        cond_num_constrained = svd_constrained.singularValues().maxCoeff() / svd_constrained.singularValues().minCoeff();
     }
    ROS_INFO("solveLinearSystem (Constrained Fit for y): Condition number = %.2e", cond_num_constrained);

    Eigen::Matrix<double, 5, 1> y_scaled = svd_constrained.solve(b_y);
     if (y_scaled.hasNaN()) {
         ROS_ERROR("solveLinearSystem Error: Constrained SVD solve for y resulted in NaN!");
         // 尝试回退到无约束解
         ROS_WARN("Falling back to the unconstrained solution.");
         x_out = x_svd;
         // 再次检查回退解的有效性
         if (std::abs(x_out(0)) < 1e-4 || x_out.segment<3>(5).norm() < 5.0) {
              ROS_ERROR("solveLinearSystem Error: Fallback unconstrained solution is also unreasonable. Failed.");
              return false;
         }
         return true;
     }
    Eigen::Matrix<double, 5, 1> y = S_y * y_scaled; // 恢复原始尺度

    // --- 8. 组合最终解 ---
    x_out.head<5>() = y;          // a, b, v_I0
    x_out.tail<3>() = g_normed;   // 约束后的 g

    ROS_INFO("solveLinearSystem finished successfully.");
    return true;
}