#include "initial_fast_mono.h"

// 构造函数
FastInitializer::FastInitializer(FeatureManager* f_manager_ptr)
    : m_feature_manager(f_manager_ptr),
      m_random_generator(std::random_device{}()) // 初始化随机数生成器
{
}

// 主初始化函数
bool FastInitializer::initialize(const std::map<double, ImageFrame>& image_frames,
                               const cv::Mat& first_frame_norm_inv_depth, // CV_32F, [0,1]
                               Eigen::Vector3d& G_gravity_world, // Global gravity (e.g., [0, 0, 9.8])
                               std::map<int, Eigen::Vector3d>& Ps_out,
                               std::map<int, Eigen::Vector3d>& Vs_out,
                               std::map<int, Eigen::Quaterniond>& Rs_out)
{
    // --- 1. 数据收集 ---
    ROS_INFO("FastInit: Collecting observations...");
    std::vector<ObservationData> all_observations;
    std::vector<IntegrationBase*> pre_integrations_compound; // 存储 $\Delta_{I_0}^{I_k}$

    // 1.1. 复合 IMU 预积分: VINS-Mono 的 all_image_frame[k].pre_integration 是 $\Delta_{I_{k-1}}^{I_k}$
    // --- MODIFICATION START: Use parameterized constructor ---
    IntegrationBase* current_pre_integration = new IntegrationBase(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                                                   Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    // --- MODIFICATION END ---
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

            // --- MODIFICATION START: Use parameterized constructor ---
            IntegrationBase* new_compound = new IntegrationBase(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                                                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            // --- MODIFICATION END ---
            new_compound->sum_dt = prev_compound->sum_dt + delta_integration->sum_dt;
            new_compound->delta_q = prev_compound->delta_q * delta_integration->delta_q;
            new_compound->delta_p = prev_compound->delta_p + prev_compound->delta_q * delta_integration->delta_p;
            new_compound->delta_v = prev_compound->delta_v + prev_compound->delta_q * delta_integration->delta_v;
            // 注意: 雅可比和协方差的复合在这里不是必需的

            pre_integrations_compound.push_back(new_compound);
        }
        k_index++;
    }

    // 1.2. 收集特征观测和深度
    int first_frame_rows = first_frame_norm_inv_depth.rows;
    int first_frame_cols = first_frame_norm_inv_depth.cols;

    // --- 新增: 找到当前滑动窗口的起始帧ID ---
    int window_start_frame_id = -1;
    if (!m_feature_manager->feature.empty()) {
        window_start_frame_id = m_feature_manager->feature.front().start_frame;
    }
    for (const auto& feature : m_feature_manager->feature) {
        if (feature.start_frame < window_start_frame_id) {
            window_start_frame_id = feature.start_frame;
        }
    }
    if (window_start_frame_id < 0) window_start_frame_id = 0; // 安全保护

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
    
    // 检查尺度因子 a 是否过小 (接近 0)。如果 a 太小，会导致深度 z = 1 / (a*d_hat + b)
    // 对 d_hat 的变化不敏感，或者深度值趋于无穷，这通常是不稳定或错误的解。
    // 1e-3 是一个经验阈值。
    // 检查估计出的重力向量模长 g_in_I0.norm() 是否在合理范围内 (例如 5 到 15 m/s^2)。
    // 远小于或远大于标准重力加速度 (约 9.8) 都表明解可能错误。
    if (std::abs(a) < 1e-3 || g_in_I0.norm() < 5.0 || g_in_I0.norm() > 15.0)
    {
        // 如果检查不通过，打印警告信息并返回 false，表示初始化失败。
        ROS_WARN("FastInit: RANSAC solution seems unreasonable (a=%.2f, |g|=%.2f). Aborting.", a, g_in_I0.norm());
        return false;
    }

    // --- 4. 计算特征点深度 (在 C0 系) ---
    // 利用 RANSAC 估计出的 a 和 b，为所有在第一帧 (C0) 观测到的特征点计算初始深度。
    // 这些深度值将被用于后续的非线性优化 (BA)。
    
    int features_initialized = 0; // 计数成功初始化深度的特征数量。

    // 遍历 FeatureManager 中存储的所有特征点。
    // 注意：这里的 `m_feature_manager->feature` 是一个 map 或 unordered_map。
    // C++11/17 的 range-based for loop 写法。`auto& feature_pair` 效率更高。
    // 为了清晰，这里展开写：
    for (auto& feature : m_feature_manager->feature) // feature_iterator 是指向 map 中元素的迭代器
    {
        // 检查该特征是否是从第一帧开始被跟踪的。
        // !! 注意 !!: `feature.start_frame` 在 VINS-Mono 中是全局帧 ID。
        // 如果您的窗口是从全局第 N 帧开始的，这里的判断需要修改为
        // `feature.start_frame == window_start_frame_id` (您需要传入这个起始 ID)。
        // 假设当前窗口就是从全局第 0 帧开始的，所以判断 `feature.start_frame == 0`。
        // 同时确保该特征至少有一帧观测数据。
        if (feature.start_frame == window_start_frame_id && feature.feature_per_frame.size() > 0)
        {
            // 获取该特征在第一帧 (索引 0) 的观测信息。
            const FeaturePerFrame& obs_frame_0 = feature.feature_per_frame[0];
            
            // 获取该特征在第一帧的像素坐标 (u, v)，并进行四舍五入取整。
            int u = static_cast<int>(round(obs_frame_0.uv.x()));
            int v = static_cast<int>(round(obs_frame_0.uv.y()));

            // 检查 (u, v) 坐标是否在深度图的有效范围内。
            if (v >= 0 && v < first_frame_rows && u >= 0 && u < first_frame_cols)
            {
                // 从输入的归一化逆深度图 (CV_32F, 范围 [0,1]) 中获取对应的预测值 d_hat_i。
                double d_hat_i = static_cast<double>(first_frame_norm_inv_depth.at<float>(v, u));
                
                // 检查预测的逆深度值是否有效 (例如大于一个很小的正数，避免除零)。
                if (d_hat_i > 1e-6)
                {
                    // 应用核心公式: 1/z = a * d_hat + b，计算估计的逆深度。
                    double estimated_inv_depth = a * d_hat_i + b;
                    
                    // 检查估计的逆深度是否为正数 (因为深度 z 必须为正)。
                    if (estimated_inv_depth > 1e-6)
                    {
                        // 计算最终的深度值 z = 1 / estimated_inv_depth。
                        double estimated_depth = 1.0 / estimated_inv_depth;
                        
                        // 对计算出的深度值进行范围检查，剔除过近或过远的点。
                        // 0.1m 到 50m 是一个常用的经验范围。
                        if (estimated_depth > 0.1 && estimated_depth < 50.0)
                        {
                            // 如果深度值有效，则调用 FeatureManager 的函数来存储这个深度值。
                            // 这个深度值将在后续的非线性优化 (solveOdometry) 中被用作初始值。
                            m_feature_manager->setFeatureDepth(feature.feature_id, estimated_depth);
                            // 成功初始化一个特征点的深度，计数器加一。
                            features_initialized++;
                        }
                    }
                }
            }
        }
    } // 遍历所有特征点结束。
    
    // 打印成功初始化深度的特征数量。
    ROS_INFO("FastInit: Initialized depth for %d features.", features_initialized);
    
    // 检查成功初始化深度的特征数量是否足够。
    // 如果太少 (例如少于 10 个)，后续的非线性优化 (BA) 可能会因为约束不足而失败或不稳定。
    if (features_initialized < 10) {
        ROS_WARN("FastInit: Too few features (%d) were successfully initialized with depth. Aborting.", features_initialized);
        return false; // 返回失败，表示初始化不成功。
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

    // 将第一帧的速度 v_I0_in_I0 从 I0 系变换到 W' 系。
    Eigen::Vector3d v_I0_in_W_prime = R_I0_to_W_prime * v_I0_in_I0;
    // 定义第一帧 IMU 在 W' 系下的位置为原点。
    Eigen::Vector3d p_I0_in_W_prime = Eigen::Vector3d::Zero(); 

    // --- 6. 前向传播状态到窗口内所有关键帧 ---
    // 目标: 我们已经有了第 0 帧 (I0) 在 W' 系下的姿态 R_I0_W' (即 R_wb_0), 速度 v_I0_in_W', 位置 p_I0_in_W'。
    //       现在，我们需要利用 IMU 预积分结果，将这个状态依次传播到窗口内的第 1, 2, ..., k 帧。
    
    // 清空输出的 map 容器，准备填充新的状态。
    Ps_out.clear();
    Vs_out.clear();
    Rs_out.clear();

    k_index = 0; // 窗口内帧的索引。
    
    // R_prev, p_prev, v_prev 存储的是上一帧 (k-1) 在 W' 系下的状态。
    // 初始化为第 0 帧的状态。
    // 注意 VINS-Mono 的 Rs 数组存储的是从 World 到 Body 的旋转 R_wb。
    // R_I0_to_W_prime 是从 I0 (Body) 到 W' (World) 的旋转 R_wi。
    // 所以，第一帧的 R_wb_0 应该是 R_I0_to_W_prime 的逆，即 R_W_prime_to_I0。
    // 这里可能需要根据 VINS-Mono 的 Rs 约定调整 R_prev 的初始值
    // 假设 VINS Rs 就是 R_wb:
    Eigen::Quaterniond R_prev = R_W_prime_to_I0; // R_wb_0
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
            // R_{w}^{k} 表示w到I_{k-1}的旋转
            // R_{w}^{k} = delta_R_{k-1}^{k} * R_{w}^{k-1}
            Eigen::Quaterniond R_curr = pre_int_delta->delta_q * R_prev;


            // p_k = p_{k-1} + v_{k-1} * dt - 0.5*g*dt^2 + R_{w}^{k-1} * delta_p
            // 所有向量都在 W' (World) 系下。
            // delta_p 是在 Body(k-1) 系下表示的位移。需要用 R_bw_k-1 (即 R_prev 的逆) 转换到世界系。
            Eigen::Vector3d p_curr = p_prev + v_prev * dt - 0.5 * G_gravity_world * dt * dt + R_prev.inverse() * pre_int_delta->delta_p;

            // v_k = v_{k-1} - g * dt + R_{w}^{k-1} * delta_v
            // delta_v 是在 Body(k-1) 系下表示的速度变化。需要用 R_bw_k-1 (即 R_prev 的逆) 转换到世界系。
            Eigen::Vector3d v_curr = v_prev - G_gravity_world * dt + R_prev.inverse() * pre_int_delta->delta_v;

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
    // --- 1. 提取外参 ---
    // R_c_i (RIC[0]) = {}_{C}^{I}R (从 Camera 到 IMU 的旋转)
    // T_c_i (TIC[0]) = {}^{I}p_C (Camera 在 IMU 系下的平移)
    const Eigen::Matrix3d& R_c_i = RIC[0];
    const Eigen::Vector3d& T_c_i_in_I = TIC[0];

    // R_i_c = R_c_i^T = {}_{I}^{C}R (从 IMU 到 Camera 的旋转)
    Eigen::Matrix3d R_i_c = R_c_i.transpose();
    // T_i_c = -R_i_c * T_c_i_in_I = {}^{C}p_I (IMU 在 Camera 系下的平移)
    Eigen::Vector3d T_i_c_in_C = -R_i_c * T_c_i_in_I;

    // --- 2. 提取 IMU 预积分信息 (I0 -> Ik) ---
    // R_I0_Ik = {}_{I_0}^{I_k}R
    Eigen::Matrix3d R_I0_Ik = pre_int_k->delta_q.toRotationMatrix();
    // I0_alpha_Ik = {}^{I_0}\alpha_{I_k} (VINS-Mono 的 delta_p 对应论文的 alpha)
    Eigen::Vector3d I0_alpha_Ik = pre_int_k->delta_p;
    // delta_t = \Delta T_k
    double delta_t = pre_int_k->sum_dt;
    double delta_t_sq = delta_t * delta_t;

    // --- 3. 计算论文中的关键中间变量 ---

    // Upsilon_i,k = Gamma_i,k * {}_{I}^{C}R * {}_{I_0}^{I_k}R
    // 
    Eigen::Matrix3d z_ik_hat = Utility::skewSymmetric(z_ik); // Gamma_i,k
    Eigen::Matrix3d Upsilon_3x3 = z_ik_hat * R_i_c * R_I0_Ik; // 注意：这里 R_i_c 是 {}_{I}^{C}R

    // d_i = 1 / d_hat_i (因为 d_hat_i 是逆深度)
    // [cite: 183-184]
    double d_i = 1.0 / d_hat_i; // d_hat_i 保证在 [1, 2] 范围，不会除零

    // {}^{I_0}\theta_{C_0 \to f_i} = {}_{C_0}^{I_0}R * z_i0
    // [cite: 169-171]
    // {}_{C_0}^{I_0}R (从 C0 到 I0) 就是外参 R_c_i
    Eigen::Vector3d I0_theta_C0_fi = R_c_i * z_i0;

    // --- 4. 构建 A' 矩阵的系数 [M1, M2, Tv, Tg] ---
    // x' = [a, b, v_I0(3), g_I0(3)]^T

    // M1 = Upsilon * d_i * I0_theta_C0_fi
    Eigen::Vector3d M1_3d = Upsilon_3x3 * d_i * I0_theta_C0_fi;

    // M2 = Upsilon * I0_theta_C0_fi
    Eigen::Vector3d M2_3d = Upsilon_3x3 * I0_theta_C0_fi;

    // Tv = -Upsilon * \Delta T_k
    // 
    Eigen::Matrix<double, 3, 3> Tv_3d = -Upsilon_3x3 * delta_t;

    // Tg = Upsilon * (0.5 * \Delta T_k^2)
    // 
    Eigen::Matrix<double, 3, 3> Tg_3d = Upsilon_3x3 * (0.5 * delta_t_sq);

    // --- 5. 构建 b' 向量 (RHS) ---
    // b' = Upsilon * I0_alpha_Ik - Upsilon * {}^{I}p_C - Gamma_i,k * {}^{C}p_I
    // 
    Eigen::Vector3d b_prime_3d = Upsilon_3x3 * I0_alpha_Ik - Upsilon_3x3 * T_c_i_in_I - z_ik_hat * T_i_c_in_C;

    // --- 6. 填充 A_row (2x8) 和 b_row (2x1) ---
    // 取 3D 向量/矩阵的前两行

    A_row.block<2, 1>(0, 0) = M1_3d.head<2>();      // M1 (a)
    A_row.block<2, 1>(0, 1) = M2_3d.head<2>();      // M2 (b)
    A_row.block<2, 3>(0, 2) = Tv_3d.block<2, 3>(0, 0); // Tv (v_I0)
    A_row.block<2, 3>(0, 5) = Tg_3d.block<2, 3>(0, 0); // Tg (g_I0)

    b_row = b_prime_3d.head<2>(); // b'

    // 函数结束时，A_row 和 b_row 就包含了由这一个 (i, k) 观测所贡献的两个线性方程。
    // RANSAC 和最终求解函数会调用这个函数多次，将得到的 A_row 和 b_row 堆叠起来，
    // 构成最终需要求解的 (超定) 线性系统 A'x' = b'。
}


// RANSAC 求解器
bool FastInitializer::solveRANSAC(const std::vector<ObservationData>& all_observations,
                                Eigen::Matrix<double, 8, 1>& best_x)
{   
    // 获取总观测数量 (N)。每个观测对应一对 (特征点 i, 图像帧 k)
    int num_observations = all_observations.size();
    // 用于存储迄今为止找到的最佳模型所对应的内点的索引
    std::vector<int> best_inlier_indices;
    // RANSAC 通常只关心内点的数量，而不是残差总和。这行可以忽略
    // double min_residual_sum = std::numeric_limits<double>::max(); // Or just track max inliers

    // 创建一个均匀分布的随机数生成器，用于从 [0, N-1] 的范围内随机抽取观测的索引
    std::uniform_int_distribution<int> distribution(0, num_observations - 1);

    for (int iter = 0; iter < fast_mono::RANSAC_MAX_ITERATIONS; ++iter)
    {
        // 1. 随机采样最小集 (4 对观测, 8 个方程)
        // 用于存储当前选中的观测索引
        std::vector<int> current_indices;
        // 用于存储实际选中的观测数据
        std::vector<ObservationData> minimal_set;

        // 循环直到选够最小集所需的观测数量。
        // 我们有 8 个未知数 (a, b, v_x, v_y, v_z, g_x, g_y, g_z)。
        // 每个 (i, k) 观测提供了 2 个线性方程 (来自 buildLinearSystemRow)。
        // 因此，我们需要 8 / 2 = 4 个观测来构成一个恰定方程组 (8x8)。
        while (minimal_set.size() < fast_mono::RANSAC_MIN_MEASUREMENTS) // RANSAC_MIN_MEASUREMENTS 通常为 4
        {
            // 从 [0, N-1] 中随机抽取一个索引。
            int rand_idx = distribution(m_random_generator);
            
            // 检查这个索引是否已经被选过了，以避免重复选择同一个观测。
            bool found = false;
            for(int idx : current_indices) if(idx == rand_idx) found = true;
            
            // 如果没选过，就加入当前索引列表和最小集数据列表。
            if(!found) {
                current_indices.push_back(rand_idx);
                minimal_set.push_back(all_observations[rand_idx]);
            }
        } // 循环结束后, minimal_set 中有 4 个随机选出的观测数据

        // 2. 求解最小集对应的线性系统 A_min * x = b_min
        // 构建用于求解最小系统的 8x8 矩阵 A_min 和 8x1 向量 b_min
        Eigen::Matrix<double, fast_mono::RANSAC_MIN_MEASUREMENTS * 2, 8> A_min;
        Eigen::Matrix<double, fast_mono::RANSAC_MIN_MEASUREMENTS * 2, 1> b_min;
        
        // 遍历最小集中的 4 个观测
        for (int i = 0; i < fast_mono::RANSAC_MIN_MEASUREMENTS; ++i)
        {   
            // --- 在此添加调试代码 ---
            const auto& obs = minimal_set[i];
            if (std::isnan(obs.d_hat_i) || std::isinf(obs.d_hat_i))
                ROS_ERROR("FastInit DEBUG: d_hat_i is NaN/Inf!");

            if (obs.pre_integration_k->delta_q.coeffs().hasNaN())
                ROS_ERROR("FastInit DEBUG: pre_int_k->delta_q is NaN!");

            if (obs.pre_integration_k->delta_p.hasNaN())
                ROS_ERROR("FastInit DEBUG: pre_int_k->delta_p is NaN!");

            if (std::isnan(obs.pre_integration_k->sum_dt) || obs.pre_integration_k->sum_dt <= 0)
                ROS_ERROR("FastInit DEBUG: sum_dt is invalid: %f", obs.pre_integration_k->sum_dt);

            // 为每个观测构建对应的 2x8 的 A_row 和 2x1 的 b_row
            Eigen::Matrix<double, 2, 8> A_row;
            Eigen::Vector2d b_row;
            buildLinearSystemRow(minimal_set[i].pre_integration_k, minimal_set[i].z_i0,
                                 minimal_set[i].z_ik, minimal_set[i].d_hat_i, A_row, b_row);
            
            // 将 A_row 和 b_row 填充到 A_min 和 b_min 的对应行
            A_min.block<2, 8>(i * 2, 0) = A_row;
            b_min.block<2, 1>(i * 2, 0) = b_row;
        }

        // ***** 新增：数值预处理 *****
        // 创建一个对角缩放矩阵 S
        // 我们保持 a, b 的尺度 O(1)
        // 我们将 v 的列 (col 2,3,4) 放大 10 倍 (O(1e-1) -> O(1))
        // 我们将 g 的列 (col 5,6,7) 放大 100 倍 (O(1e-2) -> O(1))
        Eigen::Matrix<double, 8, 8> S = Eigen::Matrix<double, 8, 8>::Identity();
        S(2, 2) = 10.0;
        S(3, 3) = 10.0;
        S(4, 4) = 10.0;
        S(5, 5) = 100.0;
        S(6, 6) = 100.0;
        S(7, 7) = 100.0;

        // 求解缩放后的系统: (A_min * S) * x_scaled = b_min
        // 其中 x_scaled = S_inv * x_candidate
        Eigen::Matrix<double, 8, 1> x_scaled = (A_min * S).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b_min);
        
        // 从 x_scaled 恢复回 x_candidate
        // x_candidate = S * x_scaled
        Eigen::Matrix<double, 8, 1> x_candidate = S * x_scaled;
        
        // ***** 结束新增代码 *****

        // --- MODIFICATION START: Remove ThinU/ThinV for fixed-size matrix ---
        // 使用 SVD 求解这个 8x8 的线性方程组 A_min * x = b_min。
        // 对于固定大小的方阵，不能使用 ComputeThinU 或 ComputeThinV 选项。
        // Eigen::Matrix<double, 8, 1> x_candidate = A_min.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b_min);
        // --- MODIFICATION END ---

        // ***** 在此添加调试打印 *****
        if (iter < 10) // 只打印前 10 次迭代
        {
            ROS_INFO("FastInit DEBUG (iter %d):", iter);
            ROS_INFO("  a=%.3f, b=%.3f", x_candidate(0), x_candidate(1));
            ROS_INFO("  v = [%.3f, %.3f, %.3f]", x_candidate(2), x_candidate(3), x_candidate(4));
            ROS_INFO("  g = [%.3f, %.3f, %.3f] (norm: %.3f)", 
                    x_candidate(5), x_candidate(6), x_candidate(7), x_candidate.segment<3>(5).norm());
            
            // (可选) 打印 A_min 和 b_min 的范数，看看它们是否有问题
            ROS_INFO("  A_min norm: %.3f, b_min norm: %.3f", A_min.norm(), b_min.norm());
        }
        // ***** 调试结束 *****

        // (可选) 检查解是否有效 (例如，重力幅度)
        double g_norm_cand = x_candidate.segment<3>(5).norm();
        if (g_norm_cand < 5.0 || g_norm_cand > 15.0 || x_candidate.hasNaN()) //
        {
            if (iter < 10)
                ROS_WARN("FastInit DEBUG (iter %d): Invalid candidate solution detected. Skipping this iteration.", iter);
            continue; // 必须跳过这次迭代！
        }

        // 3. 检验内点 (使用代数误差的平方)
        // 用于存储当前假设 x_candidate 所对应的内点的索引
        std::vector<int> current_inlier_indices;
        
        // 遍历所有的观测数据 (从 0 到 N-1)
        for (int i = 0; i < num_observations; ++i)
        {
            const auto& obs = all_observations[i];

            // 为当前观测构建 A_row 和 b_row
            Eigen::Matrix<double, 2, 8> A_row;
            Eigen::Vector2d b_row;
            buildLinearSystemRow(obs.pre_integration_k, obs.z_i0, obs.z_ik, obs.d_hat_i, A_row, b_row);
            
            // 计算残差向量: residual = A_row * x_candidate - b_row
            // 这就是论文中提到的 "代数误差"。理想情况下，如果 x_candidate 是真解，
            // 且观测 obs 没有噪声，那么 residual 应该是一个 2x1 的零向量。
            // squaredNorm() 计算残差向量的 L2 范数的平方 (|residual|^2)
            double residual_sq_norm = (A_row * x_candidate - b_row).squaredNorm();
            

            // 将残差的平方与预设的阈值 (平方) 进行比较。
            // RANSAC_THRESHOLD_SQ 通常需要根据经验调整，它决定了我们对内点的容忍程度。
            if (residual_sq_norm < fast_mono::RANSAC_THRESHOLD_SQ)
            {
                // 如果残差足够小，则认为这个观测是当前假设 x_candidate 的一个内点。
                current_inlier_indices.push_back(i);
            }
        }

        // 4. 如果找到更多内点，则更新最优解
        if (current_inlier_indices.size() > best_inlier_indices.size())
        {
            best_inlier_indices = current_inlier_indices;
            ROS_DEBUG("RANSAC iter %d: Found new best model with %zu inliers.", iter, best_inlier_indices.size());
        }

        // (可选) 提前退出 RANSAC
        if (best_inlier_indices.size() > num_observations * 0.7) // Example: 70% inliers
            break;
    }

    ROS_INFO("RANSAC finished. Best model has %zu inliers.", best_inlier_indices.size());

    // 检查内点数是否足够
    if (best_inlier_indices.size() < fast_mono::RANSAC_MIN_INLIERS)
    {
        ROS_WARN("RANSAC failed: Not enough inliers found (%zu / %d required).",
                 best_inlier_indices.size(), fast_mono::RANSAC_MIN_INLIERS);
        return false;
    }

    // 5. 使用所有找到的内点重新拟合模型
    std::vector<ObservationData> best_inliers_data;

    // 从 all_observations 中提取出所有被判定为最佳内点的观测数据
    for (int idx : best_inlier_indices)
    {
        best_inliers_data.push_back(all_observations[idx]);
    }

    // 调用 solveLinearSystem 函数，使用所有内点数据来计算一个更精确的解 best_x。
    // 这个解会考虑重力约束。
    if (!solveLinearSystem(best_inliers_data, best_x))
    {
        ROS_WARN("RANSAC final fit failed.");
        return false;
    }

    // 如果 RANSAC 找到足够内点，并且最终拟合成功，则函数返回 true。
    // 最优解存储在 best_x 中。
    return true;
}


// 使用所有内点求解线性系统 (带简化的重力约束)
bool FastInitializer::solveLinearSystem(const std::vector<ObservationData>& observations,
                                        Eigen::Matrix<double, 8, 1>& x_out)
{
    // --- 1. 输入检查与矩阵构建 ---
    
    // 获取内点（可信观测）的数量。
    int num_inliers = observations.size();
    
    // 检查内点数量是否足够构成一个超定系统（方程数 > 未知数）。
    // 理论上，只要 num_inliers >= 4 就可以解，但通常需要更多点来获得稳定解。
    if (num_inliers < fast_mono::RANSAC_MIN_MEASUREMENTS) return false;

    // 构建一个大的 (N*2 x 8) 矩阵 A 和 (N*2 x 1) 向量 b。
    // N 是内点数量 (num_inliers)。
    Eigen::MatrixXd A(num_inliers * 2, 8);
    Eigen::VectorXd b(num_inliers * 2);

    // 遍历所有内点观测。
    for (int i = 0; i < num_inliers; ++i)
    {
        // 为每个观测构建对应的 2x8 A_row 和 2x1 b_row。
        Eigen::Matrix<double, 2, 8> A_row;
        Eigen::Vector2d b_row;
        buildLinearSystemRow(observations[i].pre_integration_k, observations[i].z_i0,
                             observations[i].z_ik, observations[i].d_hat_i, A_row, b_row);
                             
        // 将 A_row 和 b_row 堆叠到大矩阵 A 和大向量 b 中。
        A.block<2, 8>(i * 2, 0) = A_row;
        b.block<2, 1>(i * 2, 0) = b_row;
    } // 循环结束后，A 和 b 构成了超定线性系统 Ax = b

    // --- 2. 步骤 1: 求解无约束最小二乘问题 ---
    
    // 使用 SVD 求解超定系统 Ax = b 的最小二乘解。
    // 这个解 x_svd 会最小化残差的平方和 |Ax - b|^2。
    Eigen::Matrix<double, 8, 1> x_svd = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    
    // 从无约束解 x_svd 中提取重力向量部分 (最后 3 个元素)。
    Eigen::Vector3d g_svd = x_svd.segment<3>(5);
    // 计算其模长。
    double g_norm_svd = g_svd.norm();

    // --- 3. (可选) 检查无约束解的有效性 ---
    // 如果无约束解得到的重力模长非常接近于零，说明数据可能有问题或者存在退化配置。
    // 在这种情况下，强行施加约束可能会得到更差的结果。
    if (g_norm_svd < 1.0) {
        ROS_WARN("Linear system solve (SVD) resulted in near-zero gravity (%.3f).", g_norm_svd);
        // 这里选择返回失败。另一种策略是直接返回 x_svd。
        return false;
    }

    // --- 4. 步骤 2: 施加重力大小约束 ---
    
    // g_normed 是最终我们希望得到的重力向量。
    // 它的方向来自无约束解 g_svd (g_svd / g_norm_svd 是单位方向向量) 即 g_svd.normalized()。
    // 它的模长来自全局参数 G.norm() (在 parameters.h 中定义，例如 9.805)。
    Eigen::Vector3d g_normed = g_svd.normalized() * G.norm(); 

    // --- 5. 步骤 3: 固定 g_normed，重新求解 y = [a, b, v_I0]^T ---
    
    // 将原系统 Ax = b 分解为 A = [A_y | A_g]，其中 A_y 是前 5 列，A_g 是后 3 列。
    // A_y 对应变量 y = [a, b, v_I0]^T，A_g 对应变量 g_I0。
    // 原方程变为 A_y * y + A_g * g = b。
    
    // 固定 g = g_normed，我们需要解的方程变为: A_y * y = b - A_g * g_normed。
    
    // 提取 A_y (前 5 列)。
    Eigen::MatrixXd A_y = A.leftCols<5>(); 
    // 提取 A_g (后 3 列)。
    Eigen::MatrixXd A_g = A.rightCols<3>(); 
    
    // 计算新的右端项 b_y = b - A_g * g_normed。
    Eigen::VectorXd b_y = b - A_g * g_normed;

    // --- 6. 步骤 4: 使用 SVD 求解 A_y * y = b_y ---
    
    // 再次使用 SVD 求解这个关于 y 的超定系统的最小二乘解。
    Eigen::VectorXd y = A_y.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_y);

    // --- 7. (可选) 检查约束解的有效性 ---
    // y(0) 对应的是变量 a。
    // 如果在施加重力约束后，计算出的 a 非常接近于 0，这意味着深度 z = 1 / (a*d_hat + b)
    // 变得非常大或者主要由 b 决定，这通常是不稳定或不物理的。
    if (std::abs(y(0)) < 1e-3) {
        ROS_WARN("Linear system solve (Constrained) resulted in near-zero 'a' (%.3f). Using SVD solution instead.", y(0));
        // 在这种情况下，我们宁愿放弃约束解，选择更可能稳定的无约束解 x_svd。
        x_out = x_svd; 
        return true; // 仍然认为求解是成功的，只是用了无约束结果
    }

    // --- 8. 步骤 5: 组合最终解 ---
    
    // 将求解得到的 y (前 5 个元素) 和施加约束后的 g_normed (后 3 个元素) 
    // 组合成最终的 8x1 解向量 x_out。
    x_out.head<5>() = y;          // x_out[0]..x_out[4] = a, b, v_I0
    x_out.tail<3>() = g_normed;   // x_out[5]..x_out[7] = g_normed
    
    // 如果所有步骤都顺利完成，返回 true。
    return true;
}