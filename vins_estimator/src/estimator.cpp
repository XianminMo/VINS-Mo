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
}

/**
 * @brief 初始化深度估计器（如果启用）
 * 这个函数应该在 readParameters() 之后被调用
 */
void Estimator::initDepthEstimator()
{
    if (USE_FAST_INIT)
    {
        ROS_INFO("Initializing DepthEstimator...");
        // 创建一个深度估计器的智能指针实例
        mp_depth_estimator = std::make_unique<DepthEstimator>();
        // 使用配置文件中指定的模型路径初始化深度估计器
        if (!mp_depth_estimator->init(DEPTH_MODEL_PATH))
        {
        // 如果初始化失败
            // 打印致命错误日志，提示用户检查模型路径和相关配置（如ONNX, CUDA）
            ROS_FATAL("DepthEstimator initialization failed! Please check the model path and ONNX/CUDA configuration.");
            // 可以在此处添加 ros::shutdown() 来终止节点，防止进一步错误
            ros::shutdown();
        }

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

    // 添加对我们新成员变量的重置
    {
        // 使用互斥锁保护深度图相关成员变量的访问，确保线程安全
        std::lock_guard<std::mutex> lock(m_depth_mutex);
        m_first_frame_depth_computed = false; // 标记第一帧深度图未计算
        m_first_frame_depth_map.release();    // 释放深度图内存
    }

    failure_occur = 0; // 失败标志位清零
    relocalization_info = 0; // 重定位信息标志位清零

    // 漂移校正参数初始化
    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
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
                // 计算当前滑窗起始帧（最小 start_frame，若无则 0）
                int window_start_frame_id = 0;
                if (!f_manager.feature.empty()) {
                    window_start_frame_id = f_manager.feature.front().start_frame;
                    for (const auto& fe : f_manager.feature)
                        window_start_frame_id = std::min(window_start_frame_id, fe.start_frame);
                    if (window_start_frame_id < 0) window_start_frame_id = 0;
                }

                // 1. 计算第一帧的深度图 (如果尚未计算)
                if (!m_first_frame_depth_computed || m_depth_window_start_id != window_start_frame_id)
                {
                    double first_frame_stamp = Headers[window_start_frame_id].stamp.toSec();
                    cv::Mat first_img;

                    auto it = all_image_frame.find(first_frame_stamp);
                    if (it != all_image_frame.end() && !it->second.raw_image.empty())
                    {
                        first_img = it->second.raw_image;
                    }

                    if (!first_img.empty())
                    {
                        ROS_INFO("Fast-Init: Calculating depth for the first frame...");
                        TicToc t_depth;
                        cv::Mat depth_map;
                        // 调用深度学习模型进行深度预测, first_img 为 bgr
                        bool success = mp_depth_estimator->predict(first_img, depth_map);
                        if (success)
                        {
                            std::lock_guard<std::mutex> lock(m_depth_mutex);
                            m_first_frame_depth_map = depth_map; // 存储深度图
                            m_first_frame_depth_computed = true; // 标记深度图已计算
                            ROS_INFO("Fast-Init: Depth prediction succeeded (%.2f ms).", t_depth.toc());
                        }
                    }
                    else
                    {
                        ROS_WARN_THROTTLE(1.0, "Fast-Init: Waiting for the raw image of the first frame...");
                    }
                }

                // 2. 检查深度图是否就绪
                if (m_first_frame_depth_computed)
                {
                    // 在这里调用基于RANSAC的快速初始化求解器，利用深度图恢复尺度、重力和速度。
                    TicToc t_fast_init;
                    std::map<int, Eigen::Vector3d> Ps_init;
                    std::map<int, Eigen::Vector3d> Vs_init;
                    std::map<int, Eigen::Quaterniond> Rs_init;

                    bool fast_init_ok = false;
                    // 初始化门槛：特征数
                    if (f_manager.getFeatureCount() < 80) {
                        ROS_WARN_THROTTLE(1.0, "Fast-Init gate: too few features (%d).", f_manager.getFeatureCount());
                    } else {
                        // 旋转激励（累计角度）
                        double rot_sum = 0.0;
                        for (int k = 1; k <= WINDOW_SIZE; ++k) {
                            if (pre_integrations[k]) {
                                Eigen::AngleAxisd aa(pre_integrations[k]->delta_q);
                                rot_sum += std::abs(aa.angle());
                            }
                        }
                        if (rot_sum < 0.35) { // ~20度，按需调整
                            ROS_WARN_THROTTLE(1.0, "Fast-Init gate: insufficient rotation (%.3f rad).", rot_sum);
                        } else {
                            // 满足门槛再调初始化
                            std::lock_guard<std::mutex> lock(m_depth_mutex);
                            fast_init_ok = mp_fast_initializer->initialize(all_image_frame,
                                                                        m_first_frame_depth_map,
                                                                        g,
                                                                        Ps_init, Vs_init, Rs_init);
                        }
                    }

                    if (fast_init_ok)
                    {
                        ROS_INFO("Fast Monocular Init Succeeded! (%.2f ms)", t_fast_init.toc());

                        // 将 FastInitializer 返回的结果复制到 Estimator 的状态变量 Ps, Vs, Rs
                        // 注意索引 k (0 to WINDOW_SIZE)
                        for (int k = 0; k <= WINDOW_SIZE; ++k)
                        {
                            // 检查 map 中是否存在该帧的结果
                            if (Ps_init.count(k) && Vs_init.count(k) && Rs_init.count(k))
                            {
                                Ps[k] = Ps_init[k];
                                Vs[k] = Vs_init[k];
                                Rs[k] = Rs_init[k];
                                // (重要) 将所有初始化的帧标记为关键帧
                                // 注意：all_image_frame 的 key 是时间戳，需要查找
                                double timestamp = Headers[k].stamp.toSec();
                                if (all_image_frame.count(timestamp)) {
                                    all_image_frame[timestamp].is_key_frame = true;
                                }
                            }
                            else {
                                // 如果 FastInitializer 没有返回某个帧的状态，这是个严重错误
                                ROS_ERROR("FastInit Error: Missing state for frame %d in initializer output!", k);
                                fast_init_ok = false; // 标记为失败
                                break;
                            }
                        }

                        if (fast_init_ok) {
                            is_init_success = true; // 标记整个初始化成功
                        }
                    }
                    else
                    {
                        ROS_WARN("Fast Monocular Init Failed.");
                        std::lock_guard<std::mutex> lock(m_depth_mutex);
                        m_first_frame_depth_computed = false;
                        m_depth_window_start_id = -1;
                        // 如果失败，m_first_frame_depth_computed 仍然是 true，
                        // 但 is_init_success 是 false，会触发 slideWindow()，
                        // slideWindow() 中的重置逻辑会处理 m_first_frame_depth_computed
                    }

                    
                }
                else {
                 ROS_WARN_THROTTLE(1.0, "Fast-Init: Waiting for depth map computation...");
                }
            }
            else
            {
                // --- 分支 2: 执行 VINS-Mono 原版初始化流程 ---
                if(ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
                {
                    // 步骤 1: 纯视觉SFM（Structure from Motion）
                    if (initialStructure())
                    {
                        // 步骤 2: 视觉-IMU对齐
                        if (visualInitialAlign())
                        {
                            is_init_success = true; // 初始化成功
                        }
                    }
                   initial_timestamp = header.stamp.toSec();
                }
            }
            
            // --- 统一处理初始化结果 ---
            if(is_init_success)
            {
                solver_flag = NON_LINEAR; // 切换到非线性优化模式
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
    // 找到一个与最新帧有足够视差和共视点的历史帧 l
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    
    // 使用 GlobalSFM 类来恢复所有帧的位姿和地图点
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
    // 全局SFM只恢复了被选为关键帧的位姿，其他帧需要通过PnP来确定位姿
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
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
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
    // 将尺度 s 应用到所有位置和速度上
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
    // 从滑窗内第一帧开始向前遍历
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
            
            // 如果平均视差（乘以焦距后，近似为像素距离）足够大
            // 并且能成功求解相对位姿（通过8点法等）
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
        // 只使用被观测到至少两次，且起始帧在滑动窗口内的特征点
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历该特征点被观测到的所有其他帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else // 不考虑时间延迟
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
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
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i; // 找到它在滑动窗口中的局部索引
            relocalization_info = 1;    // 设置标志，以便在下一次优化中加入闭环约束
            // 记录下当前对该帧位姿的估计值
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}
