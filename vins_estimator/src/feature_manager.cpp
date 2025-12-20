#include "feature_manager.h"
#include <random>
#include <unordered_set>

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int current_frame_idx, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0; // 记录从历史帧到当前帧成功追踪的特征点数量

    // 1. 将当前帧的特征点添加到管理器中
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td, current_frame_idx); // [0] 是指第一个相机

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 如果是新特征，则创建新的 FeaturePerId
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, current_frame_idx));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 如果是已存在的特征，则添加到观测序列中
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // 2. 提前判断：如果帧数太少或跟踪点太少，则直接接受为关键帧
    const int MIN_TRACK_FOR_PARALLAX = 20;
    if (current_frame_idx < 2 || last_track_num < MIN_TRACK_FOR_PARALLAX)
        return true;

    // 3. 计算当前帧与上一帧之间的平均视差
    for (auto &it_per_id : feature)
    {
        // 确保特征点在上一帧和当前帧都被观测到
        if (it_per_id.start_frame <= current_frame_idx - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= current_frame_idx - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, current_frame_idx);
            parallax_num++;
        }
    }

    // 4. 根据平均视差决定是否为关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}


/**
 * @brief 获取在两帧之间被同时观测到的特征点的对应关系
 *
 * 该函数遍历所有特征点，查找那些在输入的两帧（frame_count_l, frame_count_r）都被观测到的特征点。
 * 对于每个满足条件的特征点，取得其在这两帧中的归一化像素坐标，作为一组对应点输出。
 *
 * 通常用于双目/前后帧间的两视图几何估计，比如估算相对位姿（PnP，E/F矩阵等）。
 * 
 * @param frame_count_l 左图/第一帧在滑动窗口中的frame_count索引
 * @param frame_count_r 右图/第二帧在滑动窗口中的frame_count索引
 * @return vector<pair<Vector3d, Vector3d>> 返回所有在两帧中都被观测到的特征点的归一化坐标对
 *         pair的first为frame_count_l帧中的归一化点坐标，second为frame_count_r帧的归一化点坐标
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres; // 存储输出的对应点对
    for (auto &it : feature)
    {
        // 检查该特征点是否在frame_count_l和frame_count_r都被观测到
        // 只有当特征点的观测起始帧早于等于frame_count_l，结束帧晚于等于frame_count_r才满足条件
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            // 计算feature_per_frame容器中对应两帧的索引
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            // 获取该特征点在左帧和右帧（frame_count_l和frame_count_r）下的归一化坐标
            Vector3d a = it.feature_per_frame[idx_l].point;
            Vector3d b = it.feature_per_frame[idx_r].point;
            
            // 保存到输出vector
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

/**
 * @brief 设置单个特征点的估计深度
 * @param feature_id 要设置深度的特征点ID
 * @param depth 估计的深度值
 */
void FeatureManager::setFeatureDepth(int feature_id, double depth)
{
    auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                      {
        return it.feature_id == feature_id;
                      });
    if (it != feature.end())
    {
        it->estimated_depth = depth;
        it->solve_flag = (depth > 0) ? 1 : 2; // 1: succ, 2: fail
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int marg_frame_idx)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == marg_frame_idx)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < marg_frame_idx - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @brief 计算补偿后的视差
 *
 * @param it_per_id 特征点数据
 * @param current_frame_idx 当前帧在滑动窗口中的索引
 * @return 视差值
 *
 * **功能**: 计算该特征点在 frame[current_frame_idx-2] 和 frame[current_frame_idx-1] 之间的视差
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int current_frame_idx)
{
    // 检查倒数第二帧是否为关键帧
    // 计算倒数第二帧与倒数第三帧之间的视差
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[current_frame_idx - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[current_frame_idx - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = current_frame_idx - 2;
    //int r_j = current_frame_idx - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}

/**
 * @brief 前端线性尺度对齐: VINS逆深度与网络逆深度对齐
 *
 * 执行前端尺度-偏移对齐以立即处理尺度跳变:
 *   min_sum || inv_d_vins - (s * inv_d_mean + t) ||^2
 *
 * 过滤无效对齐(负尺度、奇异矩阵、点数不足)
 *
 * @param current_frame_idx 当前帧在滑动窗口中的索引 (通常为 WINDOW_SIZE)
 * @return true 如果对齐成功
 */
 bool FeatureManager::SolveLinearAlignment(int current_frame_idx,
                                           const cv::Mat& depth_mean,
                                           const cv::Mat& depth_sigma,
                                           Vector3d Ps[],
                                           Matrix3d Rs[],
                                           const Vector3d& tic,
                                           const Matrix3d& ric,
                                           double& scale,
                                           double& shift)
 {
     // ========================================================================
     // Robust Linear Alignment (RANSAC + Refinement + Physical Gating)
     // Model: inv_d_vins = s * inv_d_net + t
     // Optimizations: Adaptive RANSAC, Scale Smoothing, Reduced Logging
     // ========================================================================

     const int MIN_POINTS = 10;  // Minimum points to attempt alignment

     if (depth_mean.empty() || depth_sigma.empty()) {
         return false;
     }

     // ========================================================================
     // Step 1: Data Preparation - Collect matched pairs
     // ========================================================================
     std::vector<double> inv_d_vins_vec;  // Target: VINS inverse depth (y)
     std::vector<double> inv_d_net_vec;   // Source: Network inverse depth (x)
     std::vector<double> sigma_vec;       // Network uncertainty
     std::vector<int> feature_id_vec;     // Track feature_id for each data point

     for (auto &it_per_id : feature)
     {
         // 1. Filter invalid VINS features
         if (it_per_id.solve_flag != 1 || it_per_id.estimated_depth <= 0) {
             continue;
         }

         // 2. Check observation in current frame
         if (it_per_id.start_frame > current_frame_idx || it_per_id.endFrame() < current_frame_idx) {
             continue;
         }

         int idx = current_frame_idx - it_per_id.start_frame;
         const FeaturePerFrame& frame_obs = it_per_id.feature_per_frame[idx];

         // 3. Project to pixel coordinates
         int u = static_cast<int>(frame_obs.uv.x() + 0.5);
         int v = static_cast<int>(frame_obs.uv.y() + 0.5);

         if (u < 0 || u >= depth_mean.cols || v < 0 || v >= depth_mean.rows) {
             continue;
         }

         // 4. Get Network Prediction
         float inv_d_nn = depth_mean.at<float>(v, u);
         float sigma_nn = depth_sigma.at<float>(v, u);

         if (inv_d_nn <= 1e-6 || std::isnan(inv_d_nn) || std::isinf(inv_d_nn)) {
             continue;
         }

         // 5. Convert VINS depth (meters) to Inverse Depth (1/m)
         // estimated_depth is defined as positive metric depth in VINS-Mono
         double inv_d_vins = 1.0 / it_per_id.estimated_depth;

         inv_d_vins_vec.push_back(inv_d_vins);
         inv_d_net_vec.push_back(static_cast<double>(inv_d_nn));
         sigma_vec.push_back(static_cast<double>(sigma_nn));
         feature_id_vec.push_back(it_per_id.feature_id);  // Record feature_id
     }

     int N = inv_d_vins_vec.size();
     if (N < MIN_POINTS) {
         // ROS_WARN("[Frontend Align] Insufficient points: %d (min: %d)", N, MIN_POINTS);
         return false;
     }

     // ========================================================================
     // Step 2: RANSAC - Robust Model Estimation (with Adaptive Early Termination)
     // ========================================================================
     const int RANSAC_ITERATIONS = 200;
     const double INLIER_THRESHOLD = 0.1;  // Error tolerance in inverse depth (1/m)
     const double MIN_X_DISTANCE = 0.01;   // Minimum disparity to avoid unstable scale
     const double EARLY_TERM_RATIO = 0.9;  // Optimization: Early exit if >90% inliers

     int best_inlier_cnt = 0;
     std::vector<int> best_inliers;
     double best_s_ransac = 1.0;
     double best_t_ransac = 0.0;

     // Use C++11 random engine for better randomness
     std::random_device rd;
     //  std::mt19937 gen(rd());
     std::mt19937 gen(12345);
     std::uniform_int_distribution<> dis(0, N - 1);

     for (int iter = 0; iter < RANSAC_ITERATIONS; ++iter)
     {
         // A. Sample 2 random points
         int i1 = dis(gen);
         int i2 = dis(gen);
         if (i1 == i2) continue;

         double x1 = inv_d_net_vec[i1];
         double y1 = inv_d_vins_vec[i1];
         double x2 = inv_d_net_vec[i2];
         double y2 = inv_d_vins_vec[i2];

         // B. Degeneracy Check
         if (std::abs(x1 - x2) < MIN_X_DISTANCE) continue;

         // C. Model Generation (y = sx + t)
         double s = (y1 - y2) / (x1 - x2);
         double t = y1 - s * x1;

         // D. Quick Physical Validity Check (Scale must be positive)
         if (s <= 0.01 || s > 10.0) continue;

         // E. Count Inliers
         std::vector<int> current_inliers;
         current_inliers.reserve(N);
         for (int i = 0; i < N; ++i) {
             double error = std::abs(inv_d_vins_vec[i] - (s * inv_d_net_vec[i] + t));
             if (error < INLIER_THRESHOLD) {
                 current_inliers.push_back(i);
             }
         }

         // F. Update Best Model
         if (current_inliers.size() > best_inliers.size()) {
             best_inliers = current_inliers;
             best_inlier_cnt = current_inliers.size();
             best_s_ransac = s;
             best_t_ransac = t;
         }

         // G. Adaptive Early Termination: Break if >90% inliers found (Efficiency Optimization)
         double current_inlier_ratio = static_cast<double>(best_inlier_cnt) / N;
         if (current_inlier_ratio > EARLY_TERM_RATIO) {
             ROS_DEBUG("[Frontend Align] Early termination at iter %d (inlier_ratio=%.2f%%)",
                       iter + 1, 100.0 * current_inlier_ratio);
             break;
         }
     }

     // H. RANSAC Failure Check
     const double MIN_INLIER_RATIO = 0.5; // Require at least 40% inliers
     double inlier_ratio = static_cast<double>(best_inlier_cnt) / N;

     if (best_inlier_cnt < MIN_POINTS || inlier_ratio < MIN_INLIER_RATIO) {
         ROS_WARN("[Frontend Align] RANSAC failed: %d/%d inliers (%.1f%%)",
                  best_inlier_cnt, N, 100.0 * inlier_ratio);
         return false;
     }

     // ========================================================================
     // Step 3: Final Refinement (Least Squares on Inliers)
     // ========================================================================
     int M = best_inliers.size();
     Eigen::MatrixXd A(M, 2);
     Eigen::VectorXd b(M);

     for (int i = 0; i < M; ++i) {
         int idx = best_inliers[i];
         A(i, 0) = inv_d_net_vec[idx];
         A(i, 1) = 1.0;
         b(i) = inv_d_vins_vec[idx];
     }

     // Solve Normal Equations: (A^T A) x = A^T b
     Eigen::MatrixXd AtA = A.transpose() * A;
     Eigen::VectorXd Atb = A.transpose() * b;

     // Numerical Stability Check
     Eigen::JacobiSVD<Eigen::MatrixXd> svd(AtA);
     double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
     if (cond > 1e8) {
         ROS_WARN("[Frontend Align] Singular matrix in refinement (cond=%.2e)", cond);
         return false;
     }

     Eigen::VectorXd x = AtA.ldlt().solve(Atb);
     scale = x(0);
     shift = x(1);

     // ========================================================================
     // Step 4: Physical Robustness Checks (Strict Gating)
     // ========================================================================

     // 1. Scale Guard
     if (scale < 0.01 || scale > 10.0) {
         ROS_WARN("[Frontend Align] Invalid scale: %.4f", scale);
         return false;
     }

     // 2. Bias Guard (Inverse depth shift should be small)
     if (std::abs(shift) > 5.0) {
         ROS_WARN("[Frontend Align] Large shift: %.4f", shift);
         return false;
     }

     // 3. Negative Depth Guard (CRUCIAL)
     const double MIN_VALID_DEPTH = 0.1;
     const double MAX_VALID_DEPTH = 50.0;

     // Check A: Verify Inliers (Must be valid)
     int invalid_inlier_cnt = 0;
     for (int idx : best_inliers) {
         double pred_inv_d = scale * inv_d_net_vec[idx] + shift;
         if (pred_inv_d <= 1e-6 || (1.0/pred_inv_d) > MAX_VALID_DEPTH) {
             invalid_inlier_cnt++;
         }
     }
     if (invalid_inlier_cnt > 0) {
         ROS_WARN("[Frontend Align] Rejected: %d inliers produce negative/invalid depth", invalid_inlier_cnt);
         return false;
     }

     // Check B: Verify All Points (Global consistency check)
     int invalid_total_cnt = 0;
     for (int i = 0; i < N; ++i) {
         double pred_inv_d = scale * inv_d_net_vec[i] + shift;
         if (pred_inv_d <= 1e-6 || (1.0/pred_inv_d) > MAX_VALID_DEPTH) {
             invalid_total_cnt++;
         }
     }
     double invalid_ratio = static_cast<double>(invalid_total_cnt) / N;
     if (invalid_ratio > 0.4) { // Allow up to 40% outliers globally if inliers are good
         ROS_WARN("[Frontend Align] Rejected: High global invalid ratio %.1f%%", invalid_ratio * 100.0);
         return false;
     }

     // ========================================================================
     // Step 5: Update Feature Attributes (INLIERS ONLY)
     // ========================================================================
     // Build inlier feature_id set from best_inliers indices
     std::unordered_set<int> inlier_feature_ids;
     for (int inlier_idx : best_inliers) {
         inlier_feature_ids.insert(feature_id_vec[inlier_idx]);
     }

     int applied_count = 0;
     int skipped_outliers = 0;

     for (auto &it_per_id : feature)
     {
         if (it_per_id.start_frame > current_frame_idx || it_per_id.endFrame() < current_frame_idx)
             continue;

         // **KEY CHANGE**: Only apply depth to RANSAC inliers
         if (inlier_feature_ids.find(it_per_id.feature_id) == inlier_feature_ids.end()) {
             skipped_outliers++;
             continue;  // Skip outliers
         }

         int idx = current_frame_idx - it_per_id.start_frame;
         FeaturePerFrame& frame_obs = it_per_id.feature_per_frame[idx];

         int u = static_cast<int>(frame_obs.uv.x() + 0.5);
         int v = static_cast<int>(frame_obs.uv.y() + 0.5);

         if (u >= 0 && u < depth_mean.cols && v >= 0 && v < depth_mean.rows) {
             float d_nn = depth_mean.at<float>(v, u);
             float sigma_nn = depth_sigma.at<float>(v, u);

             if (d_nn > 1e-6) {
                 // Apply alignment: inv_d = s * inv_d_net + t
                 double aligned_inv_d = scale * d_nn + shift;

                 // Only store if physically valid
                 if (aligned_inv_d > 1e-6) {
                     frame_obs.aligned_depth = 1.0 / aligned_inv_d; // Metric depth
                     frame_obs.aligned_sigma = scale * sigma_nn;    // Inverse depth uncertainty
                     applied_count++;
                 }
             }
         }
     }

     // Compute fitting error for logging
     Eigen::VectorXd residuals = A * x - b;
     double rmse = residuals.norm() / std::sqrt(M);

     // Log at INFO level with a throttle of 10 seconds
     ROS_INFO_THROTTLE(10, "[Frontend Align] Success: N=%d (Inliers=%d, Outliers=%d), Applied=%d, Skipped=%d, s=%.4f, t=%.4f, RMSE=%.4f",
               N, M, N - M, applied_count, skipped_outliers, scale, shift, rmse);

     return true;
 }