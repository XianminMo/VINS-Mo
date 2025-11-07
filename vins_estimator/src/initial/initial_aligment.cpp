#include "initial_alignment.h"

/**
 * @brief 利用视觉恢复的两帧间旋转初步校准IMU的陀螺仪零偏
 * 
 * 该函数遍历窗口内所有相邻帧，通过比较视觉恢复的两帧实际旋转(R)与IMU预积分得到的旋转(delta_q)，
 * 进而估算陀螺仪的偏置bg，通过最小二乘法求解一个最优修正量delta_bg。
 * 
 * 校准完成后，更新所有滑窗帧的陀螺仪bias，并将其重新repropagate到IMU预积分对象中。
 * 
 * @param all_image_frame 时间戳到ImageFrame的映射，包含滑窗内所有帧的视/IMU信息
 * @param Bgs 存储每一帧的gyroscope bias（在滑窗初始化阶段可直接修改为矫正结果）
 */
 void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
 {
     Matrix3d A;      // 用于积累线性系统Ax=b中的A
     Vector3d b;      // 用于积累线性系统Ax=b中的b
     Vector3d delta_bg; // 求解得到的陀螺仪bias修正量
     A.setZero();
     b.setZero();
 
     map<double, ImageFrame>::iterator frame_i;
     map<double, ImageFrame>::iterator frame_j;
 
     // 遍历所有相邻帧（i, j），i为前一帧，j为后一帧
     for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
     {
         frame_j = next(frame_i);
 
         // 用于临时存储一组系数
         MatrixXd tmp_A(3, 3);
         tmp_A.setZero();
         VectorXd tmp_b(3);
         tmp_b.setZero();
 
         // 视觉恢复的两帧之间的相对旋转
         // q_ij: i->j的旋转（由R_i, R_j计算得到，R*为右乘旋转）
         Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
 
         // 预积分中的jacobian块，对应于陀螺仪偏置对旋转的影响
         tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
 
         // 将IMU旋转增量与视觉旋转对比，构建线性观测模型
         // delta_q.inverse() * q_ij 表示IMU预积分旋转与视觉旋转的差异（四元数误差的向量部分）
         // 这里乘以2是因为单位四元数误差近似的线性化关系
         tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
 
         // 将该对帧对整体系统做累加,正规方程：
         // A = Σ(J^T * J)
         // b = Σ(J^T * residual)
         A += tmp_A.transpose() * tmp_A;
         b += tmp_A.transpose() * tmp_b;
     }
     // 求解线性方程，得到最佳的陀螺仪偏置修正量
     delta_bg = A.ldlt().solve(b);
 
     ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());
 
     // 将求得的偏置修正加到所有窗口帧的Bgs上（此时各帧bias应该是同步更新）
     for (int i = 0; i <= WINDOW_SIZE; i++)
         Bgs[i] += delta_bg;
 
     // 对所有帧的IMU预积分对象，重新propagate一次以应用新bias
     for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
     {
         frame_j = next(frame_i);
         // 注意：这里加速度bias置零，仅重新设置陀螺仪bias
         frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
     }
 }

bool solveGyroscopeBiasNew(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    // 收集候选观测（带旋转幅值与残差）
    struct ObsRow { Matrix3d J; Vector3d r; double angle; double res_norm; };
    std::vector<ObsRow> rows;
    rows.reserve(all_image_frame.size());

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;

    const double MIN_ROT_ANGLE = 2.0 * M_PI / 180.0; // 放宽门限，提高低激励可用性
    const int MAX_SPAN = 3; // 跨多帧复合预积分，最多跨 3 段

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        // 复合到多步的 j（跨度 1..MAX_SPAN）
        map<double, ImageFrame>::iterator fi = frame_i;
        Eigen::Matrix3d R_i = fi->second.R;
        Eigen::Quaterniond dq_imu_comp = Eigen::Quaterniond::Identity();
        Matrix3d J_comp = Matrix3d::Zero();

        for (int span = 1; span <= MAX_SPAN; ++span)
        {
            map<double, ImageFrame>::iterator fj = std::next(fi, span);
            if (fj == all_image_frame.end()) break;

            // 复合 IMU 旋转（右乘）与累加雅可比（小角度近似叠加）
            dq_imu_comp = dq_imu_comp * fj->second.pre_integration->delta_q;
            J_comp += fj->second.pre_integration->jacobian.template block<3,3>(O_R, O_BG);

            // 视觉相对旋转（i->j）
            Eigen::Quaterniond q_ij(R_i.transpose() * fj->second.R);

            double theta_imu = Eigen::AngleAxisd(dq_imu_comp).angle();
            if (theta_imu < MIN_ROT_ANGLE) continue;

            Vector3d r = 2.0 * (dq_imu_comp.inverse() * q_ij).vec();
            double rn = r.norm();
            rows.push_back({J_comp, r, theta_imu, rn});
        }
    }

    if (rows.size() < 3)
    {
        ROS_WARN("solveGyroscopeBias: insufficient valid pairs (%zu)", rows.size());
        return false;
    }

    // 鲁棒剔除：按残差分位数丢弃最大 20%
    std::vector<double> res_norms; res_norms.reserve(rows.size());
    for (auto &row : rows) res_norms.push_back(row.res_norm);
    size_t n = res_norms.size();
    size_t idx80 = static_cast<size_t>(std::floor(0.8 * n));
    std::nth_element(res_norms.begin(), res_norms.begin() + idx80, res_norms.end());
    double thr = res_norms[idx80];

    // 组装加权正规方程（权重 ∝ 旋转幅值，且只保留残差较小的）
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    size_t used = 0;
    for (auto &row : rows)
    {
        if (row.res_norm > thr) continue;
        double w = std::max(row.angle, MIN_ROT_ANGLE);
        Matrix3d Jw = std::sqrt(w) * row.J;
        Vector3d rw = std::sqrt(w) * row.r;
        A += Jw.transpose() * Jw;
        b += Jw.transpose() * rw;
        used++;
    }

    if (used < 3)
    {
        ROS_WARN("solveGyroscopeBias: too few rows after robust filtering (%zu)", used);
        return false;
    }

    // 求解线性方程，得到最佳的陀螺仪偏置修正量
    Vector3d delta_bg = A.ldlt().solve(b);
    // 大幅度保护门：如 >0.5 rad/s 则认为数据不一致，跳过本次更新
    double mag = delta_bg.norm();
    const double MAX_GYR_BIAS_STEP = 0.5; // 可配
    if (mag > MAX_GYR_BIAS_STEP) {
        ROS_WARN_STREAM("solveGyroscopeBias: delta_bg too large (" << mag
                        << " rad/s). Skip applying this update.");
        return false;
    }

    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 将求得的偏置修正加到所有窗口帧的Bgs上（此时各帧bias应该是同步更新）
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    // 对所有帧的IMU预积分对象，重新propagate一次以应用新bias
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        // 注意：这里加速度bias置零，仅重新设置陀螺仪bias
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
    return true;
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
