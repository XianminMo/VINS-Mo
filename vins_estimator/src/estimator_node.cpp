#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

// --- MODIFICATION START ---
// 用于缓存第一帧原始图像
cv::Mat first_image_mat;
bool first_image_received = false;
// --- MODIFICATION END ---

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator; // This mutex is used to protect the estimator object

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
Estimator* estimator_ptr; // Use a global pointer instead of a global object
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator_ptr->Ps[WINDOW_SIZE];
    tmp_Q = estimator_ptr->Rs[WINDOW_SIZE];
    tmp_V = estimator_ptr->Vs[WINDOW_SIZE];
    tmp_Ba = estimator_ptr->Bas[WINDOW_SIZE];
    tmp_Bg = estimator_ptr->Bgs[WINDOW_SIZE];
    acc_0 = estimator_ptr->acc_0;
    gyr_0 = estimator_ptr->gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator_ptr->td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator_ptr->td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator_ptr->td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front()); // 用于插值计算，使得imu与img时间严格对齐
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header; // NOLINT
        header.frame_id = "world";
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// --- MODIFICATION START ---
/**
 * @brief 订阅原始图像的回调函数
 * 
 * 仅在启用快速初始化时，缓存收到的第一帧图像。
 * @param img_msg 原始图像消息
 */
void raw_image_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 仅在启用快速初始化且尚未收到第一帧图像时执行
    if (USE_FAST_INIT && !first_image_received)
    {
        try
        {
            // 1. 先将 ROS 图像消息安全地转换为 OpenCV Mat (mono8)
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
            
            // 2. 创建一个三通道的 BGR 图像
            cv::Mat bgr_image;
            cv::cvtColor(cv_ptr->image, bgr_image, cv::COLOR_GRAY2BGR);

            // 3. 缓存转换后的 BGR 图像
            m_buf.lock(); // 使用与 process 线程相同的锁来确保线程安全
            first_image_mat = bgr_image.clone();
            first_image_received = true;
            m_buf.unlock();
            ROS_INFO("Fast-Init: Successfully cached and converted the first raw image (mono8 -> bgr8).");
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}
// --- MODIFICATION END ---

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator_ptr->clearState();
        estimator_ptr->setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
        // --- MODIFICATION START ---
        // 重启时也要重置第一帧图像的缓存
        m_buf.lock();
        first_image_received = false;
        first_image_mat.release();
        m_buf.unlock();
        // --- MODIFICATION END ---
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();
        m_estimator.lock();

        // --- MODIFICATION START ---
        // 如果启用了快速初始化，并且收到了第一帧图像，则将其传递给 estimator
        if (USE_FAST_INIT && first_image_received)
        {
            // 检查 estimator_ptr 是否是第一次处理图像 (frame_count == 0)
            if (estimator_ptr->solver_flag == Estimator::SolverFlag::INITIAL && estimator_ptr->frame_count == 0)
            {
                estimator_ptr->setFirstImage(first_image_mat);
                first_image_mat.release(); // 传递后释放内存，防止重复传递
            }
        }
        // --- MODIFICATION END ---

        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator_ptr->td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;   
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator_ptr->processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else // 线性插值到img_t时刻的imu数据
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator_ptr->processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = static_cast<int>(relo_msg->channels[0].values[7]);
                estimator_ptr->setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            // 构建image数据结构
            // feature_id -> (camera_id, [x,y,z,u,v,velocity_x,velocity_y])
            // x,y,z: 归一化坐标
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator_ptr->processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(*estimator_ptr, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(*estimator_ptr, header);
            pubKeyPoses(*estimator_ptr, header);
            pubCameraPose(*estimator_ptr, header);
            pubPointCloud(*estimator_ptr, header);
            pubTF(*estimator_ptr, header);
            pubKeyframe(*estimator_ptr);
            if (relo_msg != NULL)
                pubRelocalization(*estimator_ptr);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // 初始化ROS节点，节点名称为"vins_estimator"
    ros::init(argc, argv, "vins_estimator");
    // 创建ROS节点句柄，"~"表示节点的私有命名空间
    ros::NodeHandle n("~");
    // 设置ROS日志级别为INFO（只输出INFO及以上级别的日志）
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 从ROS参数服务器读取配置参数（如相机内参、IMU参数等）
    readParameters(n);
    
    // 在读取参数后创建 Estimator 对象
    Estimator estimator;
    estimator_ptr = &estimator;
    // 初始化深度估计器（如果启用）
    estimator.initDepthEstimator();
    estimator.setParameter();

    // 如果定义了EIGEN_DONT_PARALLELIZE宏（禁用Eigen并行计算），输出调试日志
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    // 输出警告日志，提示系统正在等待图像和IMU数据
    ROS_WARN("waiting for image and imu...");

    // 注册ROS发布者（用于发布估计结果，如位姿、点云等可视化或调试信息）
    registerPub(n);

    // 订阅IMU数据：话题名为IMU_TOPIC（配置文件中定义），队列长度2000，回调函数为imu_callback
    // 使用tcpNoDelay()优化TCP传输，减少延迟
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());

    // 订阅特征跟踪器输出的特征点数据：话题为"/feature_tracker/feature"，回调函数为feature_callback
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);

    // --- MODIFICATION START ---
    // 如果启用了快速初始化，则额外订阅原始图像话题
    ros::Subscriber sub_raw_image;
    if (USE_FAST_INIT)
    {
        sub_raw_image = n.subscribe(IMAGE_TOPIC, 100, raw_image_callback);
        ROS_INFO("Fast-Init: Subscribing to raw image topic: %s", IMAGE_TOPIC.c_str());
    }
    // --- MODIFICATION END ---

    // 订阅重启信号：话题为"/feature_tracker/restart"，回调函数为restart_callback
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);

    // 订阅回环检测匹配点数据：话题为"/pose_graph/match_points"，回调函数为relocalization_callback
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // 启动一个独立线程运行process函数（核心处理逻辑，负责视觉-惯性数据融合）
    std::thread measurement_process{process};

    // ROS消息循环：阻塞等待并处理订阅的消息（回调函数在该循环中执行）
    ros::spin();

    return 0;
}
