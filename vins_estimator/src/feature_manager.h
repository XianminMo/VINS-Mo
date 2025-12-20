#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <ctime>
#include <opencv2/opencv.hpp>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

// 存储单帧图像中的单个特征点信息
// 设计要点：
// - 构造函数接收一个7维向量`_point`，包含了特征点的完整信息
// - 存储了特征点的多种表示：归一化坐标、像素坐标、速度等
// - 包含了优化所需的变量（A、b、dep_gradient）和状态标识（is_used）
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td, int _frame_id = -1)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
        frame_id = _frame_id;
        aligned_depth = -1.0;     // Invalid by default
        aligned_sigma = -1.0;     // Invalid by default
    }
    double cur_td;
    int frame_id;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;

    // Frontend-aligned depth from TTA inference
    double aligned_depth;  // d_aligned = s * d_mean + t
    double aligned_sigma;  // sigma_aligned = s * sigma_tta
};


// 存储某个特定ID的特征点在多帧中的表现
// 设计要点：
// - 管理同一个特征ID在多帧中的观测信息
// - 记录特征点的生命周期状态（首次出现帧、使用次数、边缘化状态等）
// - 维护特征点的深度估计结果和求解状态
class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame; // 该特征点在多帧中的表现

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

// 设计要点：
// - 这是整个特征管理系统的核心类，负责特征点的添加、删除、深度估计等操作
// - 维护了所有特征点的列表，并提供各种查询和管理方法
// - 特别关注视差计算和关键帧选择（通过`addFeatureCheckParallax`方法）
// - 支持多相机系统（通过`ric`数组和`NUM_OF_CAM`参数）
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    /**
     * @brief 添加当前帧的特征点并检查视差以决定是否为关键帧
     *
     * @param current_frame_idx 当前帧在滑动窗口中的索引
     *                          - 窗口未满时: 0, 1, 2, ..., frame_count
     *                          - 窗口满后: 始终为 WINDOW_SIZE (新帧总在最后位置)
     * @param image 当前帧的特征点数据 (feature_id -> 特征点信息)
     * @param td 时延参数
     * @return true 如果应该将当前帧标记为关键帧
     *
     * **工作流程**:
     * 1. 将当前帧的特征点添加到 feature 列表中
     *    - 新特征点: 创建 FeaturePerId, start_frame = current_frame_idx
     *    - 已有特征点: 追加到 feature_per_frame 末尾
     * 2. 计算当前帧与上一帧之间的平均视差
     * 3. 根据视差大小决定是否为关键帧
     */
    bool addFeatureCheckParallax(int current_frame_idx, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void setFeatureDepth(int feature_id, double depth); // <-- 新增
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    /**
     * @brief 移除次新帧(MARGIN_SECOND_NEW策略)
     *
     * @param marg_frame_idx 被边缘化帧在滑动窗口中的索引
     *                       - 通常为 WINDOW_SIZE (最新帧的索引)
     *
     * **调用时机**: 当 marginalization_flag == MARGIN_SECOND_NEW 时
     * **作用**: 移除最新帧的观测数据,或调整特征点的 start_frame
     */
    void removeFront(int marg_frame_idx);
    void removeOutlier();

    /**
     * @brief Solve linear least squares alignment between VINS and network depth
     *
     * @param current_frame_idx 当前帧在滑动窗口中的索引
     *                          - 窗口满后通常为 WINDOW_SIZE (最新帧位置)
     * @param depth_mean TTA-averaged depth map (CV_32F)
     * @param depth_sigma TTA uncertainty map (CV_32F)
     * @param Ps Position array
     * @param Rs Rotation array
     * @param tic Camera to IMU translation
     * @param ric Camera to IMU rotation
     * @param scale Output scale factor s
     * @param shift Output shift factor t
     * @return true if alignment succeeded (enough valid points, positive scale)
     *
     * **对齐公式**: inv_d_vins = s * inv_d_net + t (在逆深度空间)
     * **使用 current_frame_idx**: 用于定位特征点在 feature_per_frame 中的观测
     */
    bool SolveLinearAlignment(int current_frame_idx,
                              const cv::Mat& depth_mean,
                              const cv::Mat& depth_sigma,
                              Vector3d Ps[],
                              Matrix3d Rs[],
                              const Vector3d& tic,
                              const Matrix3d& ric,
                              double& scale,
                              double& shift);

    list<FeaturePerId> feature;
    int last_track_num;

  private:
    /**
     * @brief 计算特征点在两帧之间的补偿视差
     *
     * @param it_per_id 特征点数据
     * @param current_frame_idx 当前帧在滑动窗口中的索引
     * @return 补偿后的视差值
     *
     * **用途**: 计算 frame[current_frame_idx - 2] 与 frame[current_frame_idx - 1] 之间的视差
     */
    double compensatedParallax2(const FeaturePerId &it_per_id, int current_frame_idx);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif