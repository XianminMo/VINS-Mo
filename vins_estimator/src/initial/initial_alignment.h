#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, 
            double _t, IntegrationBase* _pre_integration, const cv::Mat& _raw_image)
            : points{_points}, t{_t}, pre_integration{_pre_integration}, raw_image{_raw_image.clone()}, 
            is_key_frame{false} {}
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration;
        cv::Mat raw_image;
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);

// 仅陀螺仪零偏求解（用于快速初始化后的小步校准）
// 返回 true 表示已成功应用零偏更新并对 all_image_frame 的预积分完成重传播；false 表示本轮跳过
bool solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs);