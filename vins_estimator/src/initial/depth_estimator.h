#pragma once

#include <string>
#include <vector>
#include <memory> // 用于 std::unique_ptr

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"


#include <ros/ros.h>

class DepthEstimator
{
public:
    /**
     * @brief 构造函数
     */
    DepthEstimator();

    /**
     * @brief 析构函数
     */
    ~DepthEstimator();

    /**
     * @brief 初始化 ONNX Runtime 环境、会话并加载模型。
     * @param model_path ONNX 模型的路径。
     * @return true 如果初始化成功。
     */
    bool init(const std::string& model_path);

    /**
     * @brief 对输入的 BGR 图像执行深度估计。
     * @param image [in] 输入的 cv::Mat 图像 (BGR, CV_8UC3)。
     * @param norm_inv_depth_map [out] 输出的 cv::Mat (CV_32F)。
     * 该 Mat 包含归一化到 [0.0, 1.0] 范围的相对逆深度，
     * 尺寸与输入图像相同，可以直接用于特征点采样。
     * @return true 如果预测成功。
     */
    bool predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map);


private:
    /**
     * @brief 预处理函数 (来自 Demo)
     */
    void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values);

    // --- ONNX Runtime 核心成员 ---
    Ort::Env m_env;
    Ort::SessionOptions m_session_options;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    // --- 模型输入/输出信息 ---
    // 使用 ONNX Runtime 原生智能指针，它会自动管理内存
    Ort::AllocatedStringPtr m_input_name_ptr;
    Ort::AllocatedStringPtr m_output_name_ptr;
    const char* m_input_name;
    const char* m_output_name;

    std::vector<int64_t> m_input_shape;
    std::vector<float> m_input_tensor_values; // 预分配的输入向量，避免重复分配

    // --- 模型常量 ---
    const int m_model_input_width = 256;
    const int m_model_input_height = 256;
    const std::vector<double> m_norm_mean = {0.485, 0.456, 0.406};
    const std::vector<double> m_norm_std = {0.229, 0.224, 0.225};
};