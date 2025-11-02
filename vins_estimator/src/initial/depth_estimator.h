#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <ros/ros.h>

class DepthEstimator
{
public:
    DepthEstimator();
    ~DepthEstimator();

    /**
     * @brief 同步初始化（保持向后兼容）
     */
    bool init(const std::string& model_path);

    /**
     * @brief 异步初始化（不阻塞）
     * @param model_path 模型路径
     * @return true 如果异步初始化已启动
     */
    bool initAsync(const std::string& model_path);

    /**
     * @brief 检查模型是否已就绪
     */
    bool isReady() const { return m_is_ready.load(); }

    /**
     * @brief 等待模型加载完成（带超时）
     * @param timeout_ms 超时时间（毫秒），-1表示无限等待
     * @return true 如果模型已就绪
     */
    bool waitForReady(int timeout_ms = -1) const;

    /**
     * @brief 预热模型（执行一次虚拟推理）
     */
    void warmup();

    bool predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map);

private:
    void preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, bool save_debug_images = true);

    /**
     * @brief 内部预测方法（用于 warmup，不检查就绪状态）
     */
    bool predictInternal(const cv::Mat& image, cv::Mat& norm_inv_depth_map, bool save_debug_images = true);
    
    // 异步初始化的工作函数
    void initWorker(const std::string& model_path);

    // --- ONNX Runtime 核心成员 ---
    Ort::Env m_env;
    Ort::SessionOptions m_session_options;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    // --- 异步初始化相关 ---
    std::atomic<bool> m_is_ready{false};
    std::atomic<bool> m_init_failed{false};
    std::mutex m_init_mutex;
    std::unique_ptr<std::thread> m_init_thread;

    // --- 模型输入/输出信息 ---
    Ort::AllocatedStringPtr m_input_name_ptr;
    Ort::AllocatedStringPtr m_output_name_ptr;
    const char* m_input_name;
    const char* m_output_name;

    std::vector<int64_t> m_input_shape;
    std::vector<float> m_input_tensor_values;

    const int m_model_input_width = 256;
    const int m_model_input_height = 256;
    const std::vector<double> m_norm_mean = {0.485, 0.456, 0.406};
    const std::vector<double> m_norm_std = {0.229, 0.224, 0.225};
};