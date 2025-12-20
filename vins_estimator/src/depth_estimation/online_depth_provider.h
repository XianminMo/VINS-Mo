/**
 * @file online_depth_provider.h
 * @brief 在线深度信息提供者（异步推理 + TTA + 图像增强）
 *
 * 该模块提供在线VINS阶段的深度信息，采用异步推理和TTA增强，
 * 并复用DepthModelONNX进行实际推理。
 *
 * 职责：
 * - 异步线程管理
 * - TTA（水平翻转增强）
 * - 生产者-消费者队列
 * - 结果缓存（按时间戳）
 * - CLAHE图像增强
 *
 * 不负责：
 * - ONNX模型加载（由DepthModelONNX负责）
 * - 模型推理细节（由DepthModelONNX负责）
 */

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "depth_model_onnx.h"

namespace depth_estimation {

/**
 * @struct DepthResult
 * @brief 深度推理结果（包含TTA不确定性）
 */
struct DepthResult {
    double timestamp;
    cv::Mat depth_mean;   // CV_32F, TTA平均深度图
    cv::Mat depth_sigma;  // CV_32F, TTA不确定性
    bool valid;

    DepthResult() : timestamp(0.0), valid(false) {}
};

/**
 * @struct ImageRequest
 * @brief 深度推理请求
 */
struct ImageRequest {
    double timestamp;
    cv::Mat image;

    ImageRequest() : timestamp(0.0) {}
    ImageRequest(double t, const cv::Mat& img) : timestamp(t), image(img.clone()) {}
};

/**
 * @class OnlineDepthProvider
 * @brief 在线深度信息提供者（使用共享的DepthModelONNX）
 */
class OnlineDepthProvider {
public:
    /**
     * @brief 构造函数
     * @param depth_model 共享的深度模型指针（由外部管理）
     * @param max_queue_size 输入队列最大长度
     */
    explicit OnlineDepthProvider(std::shared_ptr<DepthModelONNX> depth_model,
                                 int max_queue_size = 5);
    ~OnlineDepthProvider();

    /**
     * @brief 启动异步推理线程
     */
    bool start();

    /**
     * @brief 停止异步推理线程
     */
    void stop();

    /**
     * @brief 检查是否就绪
     */
    bool isReady() const;

    /**
     * @brief 推送图像（非阻塞）
     */
    bool pushImage(double timestamp, const cv::Mat& image);

    /**
     * @brief 获取结果（按时间戳查询）
     */
    bool getResult(double timestamp, DepthResult& result);

    /**
     * @brief 清理旧结果
     */
    void clearOldResults(double before_time);

private:
    /**
     * @brief 异步推理循环
     */
    void inferenceLoop();

    /**
     * @brief TTA推理（水平翻转增强）
     */
    bool runTTAInference(const cv::Mat& image, DepthResult& result);

    /**
     * @brief CLAHE图像增强
     */
    void enhanceImage(const cv::Mat& image, cv::Mat& enhanced_image) const;

private:
    // 共享的深度模型（由外部管理）
    std::shared_ptr<DepthModelONNX> depth_model_;

    // 线程管理
    std::thread inference_thread_;
    std::atomic<bool> running_;

    // 输入队列
    std::queue<ImageRequest> input_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    int max_queue_size_;

    // 输出缓存
    std::map<double, DepthResult> result_buffer_;
    std::mutex result_mutex_;

    // 统计信息
    std::atomic<int> processed_count_;
    std::atomic<int> dropped_count_;
};

} // namespace depth_estimation
