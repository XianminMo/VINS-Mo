/**
 * @file init_depth_provider.h
 * @brief 初始化深度信息提供者（用于SFM初始化阶段）
 *
 * 该模块为SFM初始化阶段提供深度信息，
 * 使用共享的DepthModelONNX进行推理，不包含异步和TTA。
 *
 * 职责：
 * - 批量深度图计算
 * - CLAHE图像增强
 * - 尺寸调整
 *
 * 不负责：
 * - ONNX模型加载（由DepthModelONNX负责）
 * - 模型推理细节（由DepthModelONNX负责）
 * - 异步处理（初始化阶段不需要）
 */

#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include "../depth_estimation/depth_model_onnx.h"  // 复用DepthModelONNX

/**
 * @class InitDepthProvider
 * @brief 初始化深度信息提供者（使用共享的DepthModelONNX）
 */
class InitDepthProvider
{
public:
    /**
     * @brief 构造函数
     * @param depth_model 共享的深度模型指针（由外部管理）
     */
    explicit InitDepthProvider(std::shared_ptr<depth_estimation::DepthModelONNX> depth_model);
    ~InitDepthProvider();

    /**
     * @brief 检查模型是否就绪
     */
    bool isReady() const;

    /**
     * @brief 预测深度图（带CLAHE增强）
     * @param image 输入图像
     * @param norm_inv_depth_map 输出深度图（与输入图像同尺寸）
     * @return 推理成功返回true
     */
    bool predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map);

private:
    /**
     * @brief CLAHE图像增强（提升暗部细节）
     * @param image 输入图像
     * @param enhanced_image 增强后的图像
     */
    void enhanceImage(const cv::Mat& image, cv::Mat& enhanced_image) const;

private:
    // 共享的深度模型（由外部管理）
    std::shared_ptr<depth_estimation::DepthModelONNX> depth_model_;
};
