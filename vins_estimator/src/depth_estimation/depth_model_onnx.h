/**
 * @file depth_model_onnx.h
 * @brief 深度模型ONNX推理核心（被FastDepthInitializer和BackendDepthConstraint共享）
 *
 * 该类封装了纯粹的ONNX Runtime推理逻辑，不包含异步、TTA等高层功能。
 * 可被多个模块复用。
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <ros/ros.h>

namespace depth_estimation {

/**
 * @enum DepthModelType
 * @brief 支持的深度模型类型
 */
enum class DepthModelType {
    MIDAS_V2,           // MiDaS V2: 256×256, quantile归一化
    DEPTH_ANYTHING_V2   // Depth Anything V2: 518×518, Z-score归一化
};

/**
 * @class DepthModelONNX
 * @brief 深度模型ONNX推理核心
 *
 * 职责：
 * - 加载ONNX模型（CUDA支持）
 * - 单次forward推理
 * - 图像预处理（ImageNet归一化）
 * - 模型特定后处理（Z-score/Quantile归一化）
 * - GPU预热
 *
 * 不负责：
 * - 异步线程管理（由BackendDepthConstraint负责）
 * - TTA增强（由BackendDepthConstraint负责）
 * - 批量处理（由调用者负责）
 * - 图像增强（由调用者负责）
 */
class DepthModelONNX {
public:
    /**
     * @brief 构造函数
     */
    DepthModelONNX();

    /**
     * @brief 析构函数
     */
    ~DepthModelONNX();

    /**
     * @brief 初始化模型
     * @param model_path ONNX模型文件路径
     * @return 初始化成功返回true
     */
    bool init(const std::string& model_path);

    /**
     * @brief 检查模型是否就绪
     * @return 模型已加载且可推理返回true
     */
    bool isReady() const { return model_loaded_; }

    /**
     * @brief 单次推理
     * @param image 输入图像（CV_8UC1或CV_8UC3）
     * @param depth_map 输出深度图（CV_32F，归一化的逆深度）
     * @return 推理成功返回true
     */
    bool predict(const cv::Mat& image, cv::Mat& depth_map);

    /**
     * @brief 获取模型类型
     */
    DepthModelType getModelType() const { return model_type_; }

    /**
     * @brief 获取模型输入尺寸
     */
    int getInputWidth() const { return model_input_width_; }
    int getInputHeight() const { return model_input_height_; }

    /**
     * @brief GPU预热（执行一次虚拟推理）
     */
    void warmup();

private:
    /**
     * @brief 从文件路径检测模型类型
     */
    void detectModelType(const std::string& model_path);

    /**
     * @brief 启用CUDA执行提供器
     */
    bool enableCUDA();

    /**
     * @brief 图像预处理（ImageNet归一化，转NCHW张量）
     */
    void preprocess(const cv::Mat& image, std::vector<float>& tensor_values);

    /**
     * @brief 模型输出后处理（模型特定归一化）
     */
    void postprocess(cv::Mat& depth_map);

    /**
     * @brief Z-score归一化（Depth Anything V2）
     */
    void normalizeZScore(cv::Mat& depth_map);

    /**
     * @brief Quantile归一化（MiDaS V2）
     */
    void normalizeQuantile(cv::Mat& depth_map);

    void normalizeRobust(cv::Mat& depth_map);

private:
    // 模型状态
    bool model_loaded_;
    DepthModelType model_type_;
    int model_input_width_;
    int model_input_height_;

    // ONNX Runtime成员
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // I/O名称
    Ort::AllocatedStringPtr input_name_ptr_;
    Ort::AllocatedStringPtr output_name_ptr_;
    const char* input_name_;
    const char* output_name_;

    // 张量形状
    std::vector<int64_t> input_shape_;
    std::vector<float> input_tensor_values_;

    // ImageNet归一化常量
    static constexpr double NORM_MEAN[3] = {0.485, 0.456, 0.406};
    static constexpr double NORM_STD[3] = {0.229, 0.224, 0.225};
};

} // namespace depth_estimation
