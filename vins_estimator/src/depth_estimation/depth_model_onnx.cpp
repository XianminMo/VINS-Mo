/**
 * @file depth_model_onnx.cpp
 * @brief 深度模型ONNX推理核心实现
 */

#include "depth_model_onnx.h"
#include "../utility/tic_toc.h"
#include <algorithm>
#include <numeric>

namespace depth_estimation {

// Initialize static constexpr members
constexpr double DepthModelONNX::NORM_MEAN[3];
constexpr double DepthModelONNX::NORM_STD[3];

DepthModelONNX::DepthModelONNX()
    : model_loaded_(false)
    , model_type_(DepthModelType::MIDAS_V2)
    , model_input_width_(256)
    , model_input_height_(256)
    , input_name_ptr_(nullptr, Ort::detail::AllocatedFree(nullptr))
    , output_name_ptr_(nullptr, Ort::detail::AllocatedFree(nullptr))
    , input_name_(nullptr)
    , output_name_(nullptr)
{
}

DepthModelONNX::~DepthModelONNX() = default;

void DepthModelONNX::detectModelType(const std::string& model_path) {
    if (model_path.find("depth_anything") != std::string::npos ||
        model_path.find("DepthAnything") != std::string::npos) {
        model_type_ = DepthModelType::DEPTH_ANYTHING_V2;
        model_input_width_ = 518;
        model_input_height_ = 518;
        ROS_INFO("[DepthModelONNX] Detected: Depth Anything V2 (518x518, Z-score normalization)");
    } else {
        model_type_ = DepthModelType::MIDAS_V2;
        model_input_width_ = 256;
        model_input_height_ = 256;
        ROS_INFO("[DepthModelONNX] Detected: MiDaS V2 (256x256, quantile normalization)");
    }
}

bool DepthModelONNX::enableCUDA() {
    const OrtApi& ort_api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    auto status_ptr = ort_api.CreateCUDAProviderOptions(&cuda_options);

    if (status_ptr != nullptr) {
        ROS_WARN("[DepthModelONNX] Failed to create CUDA options, falling back to CPU");
        return false;
    }

    status_ptr = ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options_, cuda_options);
    if (status_ptr != nullptr) {
        ROS_WARN("[DepthModelONNX] Failed to append CUDA provider, falling back to CPU");
        ort_api.ReleaseCUDAProviderOptions(cuda_options);
        return false;
    }

    ROS_INFO("[DepthModelONNX] CUDA execution provider enabled");
    ort_api.ReleaseCUDAProviderOptions(cuda_options);
    return true;
}

bool DepthModelONNX::init(const std::string& model_path) {
    TicToc t_init;

    // Detect model type
    detectModelType(model_path);

    // Create ONNX environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "depth_model");

    // Configure session options
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CUDA
    enableCUDA();

    // Load model
    try {
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);
    } catch (const Ort::Exception& e) {
        ROS_ERROR("[DepthModelONNX] Failed to load model: %s", e.what());
        return false;
    }

    // Verify I/O count
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
        ROS_ERROR("[DepthModelONNX] Model must have exactly 1 input and 1 output");
        return false;
    }

    // Get I/O names
    input_name_ptr_ = session_->GetInputNameAllocated(0, allocator_);
    output_name_ptr_ = session_->GetOutputNameAllocated(0, allocator_);
    input_name_ = input_name_ptr_.get();
    output_name_ = output_name_ptr_.get();

    // Prepare tensor shape
    input_shape_ = {1, 3, model_input_height_, model_input_width_};
    input_tensor_values_.resize(1 * 3 * model_input_height_ * model_input_width_);

    model_loaded_ = true;
    ROS_INFO("[DepthModelONNX] Initialized successfully (%.2f ms)", t_init.toc());
    return true;
}

void DepthModelONNX::preprocess(const cv::Mat& image, std::vector<float>& tensor_values) {
    // Convert to RGB if grayscale
    cv::Mat img_rgb;
    if (image.channels() == 1) {
        cv::cvtColor(image, img_rgb, cv::COLOR_GRAY2BGR);
    } else {
        img_rgb = image;
    }

    // Resize to model input size
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(model_input_width_, model_input_height_));

    // Convert to float and normalize to [0, 1]
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // ImageNet normalization to NCHW tensor
    int H = model_input_height_;
    int W = model_input_width_;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float pixel = img_float.at<cv::Vec3f>(h, w)[2 - c];  // BGR->RGB
                tensor_values[c * H * W + h * W + w] = (pixel - NORM_MEAN[c]) / NORM_STD[c];
            }
        }
    }
}

void DepthModelONNX::normalizeZScore(cv::Mat& depth_map) {
    // Collect valid pixels
    std::vector<float> valid_vals;
    valid_vals.reserve(depth_map.total());
    for (int i = 0; i < depth_map.rows * depth_map.cols; ++i) {
        float v = depth_map.at<float>(i);
        if (std::isfinite(v)) {
            valid_vals.push_back(v);
        }
    }

    if (valid_vals.empty()) {
        ROS_WARN("[DepthModelONNX] No valid pixels for Z-score normalization");
        return;
    }

    // Calculate mean and std
    double mean = std::accumulate(valid_vals.begin(), valid_vals.end(), 0.0) / valid_vals.size();
    double sq_sum = 0.0;
    for (float v : valid_vals) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / valid_vals.size());
    std_dev = std::max(std_dev, 1e-6);

    // Z-score normalization: map [-3, +3] to [1, 2]
    for (int i = 0; i < depth_map.rows * depth_map.cols; ++i) {
        float v = depth_map.at<float>(i);
        if (!std::isfinite(v)) v = mean;
        float z = (v - mean) / std_dev;
        z = std::min(std::max(z, -3.0f), 3.0f);
        depth_map.at<float>(i) = 1.0f + (z + 3.0f) / 6.0f;
    }
}

void DepthModelONNX::normalizeQuantile(cv::Mat& depth_map) {
    // Collect valid pixels
    std::vector<float> vals;
    vals.reserve(depth_map.total());
    for (int i = 0; i < depth_map.rows * depth_map.cols; ++i) {
        float v = depth_map.at<float>(i);
        if (std::isfinite(v)) vals.push_back(v);
    }

    if (vals.empty()) {
        ROS_WARN("[DepthModelONNX] No valid pixels for quantile normalization");
        return;
    }

    // Calculate 1% and 99% percentiles
    size_t n = vals.size();
    size_t i1 = std::max<size_t>(0, static_cast<size_t>(0.01 * n) - 1);
    size_t i99 = std::min<size_t>(n - 1, static_cast<size_t>(0.99 * n));
    std::nth_element(vals.begin(), vals.begin() + i1, vals.end());
    float p1 = vals[i1];
    std::nth_element(vals.begin(), vals.begin() + i99, vals.end());
    float p99 = vals[i99];
    if (p99 <= p1) p99 = p1 + 1e-6f;

    // Map [p1, p99] to [1, 2]
    for (int i = 0; i < depth_map.rows * depth_map.cols; ++i) {
        float v = depth_map.at<float>(i);
        if (!std::isfinite(v)) v = p1;
        v = std::min(std::max(v, p1), p99);
        depth_map.at<float>(i) = 1.0f + (v - p1) / (p99 - p1);
    }
}

void DepthModelONNX::normalizeRobust(cv::Mat& depth_map) {
    double min_val, max_val;
    cv::minMaxLoc(depth_map, &min_val, &max_val);

    double range = max_val - min_val;

    // ==========================================
    // [关键] 平坦区域熔断机制 (Flat Region Guard)
    // ==========================================
    // 如果一张图里最大值和最小值差得太小（说明是白墙或严重模糊），
    // 强制归一化会放大噪声。直接跳过处理，或者标记为无效。
    // 这里的 1e-4 取决于网络输出的量级，通常 DepthAnything 输出在 0~1 之间
    if (range < 1e-4) {
        ROS_WARN_THROTTLE(1.0, "[Depth] Flat depth map detected (range=%.5f). Skipping norm to avoid noise amp.", range);
        // 可以选择不做任何处理，或者直接把图置零表示无效
        return; 
    }

    // ==========================================
    // Min-Max Normalization -> [0, 1]
    // ==========================================
    // 线性映射，不改变数据的分布形状（不像 Quantile 或 Z-Score 会扭曲分布）
    // 这对于 RANSAC 线性拟合 (d = s * d_net + t) 是最友好的。
    depth_map.convertTo(depth_map, CV_32F, 1.0 / range, -min_val / range);
    
    // 现在的 depth_map 分布在 [0, 1] 之间
}

void DepthModelONNX::postprocess(cv::Mat& depth_map) {
    if (model_type_ == DepthModelType::DEPTH_ANYTHING_V2) {
        normalizeRobust(depth_map);
    } else {
        normalizeQuantile(depth_map);
    }
}

bool DepthModelONNX::predict(const cv::Mat& image, cv::Mat& depth_map) {
    if (!model_loaded_ || !session_) {
        ROS_ERROR("[DepthModelONNX] Model not loaded");
        return false;
    }

    // Preprocess
    preprocess(image, input_tensor_values_);

    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size());

    // Run inference
    try {
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            &input_name_, &input_tensor, 1,
            &output_name_, 1);

        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat raw_depth(model_input_height_, model_input_width_, CV_32F, output_data);

        // Postprocess
        depth_map = raw_depth.clone();
        postprocess(depth_map);

        return true;
    } catch (const Ort::Exception& e) {
        ROS_ERROR("[DepthModelONNX] Inference failed: %s", e.what());
        return false;
    }
}

void DepthModelONNX::warmup() {
    if (!model_loaded_) return;

    ROS_INFO("[DepthModelONNX] Warming up GPU...");
    TicToc t_warmup;

    cv::Mat dummy_img(model_input_height_, model_input_width_, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat dummy_output;

    predict(dummy_img, dummy_output);

    ROS_INFO("[DepthModelONNX] Warmup completed (%.2f ms)", t_warmup.toc());
}

} // namespace depth_estimation
