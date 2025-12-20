#include "init_depth_provider.h"

InitDepthProvider::InitDepthProvider(std::shared_ptr<depth_estimation::DepthModelONNX> depth_model)
    : depth_model_(depth_model)
{
    if (!depth_model_) {
        ROS_FATAL("[InitDepthProvider] Null depth model provided!");
    }
}

InitDepthProvider::~InitDepthProvider() = default;

bool InitDepthProvider::isReady() const {
    return depth_model_ && depth_model_->isReady();
}

void InitDepthProvider::enhanceImage(const cv::Mat& image, cv::Mat& enhanced_image) const {
    // CLAHE 直方图均衡化增强（提升暗部细节）
    if (image.channels() == 1) {
        // 灰度图：直接CLAHE
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(image, enhanced_image);
    } else if (image.channels() == 3) {
        // 彩色图：在LAB空间对L通道CLAHE
        cv::Mat lab_image;
        cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> lab_channels;
        cv::split(lab_image, lab_channels);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);

        cv::merge(lab_channels, lab_image);
        cv::cvtColor(lab_image, enhanced_image, cv::COLOR_Lab2BGR);
    } else {
        // 其他情况直接复制
        enhanced_image = image.clone();
    }
}

bool InitDepthProvider::predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map) {
    if (!isReady()) {
        ROS_ERROR("[InitDepthProvider] Model not ready");
        return false;
    }

    // 1. CLAHE图像增强（InitDepthProvider特有功能）
    cv::Mat enhanced_image;
    enhanceImage(image, enhanced_image);

    // 2. 调用共享的DepthModelONNX进行推理
    bool success = depth_model_->predict(enhanced_image, norm_inv_depth_map);

    if (!success) {
        ROS_ERROR("[InitDepthProvider] Prediction failed");
        return false;
    }

    // 3. 调整输出尺寸回原始图像大小
    if (norm_inv_depth_map.size() != image.size()) {
        cv::Mat resized_depth;
        cv::resize(norm_inv_depth_map, resized_depth, image.size(), 0, 0, cv::INTER_LINEAR);
        norm_inv_depth_map = resized_depth;
    }

    return true;
}
