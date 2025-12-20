/**
 * @file online_depth_provider.cpp
 * @brief 在线深度信息提供者实现
 */

#include "online_depth_provider.h"
#include "../utility/tic_toc.h"

namespace depth_estimation {

OnlineDepthProvider::OnlineDepthProvider(std::shared_ptr<DepthModelONNX> depth_model,
                                         int max_queue_size)
    : depth_model_(depth_model)
    , running_(false)
    , max_queue_size_(max_queue_size)
    , processed_count_(0)
    , dropped_count_(0)
{
    if (!depth_model_) {
        ROS_FATAL("[OnlineDepthProvider] Null depth model provided!");
    }
}

OnlineDepthProvider::~OnlineDepthProvider() {
    stop();
}

bool OnlineDepthProvider::start() {
    if (running_.load()) {
        ROS_WARN("[OnlineDepthProvider] Already running");
        return false;
    }

    if (!depth_model_ || !depth_model_->isReady()) {
        ROS_ERROR("[OnlineDepthProvider] Depth model not ready");
        return false;
    }

    // Start inference thread
    running_.store(true);
    inference_thread_ = std::thread(&OnlineDepthProvider::inferenceLoop, this);

    ROS_INFO("[OnlineDepthProvider] Started successfully");
    return true;
}

void OnlineDepthProvider::stop() {
    if (!running_.load()) return;

    running_.store(false);
    queue_cv_.notify_all();

    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }

    ROS_INFO("[OnlineDepthProvider] Stopped. Processed=%d, Dropped=%d",
             processed_count_.load(), dropped_count_.load());
}

bool OnlineDepthProvider::isReady() const {
    return running_.load() && depth_model_ && depth_model_->isReady();
}

bool OnlineDepthProvider::pushImage(double timestamp, const cv::Mat& image) {
    if (!running_.load()) return false;

    std::lock_guard<std::mutex> lock(queue_mutex_);

    // Drop oldest if queue full
    if (static_cast<int>(input_queue_.size()) >= max_queue_size_) {
        input_queue_.pop();
        dropped_count_++;
    }

    input_queue_.emplace(timestamp, image);
    queue_cv_.notify_one();
    return true;
}

bool OnlineDepthProvider::getResult(double timestamp, DepthResult& result) {
    std::lock_guard<std::mutex> lock(result_mutex_);

    auto it = result_buffer_.find(timestamp);
    if (it == result_buffer_.end()) {
        return false;
    }

    result = it->second;
    return result.valid;
}

void OnlineDepthProvider::clearOldResults(double before_time) {
    std::lock_guard<std::mutex> lock(result_mutex_);

    auto it = result_buffer_.begin();
    while (it != result_buffer_.end()) {
        if (it->first < before_time) {
            it = result_buffer_.erase(it);
        } else {
            ++it;
        }
    }
}

void OnlineDepthProvider::enhanceImage(const cv::Mat& image, cv::Mat& enhanced_image) const {
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

void OnlineDepthProvider::inferenceLoop() {
    ROS_INFO("[OnlineDepthProvider] Inference thread started (ID=%ld)", std::this_thread::get_id());

    while (running_.load()) {
        ImageRequest request;

        // Wait for image
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !input_queue_.empty() || !running_.load();
            });

            if (!running_.load()) break;

            request = input_queue_.front();
            input_queue_.pop();
        }

        // Run TTA inference
        TicToc t_inference;
        DepthResult result;
        result.timestamp = request.timestamp;
        result.valid = runTTAInference(request.image, result);

        if (result.valid) {
            // Store result
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_buffer_[request.timestamp] = result;
            }

            processed_count_++;
            ROS_DEBUG("[OnlineDepthProvider] Processed frame %.3f (%.2f ms)",
                     request.timestamp, t_inference.toc());
        }
    }

    ROS_INFO("[OnlineDepthProvider] Inference thread stopped");
}

bool OnlineDepthProvider::runTTAInference(const cv::Mat& image, DepthResult& result) {
    if (!depth_model_ || !depth_model_->isReady()) {
        return false;
    }

    // 1. CLAHE图像增强
    cv::Mat enhanced_image;
    enhanceImage(image, enhanced_image);

    // 2. Run inference on enhanced image
    cv::Mat depth_raw;
    if (!depth_model_->predict(enhanced_image, depth_raw)) {
        return false;
    }

    // 3. Run inference on horizontally flipped enhanced image
    cv::Mat enhanced_flipped;
    cv::flip(enhanced_image, enhanced_flipped, 1);  // Horizontal flip

    cv::Mat depth_flipped_raw;
    if (!depth_model_->predict(enhanced_flipped, depth_flipped_raw)) {
        return false;
    }

    // 4. Flip depth back
    cv::Mat depth_flipped;
    cv::flip(depth_flipped_raw, depth_flipped, 1);

    // 5. TTA aggregation: weighted mean and uncertainty
    // Use weighted average to preserve edge details from the raw image
    // since flip-back interpolation often blurs edges
    const float w_raw = 0.7f;   // Higher weight for raw image (preserves sharpness)
    const float w_flip = 0.3f;  // Lower weight for flipped image
    const float w_sum = w_raw + w_flip;

    result.depth_mean = cv::Mat(depth_raw.size(), CV_32F);
    result.depth_sigma = cv::Mat(depth_raw.size(), CV_32F);

    for (int r = 0; r < depth_raw.rows; ++r) {
        const float* ptr_raw = depth_raw.ptr<float>(r);
        const float* ptr_flip = depth_flipped.ptr<float>(r);
        float* ptr_mean = result.depth_mean.ptr<float>(r);
        float* ptr_sigma = result.depth_sigma.ptr<float>(r);

        for (int c = 0; c < depth_raw.cols; ++c) {
            float d_raw = ptr_raw[c];
            float d_flip = ptr_flip[c];

            // Weighted mean (biased towards raw image for sharpness)
            ptr_mean[c] = (w_raw * d_raw + w_flip * d_flip) / w_sum;

            // Uncertainty based on absolute difference
            // This reflects the disagreement between raw and flipped predictions
            ptr_sigma[c] = std::abs(d_raw - d_flip);
        }
    }

    return true;
}

} // namespace depth_estimation
