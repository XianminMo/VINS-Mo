#include "depth_estimator.h"
#include "../utility/tic_toc.h" 

// 构造函数：初始化所有 C++ 风格的对象
DepthEstimator::DepthEstimator()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "vins_depth_estimator"),
      m_session_options(),
      m_session(nullptr),
      m_input_name_ptr(nullptr, Ort::detail::AllocatedFree(nullptr)),
      m_output_name_ptr(nullptr, Ort::detail::AllocatedFree(nullptr)),

      m_input_name{nullptr},
      m_output_name{nullptr},
      m_model_type(DepthModelType::MIDAS_V2),  // 默认值，会在 init 中根据模型路径自动检测
      m_model_input_width(256),
      m_model_input_height(256)
{}

// 析构函数 (默认即可，std::unique_ptr 会自动管理 m_session)
DepthEstimator::~DepthEstimator()
{
    if (m_init_thread && m_init_thread->joinable()) {
        m_init_thread->join();
    }
}

// 模型类型自动检测函数
DepthModelType DepthEstimator::detectModelType(const std::string& model_path) const
{
    // 根据模型文件名自动检测模型类型
    if (model_path.find("depth_anything") != std::string::npos ||
        model_path.find("DepthAnything") != std::string::npos) {
        return DepthModelType::DEPTH_ANYTHING_V2;
    } else if (model_path.find("midas") != std::string::npos ||
               model_path.find("Midas") != std::string::npos ||
               model_path.find("MiDaS") != std::string::npos) {
        return DepthModelType::MIDAS_V2;
    } else {
        // 默认为 MiDaS V2
        ROS_WARN("Cannot auto-detect model type from path: %s. Defaulting to MiDaS V2.", model_path.c_str());
        return DepthModelType::MIDAS_V2;
    }
}

// 图像预处理增强（直方图均衡化等）
void DepthEstimator::enhanceImage(const cv::Mat& image, cv::Mat& enhanced_image) const
{
    // 如果是灰度图，直接进行CLAHE
    if (image.channels() == 1) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(image, enhanced_image);
    }
    // 如果是彩色图，转换到LAB空间，对L通道进行CLAHE
    else if (image.channels() == 3) {
        cv::Mat lab_image;
        cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

        // 分离LAB通道
        std::vector<cv::Mat> lab_channels;
        cv::split(lab_image, lab_channels);

        // 对L通道进行CLAHE
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);

        // 合并通道
        cv::merge(lab_channels, lab_image);

        // 转换回BGR
        cv::cvtColor(lab_image, enhanced_image, cv::COLOR_Lab2BGR);
    } else {
        // 其他情况直接复制
        enhanced_image = image.clone();
    }
}

bool DepthEstimator::init(const std::string& model_path)
{
    // --- 0. 自动检测模型类型并配置参数 ---
    m_model_type = detectModelType(model_path);

    if (m_model_type == DepthModelType::DEPTH_ANYTHING_V2) {
        m_model_input_width = 518;
        m_model_input_height = 518;
        ROS_INFO("Detected model type: Depth Anything V2 (input: 518x518, output: raw inverse depth values)");
    } else {
        m_model_input_width = 256;
        m_model_input_height = 256;
        ROS_INFO("Detected model type: MiDaS V2 (input: 256x256, normalization: Quantile-clipped [1%%-99%%] -> [1,2])");
    }

    // --- 1. 设置会话选项 (来自 Demo) ---
    m_session_options.SetIntraOpNumThreads(1);
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // --- 2. (核心) 启用 CUDA (来自 Demo 的 V2 C-API 绕行方案) ---
    const OrtApi& ort_api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_options_v2 = nullptr;
    auto status_ptr = ort_api.CreateCUDAProviderOptions(&cuda_options_v2);

    if (status_ptr != nullptr)
    {
        auto status = Ort::Status(status_ptr);
        ROS_WARN("Failed to create CUDA V2 default options: %s. Falling back to CPU execution.", status.GetErrorMessage().c_str());
    }
    else
    {
        status_ptr = ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(m_session_options, cuda_options_v2);
        if(status_ptr != nullptr)
        {
            auto status = Ort::Status(status_ptr);
            ROS_WARN("Failed to append CUDA V2 Provider: %s. Falling back to CPU execution.", status.GetErrorMessage().c_str());
        }
        else
        {
            ROS_INFO("Successfully requested CUDA V2 execution provider.");
        }
        ort_api.ReleaseCUDAProviderOptions(cuda_options_v2);
    }

    // --- 3. 创建会话并加载模型 ---
    try
    {
        m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), m_session_options);
    }
    catch (const Ort::Exception& e)
    {
        ROS_ERROR("ONNX: Failed to load model: %s", e.what());
        return false;
    }

    // --- 4. 获取模型输入/输出信息 (来自 Demo) ---
    if (m_session->GetInputCount() != 1 || m_session->GetOutputCount() != 1)
    {
        ROS_ERROR("ONNX: The model must have exactly 1 input and 1 output.");
        return false;
    }

    // --- 5. 获取模型输入/输出信息 (来自 Demo) ---
    // 使用会话自己的分配器来管理输入/输出名称的内存，这是更安全的方式。
    m_input_name_ptr = m_session->GetInputNameAllocated(0, m_allocator);
    m_output_name_ptr = m_session->GetOutputNameAllocated(0, m_allocator);

    m_input_name = m_input_name_ptr.get(); // get() 返回原始指针
    m_output_name = m_output_name_ptr.get();
    
    m_input_shape = {1, 3, m_model_input_height, m_model_input_width};
    
    // 预分配输入张量的内存
    m_input_tensor_values.resize(1 * 3 * m_model_input_height * m_model_input_width);

    ROS_INFO("DepthEstimator initialized successfully. Model: %s", model_path.c_str());
    ROS_INFO("Model input: '%s', Model output: '%s'", m_input_name, m_output_name);
    return true;
}

// 异步初始化方法
bool DepthEstimator::initAsync(const std::string& model_path)
{
    if (m_is_ready.load()) {
        ROS_WARN("DepthEstimator::initAsync(): Model already initialized.");
        return true;
    }

    if (m_init_thread && m_init_thread->joinable()) {
        ROS_WARN("DepthEstimator::initAsync(): Initialization already in progress.");
        return false;
    }

    m_init_failed = false;
    m_init_thread = std::make_unique<std::thread>(&DepthEstimator::initWorker, this, model_path);
    
    ROS_INFO("DepthEstimator::initAsync(): Started asynchronous initialization.");
    return true;
}

// 添加工作线程函数
void DepthEstimator::initWorker(const std::string& model_path)
{
    ROS_INFO("DepthEstimator::initWorker(): Loading model in background thread...");
    TicToc t_init;
    
    bool success = init(model_path);
    
    if (success) {
        // 预热模型
        warmup();
        m_is_ready.store(true);
        ROS_INFO("DepthEstimator::initWorker(): Model loaded and warmed up successfully (%.2f ms).", t_init.toc());
    } else {
        m_init_failed.store(true);
        ROS_ERROR("DepthEstimator::initWorker(): Model loading failed.");
    }
}

// 添加等待方法
bool DepthEstimator::waitForReady(int timeout_ms) const
{
    if (m_is_ready.load()) {
        return true;
    }

    if (timeout_ms < 0) {
        // 无限等待
        while (!m_is_ready.load() && !m_init_failed.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return m_is_ready.load();
    } else {
        // 有限等待
        auto start = std::chrono::steady_clock::now();
        while (!m_is_ready.load() && !m_init_failed.load()) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            if (elapsed_ms >= timeout_ms) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return m_is_ready.load();
    }
}

// 添加预热方法
void DepthEstimator::warmup()
{
    if (!m_session) {
        return;
    }
    TicToc t_warmup;
    
    // 创建一个小的虚拟图像进行推理
    cv::Mat dummy_img(m_model_input_height, m_model_input_width, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat dummy_output;
    
    (void)predictInternal(dummy_img, dummy_output, false, true);
}

// 预处理函数
void DepthEstimator::preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values, bool save_debug_images)
{
    static bool first_preprocess = true;
    if (first_preprocess && save_debug_images) {
        ROS_INFO("DepthEstimator::preprocess(): Input image - channels: %d, type: %d, size: %dx%d",
                 image.channels(), image.type(), image.cols, image.rows);
        first_preprocess = false;
    }

    // 0) 图像增强：使用CLAHE提升暗部细节
    cv::Mat enhanced_image;
    enhanceImage(image, enhanced_image);

    // 1) 灰度→BGR；BGR→RGB，确保模型看到的是 RGB 顺序
    cv::Mat img_bgr;
    if (enhanced_image.channels() == 1)
    {
        cv::cvtColor(enhanced_image, img_bgr, cv::COLOR_GRAY2BGR);
    }
    else if (enhanced_image.channels() == 3)
    {
        img_bgr = enhanced_image;

        // 仅在启用调试时检查伪彩色（只检查一次，避免重复警告）
        if (save_debug_images) {
            static bool checked_pseudo_color = false;
            if (!checked_pseudo_color) {
                cv::Mat channels[3];
                cv::split(img_bgr, channels);
                cv::Mat diff1, diff2;
                cv::absdiff(channels[0], channels[1], diff1);
                cv::absdiff(channels[0], channels[2], diff2);
                double max_diff1, max_diff2;
                cv::minMaxLoc(diff1, nullptr, &max_diff1);
                cv::minMaxLoc(diff2, nullptr, &max_diff2);

                if (max_diff1 < 5.0 && max_diff2 < 5.0) {
                    ROS_WARN("DepthEstimator: Input appears to be pseudo-color (grayscale converted to BGR). "
                             "All channels are nearly identical. Depth estimation may be affected!");
                }
                checked_pseudo_color = true;
            }
        }
    }

    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    // 2) 缩放到模型输入尺寸，浮点化并归一化到 [0,1]
    cv::Mat resized_image;
    cv::resize(img_rgb, resized_image, cv::Size(m_model_input_width, m_model_input_height), 0, 0, cv::INTER_LINEAR);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // 3) 按 NCHW 填充，并用 ImageNet 均值方差做标准化（RGB 顺序）
    const int H = m_model_input_height;
    const int W = m_model_input_width;
    const size_t channel_step = static_cast<size_t>(H) * static_cast<size_t>(W);

    float* output_ptr_r = input_tensor_values.data();
    float* output_ptr_g = input_tensor_values.data() + channel_step;
    float* output_ptr_b = input_tensor_values.data() + (2 * channel_step);

    for (int i = 0; i < H; ++i)
    {
        const cv::Vec3f* row = resized_image.ptr<cv::Vec3f>(i); // row[j] = [R,G,B]
        for (int j = 0; j < W; ++j)
        {
            const float r = row[j][0];
            const float g = row[j][1];
            const float b = row[j][2];
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(W) + static_cast<size_t>(j);
            output_ptr_r[idx] = static_cast<float>((r - m_norm_mean[0]) / m_norm_std[0]);
            output_ptr_g[idx] = static_cast<float>((g - m_norm_mean[1]) / m_norm_std[1]);
            output_ptr_b[idx] = static_cast<float>((b - m_norm_mean[2]) / m_norm_std[2]);
        }
    }
}

// 预测函数
bool DepthEstimator::predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map)
{
    if (!m_is_ready.load()) {
        ROS_ERROR("DepthEstimator::predict() failed: model is not ready yet.");
        return false;
    }

    // 调用内部实现
    return predictInternal(image, norm_inv_depth_map, true);
}

// 预测内部实现
bool DepthEstimator::predictInternal(const cv::Mat& image, cv::Mat& norm_inv_depth_map, bool save_debug_images, bool quiet)
{
    TicToc t_infer;
    static bool saved_input = false;
    if (!saved_input && save_debug_images) {
        cv::imwrite("/tmp/first_frame_input_image.png", image);
        if (!quiet) ROS_INFO("Saved input image to /tmp/first_frame_input_image.png (channels: %d)", image.channels());
        saved_input = true;
    }

    if (!m_session)
    {
        ROS_ERROR("DepthEstimator::predict() failed: session is not initialized.");
        return false;
    }

    // --- 1. 预处理 ---
    preprocess(image, m_input_tensor_values, save_debug_images);

    // --- 2. 创建输入张量 ---
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        m_input_tensor_values.data(),
        m_input_tensor_values.size(),
        m_input_shape.data(),
        m_input_shape.size());

    // --- 3. 执行推理 ---
    try
    {
        auto output_tensors = m_session->Run(Ort::RunOptions{nullptr},
                                             &m_input_name, &input_tensor, 1,
                                             &m_output_name, 1);

        // --- 4. 获取输出并后处理 ---
        float* output_data_ptr = output_tensors[0].GetTensorMutableData<float>();

        // 4.1 封装为 Mat（H x W, CV_32F）
        cv::Mat raw_inv_depth(m_model_input_height, m_model_input_width, CV_32F, output_data_ptr);

        // 4.2 统计 raw 的 min/max/mean
        double raw_min = 0.0, raw_max = 0.0;
        cv::minMaxLoc(raw_inv_depth, &raw_min, &raw_max);
        cv::Scalar raw_mean = cv::mean(raw_inv_depth);
        if (!quiet) ROS_DEBUG("Raw inv_depth: min=%.6f max=%.6f mean=%.6f", raw_min, raw_max, raw_mean[0]);

        // 4.3 根据模型类型选择归一化策略
        cv::Mat normalized_float_map;
        {
            if (m_model_type == DepthModelType::DEPTH_ANYTHING_V2) {
                // Depth Anything V2: 使用帧内标准化，避免连续帧之间的全局尺度跳变
                // 策略：对每帧进行 Z-score 标准化（零均值，单位方差），然后映射到 [1, 2] 范围
                // 这样可以保持帧内的相对深度关系，同时消除帧间的全局尺度差异

                // 计算有效像素的均值和标准差
                std::vector<float> valid_vals;
                valid_vals.reserve(static_cast<size_t>(raw_inv_depth.total()));
                for (int r = 0; r < raw_inv_depth.rows; ++r) {
                    const float* ptr = raw_inv_depth.ptr<float>(r);
                    for (int c = 0; c < raw_inv_depth.cols; ++c) {
                        float v = ptr[c];
                        if (std::isfinite(v)) {
                            valid_vals.push_back(v);
                        }
                    }
                }

                if (!valid_vals.empty()) {
                    // 计算均值和标准差
                    double sum = 0.0;
                    for (float v : valid_vals) {
                        sum += v;
                    }
                    double mean = sum / valid_vals.size();

                    double sq_sum = 0.0;
                    for (float v : valid_vals) {
                        double diff = v - mean;
                        sq_sum += diff * diff;
                    }
                    double std_dev = std::sqrt(sq_sum / valid_vals.size());

                    // 避免除零
                    if (std_dev < 1e-6) {
                        std_dev = 1e-6;
                    }

                    // Z-score 标准化，然后映射到 [1, 2]
                    // z = (x - mean) / std_dev
                    // 将 z 的 [-3, +3] 范围映射到 [1, 2]（覆盖 99.7% 的数据）
                    normalized_float_map.create(raw_inv_depth.size(), CV_32F);
                    for (int r = 0; r < raw_inv_depth.rows; ++r) {
                        const float* src = raw_inv_depth.ptr<float>(r);
                        float* dst = normalized_float_map.ptr<float>(r);
                        for (int c = 0; c < raw_inv_depth.cols; ++c) {
                            float v = src[c];
                            if (!std::isfinite(v)) {
                                v = static_cast<float>(mean);
                            }
                            // Z-score
                            float z = static_cast<float>((v - mean) / std_dev);
                            // 裁剪到 [-3, +3]
                            z = std::min(std::max(z, -3.0f), 3.0f);
                            // 映射到 [1, 2]: (z + 3) / 6 * 1 + 1
                            dst[c] = 1.0f + (z + 3.0f) / 6.0f;
                        }
                    }

                    if (!quiet) {
                        ROS_DEBUG("Depth Anything V2: Z-score normalization (mean=%.6f, std=%.6f) -> [1, 2]",
                                 mean, std_dev);
                    }
                } else {
                    // 如果没有有效值，直接使用原始值
                    normalized_float_map = raw_inv_depth.clone();
                    if (!quiet) {
                        ROS_WARN("Depth Anything V2: No valid pixels, using raw values");
                    }
                }
            } else {
                // MiDaS V2: 使用分位数裁剪增强鲁棒性（1%~99%），然后归一化到 [1, 2]
                std::vector<float> vals;
                vals.reserve(static_cast<size_t>(raw_inv_depth.total()));
                for (int r = 0; r < raw_inv_depth.rows; ++r) {
                    const float* ptr = raw_inv_depth.ptr<float>(r);
                    for (int c = 0; c < raw_inv_depth.cols; ++c) {
                        float v = ptr[c];
                        if (std::isfinite(v)) vals.push_back(v);
                    }
                }
                if (!vals.empty()) {
                    size_t n = vals.size();
                    size_t i1 = static_cast<size_t>(std::max<size_t>(0, static_cast<size_t>(0.01 * n) - 1));
                    size_t i99 = static_cast<size_t>(std::min<size_t>(n - 1, static_cast<size_t>(0.99 * n)));
                    std::nth_element(vals.begin(), vals.begin() + i1, vals.end());
                    float p1 = vals[i1];
                    std::nth_element(vals.begin(), vals.begin() + i99, vals.end());
                    float p99 = vals[i99];
                    if (p99 <= p1) { p99 = p1 + 1e-6f; }

                    normalized_float_map.create(raw_inv_depth.size(), CV_32F);
                    for (int r = 0; r < raw_inv_depth.rows; ++r) {
                        const float* src = raw_inv_depth.ptr<float>(r);
                        float* dst = normalized_float_map.ptr<float>(r);
                        for (int c = 0; c < raw_inv_depth.cols; ++c) {
                            float v = src[c];
                            if (!std::isfinite(v)) { v = p1; }
                            v = std::min(std::max(v, p1), p99);
                            // 线性映射到 [1, 2]
                            dst[c] = 1.0f + (v - p1) * (1.0f / (p99 - p1));
                        }
                    }

                    if (!quiet) ROS_DEBUG("MiDaS V2: Quantile-clipped normalization to [1, 2] (p1%%=%.6f, p99%%=%.6f)", p1, p99);
                } else {
                    cv::normalize(raw_inv_depth, normalized_float_map, 1.0, 2.0, cv::NORM_MINMAX, CV_32F);
                }
            }
        }

        // 4.4 统计 norm 的 min/max/mean
        double n_min = 0.0, n_max = 0.0;
        cv::minMaxLoc(normalized_float_map, &n_min, &n_max);
        cv::Scalar n_mean = cv::mean(normalized_float_map);
        if (!quiet) {
            if (m_model_type == DepthModelType::DEPTH_ANYTHING_V2) {
                ROS_DEBUG("Depth Anything V2 output: min=%.6f max=%.6f mean=%.6f", n_min, n_max, n_mean[0]);
            } else {
                ROS_DEBUG("MiDaS V2 norm_inv_depth: min=%.6f max=%.6f mean=%.6f", n_min, n_max, n_mean[0]);
            }
        }

        // 4.5 缩放回原图尺寸
        cv::resize(normalized_float_map, norm_inv_depth_map, image.size(), 0, 0, cv::INTER_LINEAR);

        // 4.6 保存首帧的可视化图（注释掉，仅在需要调试时启用）
        // static bool dumped = false;
        // if (!dumped && save_debug_images)
        // {
        //     ... 保存调试图像的代码 ...
        // }

        // 4.7 有效性与汇总输出（简化版本）
        cv::Mat isfinite_mask = norm_inv_depth_map == norm_inv_depth_map; // NaN 检测
        int finite_count = cv::countNonZero(isfinite_mask);
        int total_count = norm_inv_depth_map.rows * norm_inv_depth_map.cols;
        double finite_ratio = (total_count > 0) ? (100.0 * static_cast<double>(finite_count) / static_cast<double>(total_count)) : 0.0;

        // 只在首次或异常情况下输出
        static int predict_count = 0;
        predict_count++;

        if (!quiet)
        {
            // 首次预测时输出完整信息
            if (predict_count == 1)
            {
                ROS_INFO("[DepthEstimator] First prediction: %.1f ms, size=%dx%d, finite=%.1f%%",
                         t_infer.toc(), norm_inv_depth_map.cols, norm_inv_depth_map.rows, finite_ratio);
            }
            // 后续只在有问题时输出
            else if (finite_ratio < 95.0)
            {
                ROS_WARN("[DepthEstimator] Low finite ratio: %.1f%% at frame %d", finite_ratio, predict_count);
            }
        }
    }
    catch (const Ort::Exception& e)
    {
        ROS_ERROR("ONNX: Inference runtime failed: %s", e.what());
        return false;
    }

    return true;
}