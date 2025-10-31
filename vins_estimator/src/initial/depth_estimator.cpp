#include "depth_estimator.h"

// 构造函数：初始化所有 C++ 风格的对象
DepthEstimator::DepthEstimator()
    : m_env(ORT_LOGGING_LEVEL_WARNING, "vins_depth_estimator"),
      m_session_options(),
      m_session(nullptr),
      m_input_name_ptr(nullptr, Ort::detail::AllocatedFree(nullptr)),
      m_output_name_ptr(nullptr, Ort::detail::AllocatedFree(nullptr)),

      m_input_name{nullptr},
      m_output_name{nullptr}
{}

// 析构函数 (默认即可，std::unique_ptr 会自动管理 m_session)
DepthEstimator::~DepthEstimator()
{
}

bool DepthEstimator::init(const std::string& model_path)
{
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

// 预处理函数 
void DepthEstimator::preprocess(const cv::Mat& image, std::vector<float>& input_tensor_values)
{
    // 1) 灰度→BGR；BGR→RGB，确保模型看到的是 RGB 顺序
    cv::Mat img_bgr;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, img_bgr, cv::COLOR_GRAY2BGR);
    }
    else if (image.channels() == 3)
    {
        img_bgr = image;
    }
    else
    {
        ROS_WARN("DepthEstimator::preprocess(): unsupported channels=%d, trying to proceed.", image.channels());
        img_bgr = image;
    }

    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    // 2) 缩放到模型输入尺寸，浮点化并归一化到 [0,1]
    cv::Mat resized_image;
    cv::resize(img_rgb, resized_image, cv::Size(m_model_input_width, m_model_input_height), 0, 0, cv::INTER_CUBIC);
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
    if (!m_session)
    {
        ROS_ERROR("DepthEstimator::predict() failed: session is not initialized.");
        return false;
    }

    // --- 1. 预处理 ---
    preprocess(image, m_input_tensor_values);

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
        ROS_INFO("MiDaS raw_inv_depth: min=%.6f max=%.6f mean=%.6f", raw_min, raw_max, raw_mean[0]);

        // 4.3 归一化到 [1, 2]（按你的设置保留）
        cv::Mat normalized_float_map;
        cv::normalize(raw_inv_depth, normalized_float_map, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

        // 4.4 统计 norm 的 min/max/mean
        double n_min = 0.0, n_max = 0.0;
        cv::minMaxLoc(normalized_float_map, &n_min, &n_max);
        cv::Scalar n_mean = cv::mean(normalized_float_map);
        ROS_INFO("MiDaS norm_inv_depth: min=%.6f max=%.6f mean=%.6f", n_min, n_max, n_mean[0]);

        // 4.5 保存首帧的可视化图（只保存一次，便于人工确认是否“扁平”）
        static bool dumped = false;
        if (!dumped)
        {
            cv::Mat raw_vis_8u, norm_vis_8u;
            cv::normalize(raw_inv_depth, raw_vis_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::normalize(normalized_float_map, norm_vis_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite("/tmp/first_frame_raw_inv_depth.png", raw_vis_8u);
            cv::imwrite("/tmp/first_frame_norm_inv_depth.png", norm_vis_8u);
            ROS_INFO("Saved MiDaS debug images to /tmp/first_frame_raw_inv_depth.png and /tmp/first_frame_norm_inv_depth.png");
            dumped = true;
        }

        // 4.6 缩放回原图尺寸
        cv::resize(normalized_float_map, norm_inv_depth_map, image.size(), 0, 0, cv::INTER_LINEAR);
    }
    catch (const Ort::Exception& e)
    {
        ROS_ERROR("ONNX: Inference runtime failed: %s", e.what());
        return false;
    }

    return true;
}