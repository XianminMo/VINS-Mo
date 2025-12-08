# VINS-Mo 后端深度约束架构文档

## 目录
1. [概述](#概述)
2. [核心思想与理论基础](#核心思想与理论基础)
3. [系统架构](#系统架构)
4. [深度融合策略](#深度融合策略)
5. [参数管理机制](#参数管理机制)
6. [实现细节](#实现细节)
7. [关键代码位置](#关键代码位置)

---

## 概述

VINS-Mo 的后端深度约束系统将单目深度学习网络（如 MiDaS V2 和 Depth Anything V2）的预测深度引入 VINS 后端优化，通过仿射变换模型对齐网络深度与度量深度，实现尺度约束和轨迹精度提升。

### 核心特性
- **模型支持**: 自动检测并适配 MiDaS V2 (256×256) 和 Depth Anything V2 (518×518)
- **在线对齐**: 初始化完成后自动估计深度仿射参数 (a, b)
- **自适应权重**: 基于 IMU 陀螺仪和加速度计的物理感知权重调节
- **预热机制**: 前 N 帧使用宽松 Huber 损失快速收敛
- **随机游走约束**: 平滑参数演化，防止剧烈跳变

---

## 核心思想与理论基础

### 1. 深度仿射变换模型

深度学习网络输出的**归一化逆深度** `d_nn` 与 VINS 度量**逆深度** `1/d_metric` 之间存在仿射关系：

```
1/d_metric = a * d_nn + b
```

- **a**: 尺度因子（scale），反映深度网络输出与真实度量尺度的比例
- **b**: 偏移因子（shift），补偿网络输出的系统偏差

### 2. 残差定义

深度因子（`DepthFactor`）的残差构建如下：

```cpp
residual = sqrt_info * ((a * d_nn + b) - (1 / d_metric_j))
```

其中：
- `d_nn`: 从深度图读取的网络预测逆深度（在特征点像素位置）
- `d_metric_j`: VINS 通过特征三角化计算的度量深度（在观测帧 j）
- `sqrt_info`: 信息矩阵平方根（权重），控制约束强度

### 3. 优化框架

在 Ceres 后端优化中，深度因子连接以下参数块：

| 参数块 | 维度 | 含义 |
|--------|------|------|
| `para_Pose_i` | 7 | 特征首次观测帧位姿 [P, Q] |
| `para_Pose_j` | 7 | 深度图所在帧位姿 [P, Q] |
| `para_Ex_Pose` | 7 | 相机-IMU 外参 [t_ic, q_ic] |
| `para_Feature` | 1 | 特征点逆深度 λ_k |
| `para_DepthScaleShift` | 2 | 深度仿射参数 [a, b] |

优化目标：
```
min Σ ρ(||residual||²)  over all features with depth measurements
```

其中 `ρ` 为鲁棒核函数（Huber Loss）。

---

## 系统架构

### 整体工作流程

```
[1] 图像输入 → [2] 深度推理 → [3] 参数估计 → [4] 后端优化 → [5] 参数更新
    ↓               ↓               ↓               ↓               ↓
原始图像     depth_map        a, b 初值      Ceres Solve      a, b 优化值
  ↓            (CV_32F)        (线性回归)    (非线性优化)    (随机游走平滑)
异步推理      逆深度值          对齐VINS       多因子约束        历史约束
```

### 关键模块

#### 1. DepthEstimator（深度推理模块）
**位置**: `vins_estimator/src/initial/depth_estimator.{h,cpp}`

**功能**:
- **异步初始化**: 后台加载 ONNX 模型（`initAsync`），避免阻塞 VIO
- **模型类型检测**: 自动识别 MiDaS 或 Depth Anything V2
- **图像增强**: CLAHE 直方图均衡化提升暗部细节
- **归一化策略**:
  - **MiDaS V2**: 分位数裁剪（1%-99%）+ 线性归一化到 [1, 2]
  - **Depth Anything V2**: 保留原始逆深度值（无裁剪归一化）

**关键 API**:
```cpp
bool initAsync(const std::string& model_path);  // 异步加载模型
bool isReady();                                 // 检查模型就绪
bool predict(const cv::Mat& image, cv::Mat& norm_inv_depth_map);  // 推理
```

#### 2. Estimator（核心估计器）
**位置**: `vins_estimator/src/estimator.{h,cpp}`

**深度约束相关函数**:

| 函数名 | 调用时机 | 功能 |
|--------|----------|------|
| `ensureDepthMapForAlignment` | 初始化完成后 | 确保至少有一帧深度图用于参数对齐 |
| `estimateDepthScaleShift` | 初始化完成后 | 线性回归计算 a, b 初值 |
| `optimization` | 每帧后端优化 | 添加深度因子到 Ceres 问题 |
| `double2vector` | 优化完成后 | 同步 a, b 到全局变量 |

**状态变量**:
```cpp
double para_DepthScaleShift[1][2];  // 优化变量 [a, b]
double last_depth_a, last_depth_b;  // 上一帧值（用于随机游走）
bool has_last_depth_params;         // 是否有历史值
bool is_first_depth_optimization;   // 是否为首次优化
```

#### 3. DepthFactor（深度约束因子）
**位置**: `vins_estimator/src/factor/depth_factor.h`

**核心计算**:
```cpp
class DepthFactor : public ceres::SizedCostFunction<1, 7, 7, 7, 1, 2> {
    // 残差维度: 1 (标量)
    // 参数块: Pose_i, Pose_j, Ex_Pose, Feature, ScaleShift

    virtual bool Evaluate(...) {
        // 1. 将特征点从帧 i 变换到帧 j 的相机系
        pts_camera_j = ric^T * Rj^T * (Ri * (ric * pts_i/λ + tic) + Pi - Pj) - tic;

        // 2. 计算 VINS 估计的逆深度
        double estimated_metric_inv_depth = 1.0 / pts_camera_j.z();

        // 3. 计算残差
        residuals[0] = sqrt_info * ((a * d_nn + b) - estimated_metric_inv_depth);

        // 4. 计算雅可比矩阵（链式法则传播）
        ...
    }
};
```

**深度有效性检查**:
- 如果 `depth_metric_j <= 0.05m`，跳过该约束（点在相机后方或退化）

#### 4. DepthScaleShiftRandomWalkFactor（随机游走因子）
**位置**: `vins_estimator/src/factor/depth_scale_shift_random_walk_factor.h`

**作用**: 约束深度参数的帧间变化，防止剧烈跳变

**残差定义**:
```cpp
residuals[0] = (a_current - a_prev) / sigma_a;
residuals[1] = (b_current - b_prev) / sigma_b;
```

**默认噪声参数**:
- `sigma_a = 0.005` (尺度因子的随机游走标准差)
- `sigma_b = 0.01`  (偏移因子的随机游走标准差)

---

## 深度融合策略

### 1. 在线参数估计（初始化阶段）

**触发时机**: VIO 初始化成功后，`estimateDepthScaleShift()` 被调用

**算法流程**:

```cpp
// 步骤1: 收集配对数据
for (auto& feature : f_manager.feature) {
    if (feature.estimated_depth > 0.0) {  // 已三角化
        // 获取 VINS 逆深度
        double inv_depth_vins = 1.0 / feature.estimated_depth;

        // 遍历所有观测帧，查找有深度图的帧
        for (auto& obs : feature.observations) {
            if (frame_has_depth_map(obs.frame_id)) {
                // 从深度图读取网络逆深度
                double inv_depth_net = depth_map.at<float>(obs.uv);

                // 添加配对 (inv_depth_net, inv_depth_vins)
                data_pairs.push_back({inv_depth_net, inv_depth_vins});
            }
        }
    }
}

// 步骤2: 异常值过滤（3-sigma 规则）
filter_outliers_by_zscore(data_pairs, 3.0);

// 步骤3: 线性最小二乘求解 a, b
// 拟合: inv_depth_vins = a * inv_depth_net + b
solve_linear_system(data_pairs, a, b);

// 步骤4: 质量检查
double correlation = pearson_correlation(data_pairs);
if (correlation < 0.6) {
    // 相关性太低，拒绝结果，使用配置默认值
    return;
}

// 步骤5: 参数边界检查
if (a < 0.01 || a > 5.0 || b < -1.0 || b > 1.0) {
    // 参数超出物理合理范围，拒绝
    return;
}

// 步骤6: 更新系统参数
DEPTH_SCALE_A = a;
DEPTH_SHIFT_B = b;
para_DepthScaleShift[0][0] = a;
para_DepthScaleShift[0][1] = b;
```

**鲁棒性保障**:
1. **数据筛选**: 仅使用已三角化的特征（`estimated_depth > 0`）
2. **异常值剔除**: 3-sigma 规则移除离群点
3. **统计验证**: Pearson 相关系数 `r > 0.6`（强线性关系）
4. **物理约束**: 参数范围检查（防止病态解）
5. **数值稳定性**: SVD 求解 + Tikhonov 正则化

### 2. 自适应权重策略（运行阶段）

**核心思想**: **物理感知权重调节** — 运动模糊时降低深度约束权重

**理论依据**:
- 相机快速旋转/振动 → 运动模糊 → 深度网络预测质量下降
- IMU 陀螺仪测量角速度，加速度计测量平移扰动
- 综合评分 = `gyro_norm + ACC_DISTURBANCE_WEIGHT * acc_disturbance`

**实现代码**（`estimator.cpp:2019-2183`）:

```cpp
// A. 计算陀螺仪强度（角速度平均范数）
double current_gyro_norm = 0.0;
for (int i = 0; i < WINDOW_SIZE; i++) {
    for (const auto& gyr : pre_integrations[i]->gyr_buf) {
        current_gyro_norm += gyr.norm();
    }
}
current_gyro_norm /= total_gyro_count;

// B. 计算加速度扰动（偏离重力的程度）
double current_acc_disturbance = 0.0;
const double GRAVITY_NOMINAL = 9.81;
for (int i = 0; i < WINDOW_SIZE; i++) {
    for (const auto& acc : pre_integrations[i]->acc_buf) {
        current_acc_disturbance += abs(acc.norm() - GRAVITY_NOMINAL);
    }
}
current_acc_disturbance /= total_acc_count;

// C. 综合不稳定性评分
double combined_score = current_gyro_norm + ACC_DISTURBANCE_WEIGHT * current_acc_disturbance;

// D. 线性插值计算自适应权重
double adaptive_weight;
if (combined_score < THRESHOLD_LOW) {  // 稳定（默认 0.8）
    adaptive_weight = DEPTH_WEIGHT_STATIC;  // 高权重（默认 3.0）
} else if (combined_score > THRESHOLD_HIGH) {  // 不稳定（默认 2.5）
    adaptive_weight = DEPTH_WEIGHT_DYNAMIC;  // 低权重（默认 1.0）
} else {  // 线性过渡
    double ratio = (combined_score - THRESHOLD_LOW) / (THRESHOLD_HIGH - THRESHOLD_LOW);
    adaptive_weight = DEPTH_WEIGHT_STATIC - ratio * (DEPTH_WEIGHT_STATIC - DEPTH_WEIGHT_DYNAMIC);
}

// E. 计算自适应 Huber 阈值（保持物理误差一致性）
double adaptive_huber_threshold = adaptive_weight * PHYSICAL_ERROR_THRESHOLD;
```

**参数配置**（`euroc_config.yaml`）:
```yaml
# 静态权重（稳定状态）
depth_weight_static: 3.0
# 动态权重（运动模糊）
depth_weight_dynamic: 1.0
# 物理误差阈值（0.25 = Depth Anything V2 特征误差下限）
physical_error_threshold: 0.25
# 不稳定性评分阈值
instability_threshold_low: 0.8
instability_threshold_high: 2.5
# 加速度扰动权重因子
acc_disturbance_weight: 0.3
```

### 3. 预热机制（Warm-Up Phase）

**问题**: 初始化刚完成时，a/b 初值可能与真值偏差较大，使用窄 Huber 会导致收敛慢

**解决方案**: 前 N 帧使用**宽松 Huber Loss**（阈值 5.0），允许大梯度快速收敛

**实现**（`estimator.cpp:2141-2153`）:
```cpp
if (depth_fusion_frame_count <= DEPTH_FUSION_WARMUP_FRAMES) {  // 默认 30 帧
    depth_loss_function = new ceres::HuberLoss(5.0);  // 宽松阈值
    ROS_INFO("[Warm-up] Using wide Huber loss (threshold=5.0)");
} else {
    depth_loss_function = new ceres::HuberLoss(adaptive_huber_threshold);  // 自适应
    ROS_INFO("[Normal] Using adaptive Huber loss (threshold=%.3f)", adaptive_huber_threshold);
}
```

**注意**: 使用 `depth_fusion_frame_count` 而非 `global_frame_count`，因为深度融合仅在初始化成功后开始计数。

### 4. 随机游走约束（参数平滑）

**目的**: 避免 a, b 在帧间剧烈跳变，保持轨迹平滑

**约束形式**:
```
a_{k+1} = a_k + noise_a,  noise_a ~ N(0, sigma_a²)
b_{k+1} = b_k + noise_b,  noise_b ~ N(0, sigma_b²)
```

**添加到优化问题**（`estimator.cpp:2288-2318`）:
```cpp
if (has_last_depth_params) {
    double current_sigma_a = DEPTH_A_RANDOM_WALK;
    double current_sigma_b = DEPTH_B_RANDOM_WALK;

    // 首次优化放松约束 100 倍（允许大幅跳转修正初值）
    if (is_first_depth_optimization) {
        current_sigma_a *= 100.0;
        current_sigma_b *= 100.0;
    }

    DepthScaleShiftRandomWalkFactor* rw_factor =
        new DepthScaleShiftRandomWalkFactor(last_depth_a, last_depth_b,
                                           current_sigma_a, current_sigma_b);
    problem.AddResidualBlock(rw_factor, nullptr, para_DepthScaleShift[0]);
}
```

---

## 参数管理机制

### 1. 全局配置参数（`parameters.h/cpp`）

```cpp
// 仿射变换初始值
extern double DEPTH_SCALE_A;        // 默认 0.12（经验值，取决于数据集）
extern double DEPTH_SHIFT_B;        // 默认 0.0

// 约束强度
extern double DEPTH_FACTOR_WEIGHT;  // 默认 1.0（会被自适应权重覆盖）
extern double DEPTH_FACTOR_HUBER_THRESHOLD;  // 默认 0.5（用于边缘化，非主要约束）

// 随机游走噪声
extern double DEPTH_A_RANDOM_WALK;  // 默认 0.005
extern double DEPTH_B_RANDOM_WALK;  // 默认 0.01

// 预热参数
extern int DEPTH_FUSION_WARMUP_FRAMES;  // 默认 30
```

### 2. 运行时变量（`Estimator`）

```cpp
// Ceres 优化变量（持久化）
double para_DepthScaleShift[1][2];  // [a, b]

// 历史值（用于随机游走）
double last_depth_a;
double last_depth_b;
bool has_last_depth_params;

// 控制标志
bool is_first_depth_optimization;  // 首次优化标记
int depth_fusion_frame_count;      // 深度融合帧计数
```

### 3. 参数同步流程

```
初始化阶段:
clearState() → para_DepthScaleShift = [DEPTH_SCALE_A, DEPTH_SHIFT_B]

线性对齐:
estimateDepthScaleShift() → 更新 DEPTH_SCALE_A, DEPTH_SHIFT_B, para_DepthScaleShift

每帧优化:
vector2double() → 将 Eigen 变量转为 double 数组（para_DepthScaleShift 已是优化变量，无需转换）
  ↓
optimization() → Ceres 修改 para_DepthScaleShift[0][0], para_DepthScaleShift[0][1]
  ↓
double2vector() → 同步到全局变量
                  DEPTH_SCALE_A = para_DepthScaleShift[0][0]
                  DEPTH_SHIFT_B = para_DepthScaleShift[0][1]
                  last_depth_a = DEPTH_SCALE_A
                  last_depth_b = DEPTH_SHIFT_B
```

---

## 实现细节

### 1. 深度图计算时机

| 阶段 | 计算时机 | 代码位置 |
|------|----------|----------|
| **快速初始化** | 窗口第一帧（异步推理） | `tryComputeFirstFrameDepth()` |
| **传统初始化** | 初始化成功后计算一帧 | `ensureDepthMapForAlignment()` |
| **后端优化** | 每帧为滑动窗口内所有未计算的帧推理 | `processImage()` L943-983 |

**后端批量推理**（`estimator.cpp:943-983`）:
```cpp
if (ESTIMATE_DEPTH_SCALE_SHIFT && mp_depth_estimator && mp_depth_estimator->isReady()) {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        double timestamp = Headers[i].stamp.toSec();
        auto& frame = all_image_frame[timestamp];

        if (!frame.depth_map_computed && !frame.raw_image.empty()) {
            mp_depth_estimator->predict(frame.raw_image, frame.predicted_depth_map);
            frame.depth_map_computed = true;
        }
    }
}
```

### 2. 深度因子添加逻辑

**核心循环**（`estimator.cpp:2199-2266`）:
```cpp
for (auto& feature : f_manager.feature) {
    // 筛选条件：观测数>=2 && 起始帧 < WINDOW_SIZE-2
    if (!(feature.used_num >= 2 && feature.start_frame < WINDOW_SIZE - 2))
        continue;

    int first_obs_frame = feature.start_frame;
    Vector3d pts_i = feature.feature_per_frame[0].point;  // 首次观测归一化坐标

    // 遍历所有观测
    for (int idx = 0; idx < feature.feature_per_frame.size(); idx++) {
        int current_obs_frame = first_obs_frame + idx;

        // 跳过首次观测帧（避免重复参数块）
        if (first_obs_frame == current_obs_frame)
            continue;

        // 检查当前帧是否有深度图
        auto& frame = all_image_frame[Headers[current_obs_frame].timestamp];
        if (!frame.depth_map_computed)
            continue;

        // 从深度图读取预测逆深度
        Vector2d uv = feature.feature_per_frame[idx].uv;
        double d_nn = frame.predicted_depth_map.at<float>(uv.y(), uv.x());

        // 有效性检查
        if (d_nn <= 1e-6 || d_nn > 100.0)
            continue;

        // 创建深度因子
        DepthFactor* factor = new DepthFactor(d_nn, pts_i);
        problem.AddResidualBlock(factor, depth_loss_function,
                                para_Pose[first_obs_frame],
                                para_Pose[current_obs_frame],
                                para_Ex_Pose[0],
                                para_Feature[feature_index],
                                para_DepthScaleShift[0]);
    }
}
```

### 3. 边缘化处理

**关键点**: 深度参数 `para_DepthScaleShift` 是全局变量，不随滑动窗口移动

**地址映射**（`estimator.cpp:2594-2599`）:
```cpp
// MARGIN_OLD 分支
std::unordered_map<long, double*> addr_shift;
for (int i = 1; i <= WINDOW_SIZE; i++) {
    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];  // 位姿向前移动
    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
}
// 深度参数地址不变（全局参数）
addr_shift[reinterpret_cast<long>(para_DepthScaleShift[0])] = para_DepthScaleShift[0];
```

**与 frame 0 相关的深度因子边缘化**（`estimator.cpp:2473-2570`）:
```cpp
// 遍历所有与 frame 0 相关的深度因子
for (auto& feature : f_manager.feature) {
    if (feature.start_frame == 0 || any_obs_in_frame_0) {
        // 创建深度因子
        DepthFactor* factor = new DepthFactor(d_nn, pts_i);

        // 定义 drop_set（需要边缘化的参数块索引）
        std::vector<int> drop_set;
        if (first_obs_frame == 0)
            drop_set.push_back(0);  // para_Pose[0]
        if (current_obs_frame == 0)
            drop_set.push_back(1);  // para_Pose[0]（若 first_obs != 0）

        // 添加到边缘化信息
        ResidualBlockInfo* res_info =
            new ResidualBlockInfo(factor, depth_loss_function, para_blocks, drop_set);
        marginalization_info->addResidualBlockInfo(res_info);
    }
}
```

### 4. 失败处理机制

**情况1: 模型加载失败**
```cpp
// initDepthEstimator() 中
if (!mp_depth_estimator->initAsync(DEPTH_MODEL_PATH)) {
    ROS_FATAL("DepthEstimator initialization failed!");
    ros::shutdown();  // 终止节点
}
```

**情况2: 在线对齐失败**
```cpp
// estimateDepthScaleShift() 中
if (correlation < 0.6 || a < 0.01 || a > 5.0) {
    ROS_WARN("Alignment REJECTED. Using config defaults.");
    return;  // 保持配置文件的默认值
}
```

**情况3: 深度因子数量为0**
```cpp
// optimization() 中
if (depth_factor_cnt == 0 && !has_last_depth_params) {
    // 无深度因子且无先验，必须固定参数避免欠约束
    problem.SetParameterBlockConstant(para_DepthScaleShift[0]);
}
```

---

## 关键代码位置

### 核心文件

| 文件 | 主要内容 | 关键函数/类 |
|------|----------|-------------|
| `estimator.cpp` | 主估计器逻辑 | `estimateDepthScaleShift`, `optimization`, `processImage` |
| `estimator.h` | 状态变量定义 | `para_DepthScaleShift`, `last_depth_a/b` |
| `depth_estimator.cpp` | 深度推理模块 | `predict`, `initAsync`, `preprocess` |
| `depth_factor.h` | 深度约束因子 | `DepthFactor::Evaluate` |
| `depth_scale_shift_random_walk_factor.h` | 随机游走因子 | `DepthScaleShiftRandomWalkFactor` |
| `parameters.cpp` | 参数读取 | `readParameters` |

### 关键代码行数

| 功能 | 文件 | 行号范围 |
|------|------|----------|
| **参数初始化** | `estimator.cpp` | L151-160 |
| **在线对齐** | `estimator.cpp` | L287-712 |
| **自适应权重** | `estimator.cpp` | L2019-2197 |
| **预热机制** | `estimator.cpp` | L2134-2168 |
| **随机游走** | `estimator.cpp` | L2288-2333 |
| **深度因子添加** | `estimator.cpp` | L2199-2286 |
| **边缘化处理** | `estimator.cpp` | L2473-2570, L2594-2599 |
| **参数同步** | `estimator.cpp` | L1736-1772 |
| **深度推理** | `depth_estimator.cpp` | L318-482 |
| **归一化策略** | `depth_estimator.cpp` | L378-426 |

---

## 策略总结

### 深度融合的三阶段演化

1. **初始化阶段（线性对齐）**
   - 目标: 计算 a, b 初值
   - 方法: 线性最小二乘 + 鲁棒性检查
   - 质量门控: 相关系数 r > 0.6，参数范围 [0.01, 5.0]

2. **预热阶段（快速收敛）**
   - 目标: 快速修正初值偏差
   - 方法: 宽松 Huber Loss（阈值 5.0）+ 放松随机游走约束（100×）
   - 持续时间: 前 30 帧（可配置）

3. **稳定阶段（自适应融合）**
   - 目标: 平衡深度约束与视觉-IMU约束
   - 方法: 物理感知自适应权重 + 正常随机游走约束
   - 权重调节: 基于 IMU 运动评分（陀螺仪 + 加速度计）

### 设计哲学

1. **渐进式收敛**: 从宽松到严格，避免陷入局部最优
2. **物理约束**: 所有参数和残差都有物理意义的边界
3. **鲁棒性优先**: 异常检测 → 质量门控 → 降级策略（使用默认值）
4. **解耦设计**: 深度模块可独立禁用（`estimate_depth_scale_shift: 0`）

### 最佳实践

1. **数据集适配**: 根据场景调整 `depth_scale_a` 初值（室内 0.12，室外 0.08）
2. **硬件适配**: 高性能设备可提高 `depth_input_resolution` 至 518×518
3. **运动类型**: 快速运动场景降低 `depth_weight_static`（如无人机 2.0 → 1.5）
4. **调试技巧**: 监控日志 `[Multi-Factor]` 输出，检查权重和评分是否合理

---

## 参考文献

1. **Depth Anything V2**: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data", CVPR 2024
2. **MiDaS**: "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2020
3. **VINS-Mono**: "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator", TRO 2018
4. **Depth Priors in VIO**: "Learned Monocular Depth Priors in Visual-Inertial Initialization", ICRA 2022 (参考论文)

---

**文档版本**: 1.0
**生成日期**: 2025-12-08
**作者**: Claude Code
**代码库**: VINS-Mo feat/deep-sensor branch
