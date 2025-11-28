# 深度参数更新诊断指南

## 问题现象

运行时看不到深度参数 a, b 的更新信息，日志中只有 Visual factors。

## 可能的原因

### 1. 深度图没有计算（最常见）

**症状**：
- 只看到 `[Backend] Visual factors: XXX`
- 看不到 `[Backend] Depth factors: XXX`

**原因**：
- 深度估计模型没有异步初始化完成
- 深度图计算失败
- `all_image_frame` 中的 `depth_map_computed` 标志为 false

**检查方法**：
重新运行程序后，查看新增的诊断日志：

```bash
# 应该看到以下之一：

# 情况1：有深度因子（正常）
[INFO] [Backend] Depth factors: 150 (checked 300 features, 5 frames with depth) | a=0.080123, b=0.209876

# 情况2：没有深度因子（问题）
[WARN] [Backend] No depth factors! (checked 300 features, 0 frames with depth maps)
```

**解决方法**：
- 检查深度模型路径是否正确
- 检查深度估计器是否初始化成功
- 查看是否有 ONNX Runtime 相关错误

---

### 2. 深度约束未启用

**检查配置文件**：
```yaml
depth_constraint.estimate_scale_shift: 1  # 确保为 1
```

**当前状态**：从你的配置文件看，这个是启用的，所以不是这个问题。

---

### 3. 深度图计算但没有有效观测

**症状**：
```
[WARN] [Backend] No depth factors! (checked 300 features, 5 frames with depth maps)
```
说明有深度图，但没有成功添加深度因子。

**可能原因**：
- 特征点像素坐标超出边界
- 深度值无效（NaN, Inf, 或超出范围）
- 特征点不满足筛选条件

---

## 新增的诊断日志

重新编译后，你会看到以下新日志：

### 1. 深度因子统计（每帧）

```bash
# 有深度因子时：
[INFO] [Backend] Depth factors: 150 (checked 300 features, 5 frames with depth) | a=0.080123, b=0.209876
#                                 ↑ 添加的因子数
#                                         ↑ 检查的特征数
#                                                   ↑ 有深度图的帧数
#                                                                        ↑ 当前a,b值

# 没有深度因子时（每5帧打印一次）：
[WARN] [Backend] No depth factors! (checked 300 features, 0 frames with depth maps)
#                                                            ↑ 如果是0，说明深度图没计算
```

### 2. 参数更新日志（每帧）

```bash
[INFO] [DepthParams] Frame 1: a=0.080000 (Δ0.000000), b=0.210000 (Δ0.000000)
[INFO] [DepthParams] Frame 2: a=0.080123 (Δ0.000123), b=0.209876 (Δ-0.000124)
[INFO] [DepthParams] Frame 3: a=0.080245 (Δ0.000122), b=0.209753 (Δ-0.000123)
#                               ↑ 当前值      ↑ 变化量（验证随机游走）
```

**如果看不到这个日志**：
- 说明 `ESTIMATE_DEPTH_SCALE_SHIFT` 没有启用
- 或者系统还在初始化阶段（solver_flag == INITIAL）

---

## 快速诊断步骤

### 步骤1：重新运行程序

```bash
source devel/setup.bash
roslaunch vins_estimator euroc.launch
# 播放rosbag
```

### 步骤2：观察日志

**正常情况（有深度约束）**：
```
[INFO] Backend Depth Constraint ENABLED:
[INFO]   Initial Scale (a): 0.0800
[INFO]   Initial Shift (b): 0.2100
[INFO]   Random Walk Noise (a): 0.001000
[INFO]   Random Walk Noise (b): 0.001000
...
[INFO] [Backend] Visual factors: 990
[INFO] [Backend] Depth factors: 150 (checked 300 features, 5 frames with depth) | a=0.080000, b=0.210000
[INFO] [DepthParams] Frame 1: a=0.080000 (Δ0.000000), b=0.210000 (Δ0.000000)
...
[INFO] [Backend] Visual factors: 901
[INFO] [Backend] Depth factors: 145 (checked 280 features, 5 frames with depth) | a=0.080123, b=0.209876
[INFO] [DepthParams] Frame 2: a=0.080123 (Δ0.000123), b=0.209876 (Δ-0.000124)
```

**异常情况（深度图没计算）**：
```
[INFO] Backend Depth Constraint ENABLED:
...
[INFO] [Backend] Visual factors: 990
[WARN] [Backend] No depth factors! (checked 300 features, 0 frames with depth maps)
[INFO] [DepthParams] Frame 1: a=0.080000 (Δ0.000000), b=0.210000 (Δ0.000000)
```
↑ 注意：即使没有深度因子，参数也会输出（因为参数块已添加）

### 步骤3：根据日志判断问题

| 日志特征 | 问题 | 解决方法 |
|---------|------|---------|
| `0 frames with depth maps` | 深度图未计算 | 检查深度模型和初始化 |
| `5 frames with depth, 0 factors` | 深度值无效 | 检查深度图质量 |
| 看不到 `[DepthParams]` | 约束未启用 | 检查配置文件 |
| `Δ0.000000` 一直不变 | 参数被固定或没有先验 | 检查优化器逻辑 |

---

## 预期的随机游走效果

### 正常的参数轨迹

```
Frame 1:  a=0.080000, Δ0.000000, b=0.210000, Δ0.000000  // 初始值
Frame 2:  a=0.080123, Δ0.000123, b=0.209876, Δ-0.000124  // 开始变化
Frame 3:  a=0.080245, Δ0.000122, b=0.209753, Δ-0.000123  // 继续平滑变化
Frame 10: a=0.081200, Δ0.000110, b=0.208500, Δ-0.000115  // 累积变化
```

**特征**：
- ✅ 变化量 Δa, Δb 通常在 **1e-4 到 1e-3** 范围内
- ✅ 参数值平滑变化，不会突变
- ✅ 如果环境尺度一致，参数逐渐收敛到稳定值

### 异常情况

#### 情况1：参数不变（Δ=0）
```
Frame 1: a=0.080000, Δ0.000000, b=0.210000, Δ0.000000
Frame 2: a=0.080000, Δ0.000000, b=0.210000, Δ0.000000  ← 异常
Frame 3: a=0.080000, Δ0.000000, b=0.210000, Δ0.000000
```
**原因**：参数被固定（`SetParameterBlockConstant`）

#### 情况2：参数剧烈震荡
```
Frame 1: a=0.080000, Δ0.000000, b=0.210000, Δ0.000000
Frame 2: a=0.095000, Δ0.015000, b=0.180000, Δ-0.030000  ← 变化过大
Frame 3: a=0.070000, Δ-0.025000, b=0.240000, Δ0.060000
```
**原因**：
- 随机游走噪声太大（调小 `random_walk_a/b`）
- 深度因子权重太小（调大 `weight`）

---

## 调试技巧

### 1. 检查深度估计器初始化

在启动日志中查找：
```
[INFO] DepthEstimator initialized successfully.
[INFO] DepthEstimator::initWorker(): Model loaded and warmed up successfully (XXX ms).
```

### 2. 检查深度图计算

在 `processImage()` 函数中，深度图会被计算并存储到 `ImageFrame::predicted_depth_map`。

可以添加临时日志：
```cpp
if (frame_it->second.depth_map_computed) {
    ROS_INFO_ONCE("First depth map computed at frame %d", frame_count);
}
```

### 3. 监控优化时间

如果优化时间突然增加，可能是深度因子过多：
```bash
[INFO] solver costs: XX ms  # 正常应该 < 40ms
```

---

## 当前配置参数

从你的配置文件：
```yaml
depth_constraint.initial_scale_a: 0.08
depth_constraint.initial_shift_b: 0.21
depth_constraint.random_walk_a: 1.0e-3  # 用户调整为 1e-3（比推荐的5e-4大）
depth_constraint.random_walk_b: 1.0e-3
depth_constraint.weight: 5.0             # 用户调整为 5.0（从22降低）
```

**建议**：
- 如果深度因子很多（>200），`weight=5.0` 可能偏小，可以尝试 10-20
- 如果参数震荡，可以调小 `random_walk` 到 5e-4

---

## 下一步

1. **重新运行程序，查看新增日志**
2. **报告看到的诊断信息**，特别是：
   - `checked X features, Y frames with depth maps`
   - `[DepthParams]` 的输出
3. 根据日志进一步诊断问题
