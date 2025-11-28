# 在线深度参数初始化（线性对齐）

## 功能概述

在VIO系统初始化完成后，自动估计深度融合参数 `a` 和 `b`，避免硬编码值在切换数据集时导致的尺度漂移。

## 问题背景

当前系统使用硬编码的深度参数初始值（a=0.08, b=0.21），这在EuRoC V2_03数据集上表现良好，但切换到其他数据集（如V1_01）时，由于场景尺度差异，会导致严重的尺度漂移问题。

**根本原因**：不同数据集的环境尺度不同，网络预测的深度与VINS三角化的深度之间的对应关系（a和b参数）会发生变化。

## 解决方案

实现在线深度参数初始化：在VIO完成初始化后，使用当前滑动窗口中的特征点，通过线性回归自动计算最优的a和b参数。

## 算法原理

### 对齐模型

```
depth_vins = a * depth_net + b
```

其中：
- `depth_vins`: VINS三角化的特征点深度
- `depth_net`: 深度网络预测的深度值
- `a, b`: 待估计的线性变换参数

### 线性最小二乘求解

目标：
```
min Σ ||depth_vins - (a * depth_net + b)||²
```

正规方程：
```
[Σ(d_net²)   Σ(d_net)] [a]   [Σ(d_net * d_vins)]
[Σ(d_net)    N       ] [b] = [Σ(d_vins)        ]
```

求解：使用Eigen的LDLT分解求解2x2线性系统。

## 实现细节

### 1. 函数签名 (estimator.h:74)

```cpp
void estimateDepthScaleShift();
```

### 2. 数据收集 (estimator.cpp:187-260)

遍历特征管理器中的所有特征点，收集有效的配对数据：

**筛选条件**：
- 特征点必须成功三角化（`solve_flag == 1`）
- 深度值在合理范围内（0.1m - 10m）
- 对应帧必须有深度图
- 像素坐标在图像边界内
- 网络预测深度值有效且有限

**配对数据**：
```cpp
(depth_net, depth_vins)
```

### 3. 线性系统构建与求解 (estimator.cpp:271-310)

```cpp
// 构建 2x2 矩阵
Eigen::Matrix2d A;
A(0, 0) = sum_dn_dn;  // Σ(depth_net²)
A(0, 1) = sum_dn;      // Σ(depth_net)
A(1, 0) = sum_dn;      // Σ(depth_net)
A(1, 1) = N;           // 点数

Eigen::Vector2d b;
b(0) = sum_dn_dv;      // Σ(depth_net * depth_vins)
b(1) = sum_dv;         // Σ(depth_vins)

// LDLT求解
Eigen::Vector2d x = A.ldlt().solve(b);
```

### 4. 合理性检查 (estimator.cpp:312-325)

**数据点要求**：
- 最少需要 20 个有效配对点

**参数范围**：
- `a ∈ [1e-3, 10.0]`
- `b ∈ [-5.0, 5.0]`
- 值必须有限（不是NaN或Inf）

**失败处理**：
- 如果检查失败，打印警告并保留配置文件中的初值
- 系统继续运行，使用硬编码值

### 5. 参数更新 (estimator.cpp:337-348)

更新全局参数：
```cpp
// 更新Ceres优化参数
para_DepthScaleShift[0][0] = estimated_a;
para_DepthScaleShift[0][1] = estimated_b;

// 更新全局配置变量
DEPTH_SCALE_A = estimated_a;
DEPTH_SHIFT_B = estimated_b;

// 更新随机游走模型历史值
last_depth_a = estimated_a;
last_depth_b = estimated_b;
has_last_depth_params = true;
```

### 6. 集成到初始化流程 (estimator.cpp:558)

在VIO初始化成功后、切换到NON_LINEAR模式之前调用：

```cpp
if(is_init_success)
{
    ROS_INFO("Init completed at stamp: %.9f s", init_ts);

    // *** 在线估计深度参数 ***
    estimateDepthScaleShift();

    solver_flag = NON_LINEAR;
    solveOdometry();
    ...
}
```

## 适用范围

### ✅ 支持的初始化方法

- **快速深度初始化**（USE_FAST_INIT=1）
- **标准SFM初始化**（USE_FAST_INIT=0）

**两种方法都会执行这个对齐过程**，确保后端优化有准确的初始值。

### ⚠️ 注意事项

1. **不与快速初始化中的a,b混淆**
   - 快速初始化中的a,b是FastInitializer内部使用的临时参数
   - 这里估计的a,b是后端优化使用的全局参数

2. **不影响随机游走逻辑**
   - 这个功能仅作为初始值提供者
   - 后续的随机游走优化会在这个基础上继续调整

3. **依赖深度图计算**
   - 必须有可用的深度图（`depth_map_computed=true`）
   - 如果深度图未计算，会使用配置文件的初值

## 预期效果

### 日志输出

**成功情况**：
```bash
[INFO] [Depth Init] ✓ Online alignment successful:
[INFO] [Depth Init]   Valid points: 245 / 300 checked
[INFO] [Depth Init]   Estimated: a = 0.085432, b = 0.198765
[INFO] [Depth Init]   RMSE: 0.1234 m
```

**失败情况**（点数不足）：
```bash
[WARN] [Depth Init] Alignment failed: only 15 valid points (need >= 20). Using config values.
```

**失败情况**（不合理的值）：
```bash
[WARN] [Depth Init] Alignment failed: unreasonable values (a=15.234567, b=-8.123456). Using config values.
```

**深度约束未启用**：
```bash
[WARN] [Depth Init] Depth constraint is disabled, skipping online alignment.
```

### 参数自适应

**EuRoC V2_03**（原始数据集）：
- 估计值应该接近 a≈0.08, b≈0.21

**EuRoC V1_01**（不同尺度）：
- 估计值可能显著不同，例如 a≈0.12, b≈0.15
- **这正是我们想要的**：系统自动适应新的尺度

## 调试与验证

### 1. 检查估计质量

**RMSE指标**：
- 好：RMSE < 0.2m
- 可接受：RMSE = 0.2-0.5m
- 差：RMSE > 0.5m

如果RMSE过大，可能原因：
- 深度网络预测质量差
- VINS三角化不准确
- 场景本身有尺度模糊性

### 2. 对比配置值与估计值

在euroc_config.yaml中设置初值：
```yaml
depth_constraint.initial_scale_a: 0.08
depth_constraint.initial_shift_b: 0.21
```

观察日志中的估计值：
- 如果差异很小（<5%），说明配置合理
- 如果差异很大（>20%），说明数据集特性不同，在线估计发挥了作用

### 3. 对比有无在线估计的轨迹

**测试A**：使用硬编码初值（注释掉estimateDepthScaleShift()调用）
**测试B**：使用在线估计

对比两者的轨迹精度和尺度一致性。

## 与随机游走的关系

这个功能与之前实现的随机游走模型是**互补**的：

| 阶段 | 功能 | 作用 |
|------|------|------|
| **初始化** | 在线参数估计 | 提供准确的初始值 |
| **后端优化** | 随机游走模型 | 动态跟踪参数变化 |

**工作流程**：
```
VIO初始化成功
    ↓
在线估计a,b（本功能）
    ↓
设置para_DepthScaleShift和last_depth_a/b
    ↓
第一次后端优化（使用估计的初值）
    ↓
后续优化（随机游走约束参数变化）
```

## 参数调优

### 最小点数阈值（min_points）

当前值：20

- **调小**（10-15）：更容易触发在线估计，但可能不稳定
- **调大**（30-50）：更稳定，但可能在特征少时失败

### 深度范围（depth_vins筛选）

当前范围：[0.1m, 10m]

- 根据具体场景调整
- 室内场景：[0.5m, 5m]
- 室外场景：[1m, 20m]

### 参数合理性范围

当前：a∈[1e-3, 10.0], b∈[-5.0, 5.0]

- 如果估计值经常超出范围，考虑放宽
- 如果不确定，可以先放宽范围观察实际估计值

## 故障排查

### 问题1：始终使用配置值

**可能原因**：
- 深度图未计算
- 特征点数量不足
- 特征点未成功三角化

**检查**：
```bash
# 查看日志
grep "Depth Init" log.txt

# 应该看到：
# "only X valid points" → 点数不足
# "depth maps may not be computed" → 深度图问题
```

### 问题2：估计值不合理

**可能原因**：
- 深度网络预测质量差
- VINS三角化误差大
- 数据关联错误

**检查**：
- 查看RMSE值
- 可视化depth_net和depth_vins的散点图
- 检查深度图质量

### 问题3：不同数据集估计值差异大

**这是正常现象**！说明：
- 在线估计正常工作
- 成功捕捉到了场景尺度差异
- 避免了硬编码值的局限性

## 未来改进方向

1. **鲁棒估计**
   - 使用RANSAC过滤异常值
   - 使用Huber损失函数

2. **协方差估计**
   - 计算a,b的不确定性
   - 传播到随机游走模型的初始协方差

3. **迭代优化**
   - 不只在初始化时估计一次
   - 定期重新估计以适应场景变化

4. **多尺度估计**
   - 对不同深度范围使用不同的a,b
   - 处理深度非线性关系

## 总结

✅ **解决的问题**：硬编码参数导致的跨数据集尺度漂移

✅ **实现方式**：线性最小二乘回归

✅ **集成位置**：VIO初始化完成后、后端优化开始前

✅ **适用范围**：所有初始化方法（快速/标准SFM）

✅ **与现有功能兼容**：不影响随机游走模型

✅ **降级策略**：估计失败时使用配置值

这个功能显著提升了系统在不同场景下的通用性和鲁棒性！🎯
