# IMU Bias vs 深度参数 a,b 的随机游走模型对比

## TL;DR 核心差异

| 特性 | IMU Bias (Ba, Bg) | 深度参数 (a, b) |
|------|------------------|----------------|
| **约束方式** | 通过IMU残差隐式约束 | 显式添加先验残差因子 |
| **协方差传播** | 在预积分中显式传播 | 不传播协方差（仅通过先验约束） |
| **噪声注入** | 在预积分过程中注入 | 在优化时通过先验因子注入 |
| **理论框架** | 连续时间随机过程离散化 | 离散时间点之间的软约束 |

---

## 1. IMU Bias的实现方式

### 1.1 模型定义

**连续时间随机游走模型**：
```
dBa/dt = na(t),  na ~ N(0, σ_a²)
dBg/dt = ng(t),  ng ~ N(0, σ_g²)
```

**离散化后**：
```
Ba_{k+1} = Ba_k + na * dt,  na ~ N(0, σ_a²)
Bg_{k+1} = Bg_k + ng * dt,  ng ~ N(0, σ_g²)
```

其中：
- `σ_a = ACC_W` (加速度计bias随机游走噪声)
- `σ_g = GYR_W` (陀螺仪bias随机游走噪声)

### 1.2 实现方式：通过IMU预积分因子隐式约束

#### (1) 协方差传播 (integration_base.h:20-28, 104-126)

在预积分过程中显式传播协方差：

```cpp
// 噪声矩阵定义（18维）
noise = Eigen::Matrix<double, 18, 18>::Zero();
noise.block<3, 3>(0, 0)   = (ACC_N * ACC_N) * I;  // 加速度测量噪声
noise.block<3, 3>(3, 3)   = (GYR_N * GYR_N) * I;  // 陀螺仪测量噪声
noise.block<3, 3>(6, 6)   = (ACC_N * ACC_N) * I;
noise.block<3, 3>(9, 9)   = (GYR_N * GYR_N) * I;
noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * I;  // Ba 随机游走噪声 ✅
noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * I;  // Bg 随机游走噪声 ✅

// 协方差传播
covariance = F * covariance * F.transpose() + V * noise * V.transpose();
```

**关键点**：
- **V矩阵** (15x18) 将噪声映射到状态空间
- `V.block<3, 3>(9, 12) = I * dt`  → Ba的噪声传播
- `V.block<3, 3>(12, 15) = I * dt` → Bg的噪声传播
- 随机游走噪声**乘以dt**后累积到协方差中

#### (2) IMU残差定义 (integration_base.h:160-186)

```cpp
Eigen::Matrix<double, 15, 1> evaluate(...) {
    Eigen::Matrix<double, 15, 1> residuals;

    // 位置、旋转、速度残差（省略）
    ...

    // Bias 残差（这就是随机游走约束！）
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;  // ✅ 约束 Ba_j ≈ Ba_i
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;  // ✅ 约束 Bg_j ≈ Bg_i

    return residuals;
}
```

**关键点**：
- **直接约束相邻帧的bias相等**：`Baj - Bai = 0`
- 这个残差会被**协方差的逆矩阵**加权（通过`sqrt_info`）
- 协方差中包含了随机游走噪声，所以约束强度是**自适应的**

#### (3) 信息矩阵加权 (imu_factor.h:64)

```cpp
// 使用预积分的协方差矩阵的逆作为信息矩阵
Eigen::Matrix<double, 15, 15> sqrt_info =
    Eigen::LLT<Eigen::Matrix<double, 15, 15>>(
        pre_integration->covariance.inverse()
    ).matrixL().transpose();

residual = sqrt_info * residual;
```

**关键点**：
- 协方差大 → 信息矩阵小 → 约束弱
- 协方差小 → 信息矩阵大 → 约束强
- 通过**协方差传播**自动计算出合理的约束强度

### 1.3 为什么IMU Bias不需要显式的先验因子？

因为：
1. **Bias已经是IMU因子的一部分**：IMU因子连接了`[Pose_i, SpeedBias_i, Pose_j, SpeedBias_j]`
2. **IMU因子的残差包含了Bias约束**：`Baj - Bai`
3. **协方差矩阵隐式地定义了约束强度**：通过`covariance.inverse()`

---

## 2. 深度参数 a, b 的实现方式

### 2.1 模型定义

**离散时间随机游走模型**：
```
a_{k+1} = a_k + n_a,  n_a ~ N(0, σ_a²)
b_{k+1} = b_k + n_b,  n_b ~ N(0, σ_b²)
```

其中：
- `σ_a = DEPTH_A_RANDOM_WALK`（默认 5e-4）
- `σ_b = DEPTH_B_RANDOM_WALK`（默认 5e-4）

### 2.2 实现方式：显式添加先验残差因子

#### (1) 先验因子定义 (depth_scale_shift_random_walk_factor.h)

```cpp
class DepthScaleShiftRandomWalkFactor : public ceres::SizedCostFunction<2, 2>
{
    virtual bool Evaluate(...) {
        const double a_current = parameters[0][0];
        const double b_current = parameters[0][1];

        // 残差：当前值与上一次值的差异
        residuals[0] = (a_current - a_prev) / sigma_a;  // ✅ 约束 a_current ≈ a_prev
        residuals[1] = (b_current - b_prev) / sigma_b;  // ✅ 约束 b_current ≈ b_prev

        return true;
    }

private:
    double a_prev_, b_prev_;        // 存储上一帧的值
    double inv_sigma_a_, inv_sigma_b_;  // 1/sigma（约束强度）
};
```

**关键点**：
- **直接约束当前值与上一帧值的差异**
- 约束强度由`1/sigma`决定（**手动设定**，不自动传播）

#### (2) 添加到优化问题 (estimator.cpp:1528-1558)

```cpp
if (has_last_depth_params && solver_flag == NON_LINEAR)
{
    DepthScaleShiftRandomWalkFactor* random_walk_factor =
        new DepthScaleShiftRandomWalkFactor(
            last_depth_a, last_depth_b,
            DEPTH_A_RANDOM_WALK, DEPTH_B_RANDOM_WALK);

    problem.AddResidualBlock(random_walk_factor, nullptr, para_DepthScaleShift[0]);
}
```

#### (3) 更新历史值 (estimator.cpp:1192-1195)

```cpp
// 优化后更新"上一次"的值
last_depth_a = DEPTH_SCALE_A;
last_depth_b = DEPTH_SHIFT_B;
has_last_depth_params = true;
```

### 2.3 为什么深度参数a,b需要显式的先验因子？

因为：
1. **a, b 是全局参数**：不属于任何特定的帧，没有"自然的"前后关系
2. **没有预积分机制**：深度约束是instantaneous的，不需要连续时间积分
3. **协方差不传播**：我们只关心"相邻优化之间"的参数变化，不需要复杂的协方差传播

---

## 3. 核心差异总结

### 3.1 理论差异

| 方面 | IMU Bias | 深度参数 a,b |
|------|---------|-------------|
| **物理意义** | IMU传感器的缓慢漂移 | 深度图的仿射变换系数 |
| **时间尺度** | 连续时间过程 | 离散优化时刻 |
| **状态耦合** | 与Pose, Velocity强耦合 | 仅与深度因子耦合 |
| **约束来源** | IMU运动方程 | 深度测量 |

### 3.2 实现差异

| 方面 | IMU Bias | 深度参数 a,b |
|------|---------|-------------|
| **约束位置** | IMU因子内部 | 独立的先验因子 |
| **协方差** | 显式传播（15x15矩阵） | 隐式（通过1/sigma） |
| **噪声注入** | 在预积分中连续累积 | 在优化时离散添加 |
| **参数数量** | 6维 (3+3) | 2维 |
| **雅可比计算** | 复杂（通过链式法则传播） | 简单（对角矩阵） |

### 3.3 为什么采用不同的实现方式？

**IMU Bias 使用预积分+隐式约束**，因为：
- ✅ IMU数据是**高频连续**的（100-200 Hz）
- ✅ Bias影响**所有IMU测量**，需要在积分过程中考虑
- ✅ 协方差传播能准确反映**累积误差**
- ✅ 与姿态、速度强耦合，需要联合优化

**深度参数使用显式先验因子**，因为：
- ✅ 深度约束是**低频离散**的（每帧一次优化）
- ✅ a,b 是**全局参数**，不属于特定帧
- ✅ 不需要复杂的协方差传播
- ✅ 实现简单，易于调试和调参

---

## 4. 代码对比示例

### IMU Bias 的约束方式（隐式，通过IMU因子）

```cpp
// 在 IMUFactor::Evaluate() 中
residuals.block<3, 1>(O_BA, 0) = Baj - Bai;  // 约束相邻帧bias相等
residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

// 权重由预积分的协方差决定
sqrt_info = LLT(covariance.inverse()).matrixL().transpose();
residual = sqrt_info * residual;  // 自动加权
```

### 深度参数的约束方式（显式，独立的先验因子）

```cpp
// 创建独立的先验因子
DepthScaleShiftRandomWalkFactor* prior =
    new DepthScaleShiftRandomWalkFactor(last_a, last_b, sigma_a, sigma_b);

// 添加到优化问题
problem.AddResidualBlock(prior, nullptr, para_DepthScaleShift[0]);

// 残差定义
residuals[0] = (a_current - a_prev) / sigma_a;  // 手动设定权重
residuals[1] = (b_current - b_prev) / sigma_b;
```

---

## 5. 优缺点对比

### IMU Bias 的方式（预积分+隐式约束）

**优点**：
- ✅ 理论严谨，基于随机过程理论
- ✅ 协方差自动传播，约束强度自适应
- ✅ 与其他状态（Pose, Velocity）统一处理

**缺点**：
- ❌ 实现复杂，需要维护15x15协方差矩阵
- ❌ 计算量大（协方差传播）
- ❌ 调试困难

### 深度参数的方式（显式先验因子）

**优点**：
- ✅ 实现简单，代码清晰
- ✅ 易于调试和调参
- ✅ 计算效率高（2维简单残差）
- ✅ 灵活性高（可以轻松修改约束强度）

**缺点**：
- ❌ 需要手动调参（sigma_a, sigma_b）
- ❌ 没有理论保证的最优性
- ❌ 不考虑协方差传播（但对全局参数影响不大）

---

## 6. 何时使用哪种方式？

### 使用预积分+隐式约束（IMU Bias方式），当：
- 状态变量是**连续时间过程**
- 需要在**积分过程**中考虑随机游走
- 与其他状态**强耦合**
- 有明确的**物理模型**和**随机过程理论**支持

### 使用显式先验因子（深度参数方式），当：
- 状态变量是**全局参数**或**离散时刻参数**
- 只需要约束**相邻优化之间的变化**
- 希望**实现简单**，易于调试
- 对**理论严谨性**要求不高，注重工程实用性

---

## 7. 能否将深度参数改用IMU Bias的方式？

理论上可以，但**不推荐**，因为：

1. **过度复杂**：深度参数不需要连续时间积分
2. **协方差传播无意义**：a,b是全局参数，不随时间累积误差
3. **计算浪费**：维护2x2协方差矩阵的开销不值得

**当前的显式先验因子方式已经足够好**，因为：
- ✅ 实现简单
- ✅ 效果良好
- ✅ 易于调试
- ✅ 符合问题的本质（离散优化）

---

## 8. 总结

两种方式本质上都是实现**随机游走模型**，但实现方式不同：

| 方式 | IMU Bias | 深度参数 a,b |
|------|---------|-------------|
| **约束方式** | 通过IMU因子隐式约束 | 显式添加先验因子 |
| **适用场景** | 连续时间过程 | 离散全局参数 |
| **实现复杂度** | 高（需要协方差传播） | 低（简单残差） |
| **理论严谨性** | 高（基于随机过程理论） | 中（工程实用） |
| **调试难度** | 难 | 易 |

**推荐选择**：
- 对于IMU Bias这样的连续过程状态 → 使用预积分+隐式约束
- 对于深度参数这样的全局参数 → 使用显式先验因子

我们的实现**选择了正确的方式**！🎯
