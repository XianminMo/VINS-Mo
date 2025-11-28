# 深度参数随机游走模型实现总结

## 修改概述

成功将深度变换参数 a, b 从全局常量模型修改为随机游走模型，使系统能够自适应地调整深度尺度和偏移参数。

## 核心修改

### 1. 参数定义 (parameters.h/cpp)

添加了两个过程噪声参数：
```cpp
extern double DEPTH_A_RANDOM_WALK;  // a 参数的随机游走噪声
extern double DEPTH_B_RANDOM_WALK;  // b 参数的随机游走噪声
```

从配置文件读取，默认值为 `5e-4`。

### 2. 随机游走先验因子 (depth_scale_shift_random_walk_factor.h)

创建了新的Ceres因子来约束参数变化：
```cpp
class DepthScaleShiftRandomWalkFactor : public ceres::SizedCostFunction<2, 2>
```

残差定义：
```
residual[0] = (a_current - a_previous) / sigma_a
residual[1] = (b_current - b_previous) / sigma_b
```

### 3. 状态变量 (estimator.h)

添加了三个成员变量来跟踪上一帧的参数值：
```cpp
double last_depth_a;          // 上一次优化后的 a 值
double last_depth_b;          // 上一次优化后的 b 值
bool has_last_depth_params;   // 是否有上一次的值
```

### 4. 优化器修改 (estimator.cpp::optimization())

在添加深度约束后，立即添加随机游走先验因子：
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

关键改进：
- **第一次优化**：只要有深度因子就不固定参数（允许优化）
- **后续优化**：添加随机游走先验，即使没有深度因子也能提供约束

### 5. 参数更新 (estimator.cpp::double2vector())

优化后更新历史值：
```cpp
last_depth_a = DEPTH_SCALE_A;
last_depth_b = DEPTH_SHIFT_B;
has_last_depth_params = true;
```

添加详细的调试输出，每10帧或参数变化显著时打印。

### 6. 初始化逻辑 (estimator.cpp::clearState())

```cpp
para_DepthScaleShift[0][0] = DEPTH_SCALE_A;  // 使用实验最佳初值 0.08
para_DepthScaleShift[0][1] = DEPTH_SHIFT_B;  // 使用实验最佳初值 0.21
last_depth_a = DEPTH_SCALE_A;
last_depth_b = DEPTH_SHIFT_B;
has_last_depth_params = false;
```

### 7. 配置文件 (euroc_config.yaml)

```yaml
depth_constraint.initial_scale_a: 0.08      # 实验最佳初值
depth_constraint.initial_shift_b: 0.21      # 实验最佳初值
depth_constraint.random_walk_a: 5.0e-4      # 随机游走噪声
depth_constraint.random_walk_b: 5.0e-4      # 随机游走噪声
```

## 工作原理

### 随机游走模型

传统模型：`a_{k+1} = a_k` (恒定)

新模型：`a_{k+1} = a_k + n_a`，其中 `n_a ~ N(0, sigma_a^2)`

### 实现方式

通过添加一个"软约束"（先验因子）来实现：
- **约束内容**：当前帧的参数应该接近上一帧的参数
- **约束强度**：由过程噪声 `sigma_a`, `sigma_b` 控制
  - 噪声越大 → 约束越弱 → 参数变化越自由
  - 噪声越小 → 约束越强 → 参数变化越保守

### 优势

1. **自适应性**：参数能够随时间缓慢变化，适应环境尺度变化
2. **鲁棒性**：避免初始化敏感性问题
3. **稳定性**：通过先验约束避免参数剧烈震荡
4. **理论正确**：符合卡尔曼滤波/因子图优化的框架

## 参数调优建议

### random_walk噪声 (sigma_a, sigma_b)

- **推荐范围**：1e-4 到 1e-3
- **调大**：参数变化更自由，能更快适应环境变化，但可能震荡
- **调小**：参数变化更保守，更平滑，但可能反应迟钝

### 初始值 (initial_scale_a, initial_shift_b)

- 使用实验得到的最佳值：a=0.08, b=0.21
- 如果场景差异很大，可能需要重新实验确定

## 预期效果

1. **参数轨迹平滑**：a, b 值随时间缓慢变化，不是突变
2. **自适应调整**：当环境尺度变化时，参数能自动调整
3. **初始化鲁棒**：即使初值不准，系统也能逐渐收敛到正确值
4. **日志输出**：每10帧打印一次参数值，验证随机游走效果

## 验证方法

1. **查看日志**：
   ```
   [Backend] Frame 10 - Depth params: a=0.080123 (Δ0.000123), b=0.209876 (Δ-0.000124)
   ```

2. **绘制曲线**：将 a, b 随时间的变化绘制成曲线，应该是平滑的

3. **对比实验**：
   - 固定参数 vs 随机游走
   - 不同初值下的收敛速度
   - 不同场景下的适应能力

## 文件修改清单

1. `vins_estimator/src/parameters.h` - 添加参数声明
2. `vins_estimator/src/parameters.cpp` - 添加参数定义和读取
3. `vins_estimator/src/estimator.h` - 添加状态变量和头文件
4. `vins_estimator/src/estimator.cpp` - 修改优化器、状态更新和初始化
5. `vins_estimator/src/factor/depth_scale_shift_random_walk_factor.h` - 新增因子
6. `config/euroc/euroc_config.yaml` - 更新配置参数

## 编译状态

✅ 编译成功，无错误

有一些Ceres弃用警告（正常，与本次修改无关）
