# 深度参数对齐诊断指南

## 问题描述

用户报告即使修复了多帧遍历逻辑后，仍然出现"0个有效配对点"的问题。现在已添加详细的诊断日志系统来精确定位问题。

## 新增诊断日志

编译后运行程序时，会看到类似这样的详细诊断信息：

```bash
[INFO] [Depth Init] Diagnostic info:
[INFO] [Depth Init]   Features checked: 300
[INFO] [Depth Init]   Features triangulated (solve_flag==1): 250
[INFO] [Depth Init]   Features with valid depth_vins: 240
[INFO] [Depth Init]   Features with observations: 240
[INFO] [Depth Init]   Observation frames checked: 720
[INFO] [Depth Init]   Observation frames with depth map: 100
[INFO] [Depth Init]   Observations out of bounds: 50
[INFO] [Depth Init]   Observations with invalid depth_net: 50
[INFO] [Depth Init]   Final valid pairing points: 0
```

## 诊断指标说明

| 指标 | 含义 | 期望值 |
|------|------|--------|
| Features checked | 检查的特征点总数 | > 100 |
| Features triangulated | 成功三角化的特征点 (solve_flag==1) | > 80% of checked |
| Features with valid depth_vins | VINS深度值合理 (0.1m-10m) | > 90% of triangulated |
| Features with observations | 有观测帧的特征点 | = valid depth_vins |
| Observation frames checked | 检查的观测帧总数 | 约 3-5× features |
| Observation frames with depth map | 有深度图的观测帧数 | > 0 (关键指标) |
| Observations out of bounds | 像素坐标超出边界的观测数 | < 30% of with depth map |
| Observations with invalid depth_net | 深度值无效的观测数 | < 50% of remaining |
| Final valid pairing points | 最终有效配对点数 | ≥ 20 |

## 问题诊断流程

### 情况1：obs_frames_with_depth_map = 0

**症状**：
```
[INFO] [Depth Init]   Observation frames with depth map: 0
[WARN] [Depth Init] Alignment failed: only 0 valid points
```

**可能原因**：
- 深度图未正确存储到 ImageFrame
- 时间戳不匹配导致查找失败
- ensureDepthMapForAlignment() 失败

**请提供**：
- `ensureDepthMapForAlignment()` 的完整日志
- 检查是否有 `[Depth Alignment] Computed depth map` 或 `[Depth Alignment] Found X frames` 的日志

### 情况2：obs_out_of_bounds = obs_frames_with_depth_map

**症状**：
```
[INFO] [Depth Init]   Observation frames with depth map: 100
[INFO] [Depth Init]   Observations out of bounds: 100
```

**可能原因**：
- 特征点uv坐标与深度图尺寸不匹配
- 图像裁剪或尺度变换问题

**请提供**：
- 原始图像尺寸（从launch文件或数据集信息）
- 深度图尺寸（应该与原始图像一致）

### 情况3：obs_invalid_depth_net ≈ (obs_frames_with_depth_map - obs_out_of_bounds)

**症状**：
```
[INFO] [Depth Init]   Observations with invalid depth_net: 80
[INFO] [Depth Init]   Final valid pairing points: 0
```

**可能原因**：
- 深度估计网络输出异常（NaN, Inf, 0）
- 图像质量问题（灰度图转BGR等）

**已知问题**：
用户日志显示：
```
[WARN] DepthEstimator: Input appears to be pseudo-color (grayscale converted to BGR).
```

这可能影响深度估计质量。

## 下一步操作

请运行新编译的程序，并提供完整的诊断日志（所有9行 `[Depth Init]` 开头的输出）。

根据诊断结果，我们可以精确定位是以下哪种问题：
1. 深度图数据存储问题
2. 像素坐标问题
3. 深度值质量问题
4. 特征三角化问题

这将帮助我们快速找到解决方案！🔍
