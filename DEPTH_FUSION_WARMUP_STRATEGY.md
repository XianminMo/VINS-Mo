# Depth Fusion Warm-up Strategy - Implementation Summary

## Problem Statement

### Root Cause Analysis

When testing on difficult datasets (e.g., EuRoC MH_05), we discovered a critical issue with the previously implemented Huber Loss:

**The Problem**:
- Initial values of depth parameters (a, b) might be far from the truth (e.g., initialized at a=0.05, truth is a=0.18)
- This creates huge initial residuals (e.g., > 5.0)
- Huber Loss interprets this huge initialization error as an "outlier" and suppresses the gradient
- Result: The "Relaxed Random Walk" constraint cannot work because gradients are too weak
- System stays "locked" at wrong scale for 20+ seconds (observed in flat error plots)

**Why This Happens**:
- Huber Loss was designed to reject outliers during motion blur
- But it cannot distinguish between:
  1. Legitimate initialization error (large residual due to bad initial guess)
  2. Measurement outlier (large residual due to depth network error)

## Solution: Two-Phase Strategy

### Overview

Implement a warm-up period where the system uses different loss functions based on the number of frames processed:

1. **Phase 1 - Warm-up (frames 0 to N)**: Use L2 loss (no robust kernel)
   - Purpose: Allow maximum gradient to quickly pull (a, b) from bad initialization to correct values
   - Assumption: Drone is not performing aggressive maneuvers in first 2-3 seconds
   - Trust: Initial frames have good depth predictions (no motion blur yet)

2. **Phase 2 - Robust (frames N+)**: Use Huber loss
   - Purpose: Once (a, b) are converged, protect system from outliers
   - Handles: Motion blur, rapid turns, and other depth network failures

### Default Configuration

- **Warm-up Duration**: 50 frames (approximately 2-3 seconds at 20 Hz)
- **Huber Threshold**: 1.0 (unchanged from previous implementation)

## Implementation Details

### 1. Parameter Addition

**Files Modified**:
- `vins_estimator/src/parameters.h:59`
- `vins_estimator/src/parameters.cpp:37, 241, 254`

**New Parameter**:
```cpp
extern int DEPTH_FUSION_WARMUP_FRAMES;
```

**Configuration** (`config/euroc/euroc_config.yaml:59`):
```yaml
depth_constraint.warmup_frames: 50
```

### 2. Dynamic Loss Function Switching

**Location**: `vins_estimator/src/estimator.cpp:1888-1912`

**Implementation**:
```cpp
// 创建鲁棒核函数（用于处理深度估计的异常值）
// 实施两阶段策略：
// 阶段1 (前N帧): 使用L2损失 (nullptr) 以获得最大梯度，快速从错误初值收敛
// 阶段2 (N帧后): 使用Huber损失来抑制运动模糊等异常值
ceres::LossFunction *depth_loss_function = nullptr;
static bool warmup_finished = false;

if (frame_count >= DEPTH_FUSION_WARMUP_FRAMES)
{
    depth_loss_function = new ceres::HuberLoss(DEPTH_FACTOR_HUBER_THRESHOLD);

    // 只在第一次切换时打印警告
    if (!warmup_finished)
    {
        ROS_WARN("[Backend] Depth fusion warm-up FINISHED at frame %d. Enabling Huber Loss (threshold=%.2f) for outlier rejection.",
                 frame_count, DEPTH_FACTOR_HUBER_THRESHOLD);
        warmup_finished = true;
    }
}
else
{
    // 预热阶段：使用标准L2损失（不抑制梯度）
    ROS_INFO_THROTTLE(5.0, "[Backend] Depth fusion WARM-UP phase (frame %d/%d). Using L2 loss for fast convergence.",
                     frame_count, DEPTH_FUSION_WARMUP_FRAMES);
}
```

**Logic**:
1. Check if `frame_count >= DEPTH_FUSION_WARMUP_FRAMES`
2. If NO: Use `nullptr` (standard L2 loss)
3. If YES: Create `HuberLoss` with threshold
4. Use static flag `warmup_finished` to log transition only once

### 3. Integration with Relaxed Random Walk

This warm-up strategy works synergistically with the previously implemented "Relaxed Random Walk" feature:

**Combined Effect**:
- Frame 0: Bad initialization (a=0.05, truth=0.18)
- Frames 1-50:
  - L2 loss provides strong gradients
  - Relaxed random walk (100x) allows large parameter jumps
  - Parameters quickly converge to truth
- Frame 50+:
  - Parameters are now correct
  - Switch to Huber loss
  - Normal random walk constraint activated
  - System is robust to outliers

## Logging and Diagnostics

### Log Messages

1. **During Warm-up** (every 5 seconds):
   ```
   [Backend] Depth fusion WARM-UP phase (frame X/50). Using L2 loss for fast convergence.
   ```

2. **Warm-up Transition** (once at frame 50):
   ```
   [Backend] Depth fusion warm-up FINISHED at frame 50. Enabling Huber Loss (threshold=1.00) for outlier rejection.
   ```

3. **Parameter Loading** (on startup):
   ```
   Backend Depth Constraint ENABLED:
     Initial Scale (a): 0.0500
     Initial Shift (b): 0.0500
     Random Walk Noise (a): 0.001000
     Random Walk Noise (b): 0.001000
     Factor Weight: 2.5000
     Huber Threshold: 1.0000
     Warmup Frames: 50
   ```

### Monitoring Behavior

To verify the warm-up strategy is working:

1. **Check logs** for "WARM-UP phase" messages in first 2-3 seconds
2. **Watch for transition** message at frame 50
3. **Plot depth parameters** (a, b) over time - should see rapid initial convergence
4. **Monitor residuals** - should decrease quickly in first 50 frames

## Configuration Tuning

### Adjusting Warm-up Duration

**Increase warm-up frames** if:
- Parameters take longer than 50 frames to converge
- Initial guess is very far from truth
- Dataset starts with gentle motion

**Decrease warm-up frames** if:
- Dataset has aggressive motion from the start
- You want faster transition to robust mode
- Initial guess is already close to truth

**Recommended Range**: 30-100 frames

**Example**:
```yaml
# Conservative (more warm-up time)
depth_constraint.warmup_frames: 80

# Aggressive (fast transition to robust mode)
depth_constraint.warmup_frames: 30
```

### Interaction with Other Parameters

The warm-up strategy interacts with:

1. **Initial values** (`initial_scale_a`, `initial_shift_b`):
   - Better initialization → can use shorter warm-up
   - Worse initialization → need longer warm-up

2. **Random walk noise** (`random_walk_a`, `random_walk_b`):
   - Larger noise → faster convergence → can use shorter warm-up
   - Smaller noise → slower convergence → need longer warm-up

3. **Huber threshold** (`huber_threshold`):
   - Only affects Phase 2 (after warm-up)
   - Lower threshold → more aggressive outlier rejection
   - Higher threshold → more tolerant to large residuals

## Testing and Validation

### Test Cases

1. **EuRoC MH_05** (difficult dataset with rapid turns):
   - Before: Parameters stuck at wrong values for 20+ seconds
   - After: Should converge within 2-3 seconds (50 frames)
   - Verify: Plot shows rapid initial descent followed by stable tracking

2. **Normal datasets** (gentle motion):
   - Warm-up should complete without issues
   - Huber loss should activate smoothly
   - No degradation in performance

3. **Edge case** (aggressive motion from start):
   - Monitor if L2 loss causes issues
   - May need to decrease warm-up frames
   - Or improve initialization

### Expected Results

**Phase 1 (Warm-up)**:
- Depth parameters (a, b) rapidly converge to correct values
- Large initial residuals decrease quickly
- No outlier rejection (trust depth network)

**Phase 2 (Robust)**:
- Parameters stable and tracking slowly
- Outliers (motion blur) are rejected
- System maintains good trajectory

## Files Modified

1. `vins_estimator/src/parameters.h:59` - Added `DEPTH_FUSION_WARMUP_FRAMES` declaration
2. `vins_estimator/src/parameters.cpp:37` - Added variable definition
3. `vins_estimator/src/parameters.cpp:241` - Added parameter reading from config
4. `vins_estimator/src/parameters.cpp:254` - Added logging
5. `vins_estimator/src/estimator.cpp:1888-1912` - Implemented dynamic loss function switching
6. `config/euroc/euroc_config.yaml:56-59` - Added configuration parameter

## Compilation

```bash
catkin build vins_estimator -j4 --no-status
```

✅ Verified: Compiles successfully with no new errors

## Related Documents

- `ROBUST_KERNEL_AND_RELAXED_INIT_IMPLEMENTATION.md` - Previous implementation (Huber loss + relaxed random walk)
- `RANDOM_WALK_IMPLEMENTATION.md` - Random walk constraint design
- `DEPTH_PARAM_ESTIMATION_ROOT_CAUSE_ANALYSIS.md` - Parameter alignment analysis

## Summary

The warm-up strategy solves the "locked at wrong scale" problem by:

1. Using L2 loss during warm-up for maximum gradient
2. Pairing with relaxed random walk for fast parameter jumps
3. Transitioning to Huber loss after convergence for outlier protection
4. Making the entire system robust to both initialization errors AND measurement outliers

This creates a two-phase adaptive system that handles both initialization and runtime challenges effectively.
