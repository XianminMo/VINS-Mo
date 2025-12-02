# Plot Script Fix: Common Time Range Alignment

## Problem

**Issue**: Unfair comparison when trajectories have different start times.

**Example Scenario**:
```
Original VINS starts at: t = 10.5s
Ours (Deep Fusion) starts at: t = 12.3s (1.8s later)
```

**Impact**:
- Original VINS includes initialization phase (10.5s - 12.3s) with potentially high errors
- Ours skips this period entirely
- RMSE comparison is **biased** in favor of the later-starting trajectory

## Solution

Modified `plot_comprehensive.py` to:

1. **Find Common Time Range**:
   ```python
   common_start = max(baseline_start, ours_start)  # Later of the two starts
   common_end = min(baseline_end, ours_end)        # Earlier of the two ends
   ```

2. **Crop Both Trajectories**:
   - Baseline: Remove frames before `common_start`
   - Ours: Remove frames before `common_start`
   - GT: Crop to match common time range

3. **Recalculate Metrics**:
   - APE (RMSE, mean, max, std)
   - RPE (RMSE, mean, max, std)
   - All based on **identical time range**

4. **Update Plots**:
   - Trajectory plot: Only show common segments
   - Boxplot: Only use common-range errors
   - Time evolution: Start from t=0 (common start)

## Implementation Details

### Key Functions Added

```python
def crop_to_time_range(traj, ape_data, rpe_data, start_time, end_time):
    """Crop trajectory and error arrays to specified time range"""
    mask = (traj.timestamps >= start_time) & (traj.timestamps <= end_time)
    
    cropped_traj = trajectory.Trajectory(
        positions_xyz=traj.positions_xyz[mask],
        orientations_quat_wxyz=traj.orientations_quat_wxyz[mask],
        timestamps=traj.timestamps[mask]
    )
    
    cropped_ape = ape_data[mask]
    cropped_rpe = rpe_data[mask[:len(rpe_data)]]  # RPE may be shorter
    
    return cropped_traj, cropped_ape, cropped_rpe

def compute_stats(data):
    """Recalculate statistics for cropped data"""
    return {
        'rmse': np.sqrt(np.mean(data**2)),
        'mean': np.mean(data),
        'max': np.max(data),
        'std': np.std(data)
    }
```

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Original Workflow (BIASED)                                 │
├─────────────────────────────────────────────────────────────┤
│  Baseline: [10.5s ─────────────────────── 95.2s]           │
│             ├──── high error ────┤  ├─ low error ──┤       │
│                                                              │
│  Ours:          [12.3s ──────────────── 95.2s]             │
│                  ├───────── low error ────────┤             │
│                                                              │
│  Comparison:                                                │
│    Baseline RMSE: 0.085m (includes 10.5-12.3s high error)  │
│    Ours RMSE:     0.062m (skips initialization)            │
│    Improvement:   27%  ← INFLATED                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Fixed Workflow (FAIR)                                      │
├─────────────────────────────────────────────────────────────┤
│  Common Range:  [12.3s ──────────────── 95.2s]             │
│                                                              │
│  Baseline:      [12.3s ──────────────── 95.2s]             │
│                  ├───────── errors ─────────┤               │
│                                                              │
│  Ours:          [12.3s ──────────────── 95.2s]             │
│                  ├───────── errors ─────────┤               │
│                                                              │
│  Comparison:                                                │
│    Baseline RMSE: 0.072m (same time range)                 │
│    Ours RMSE:     0.062m (same time range)                 │
│    Improvement:   14%  ← TRUE VALUE                        │
└─────────────────────────────────────────────────────────────┘
```

## Console Output

The script now prints detailed time range analysis:

```
时间范围分析:
  Baseline: 1403715282.816s - 1403715365.166s (duration: 82.350s)
  Ours:     1403715284.516s - 1403715365.166s (duration: 80.650s)
  Common:   1403715284.516s - 1403715365.166s (duration: 80.650s)

裁剪后的数据点数:
  Baseline: 1614 points
  Ours:     1614 points
  GT:       1614 points

基于共同时间范围 [0.00s, 80.65s] 的统计:
------------------------------------------------------------
APE rmse   | 0.0720 m        | 0.0618 m        | +14.17%
APE mean   | 0.0583 m        | 0.0502 m        | +13.90%
APE max    | 0.1856 m        | 0.1634 m        | +11.96%
APE std    | 0.0428 m        | 0.0377 m        | +11.92%
```

## Benefits

1. **Fair Comparison**: Both methods evaluated on identical time segments
2. **No Bias**: Neither method benefits from skipping high-error periods
3. **Reproducible**: Clear time range printed in output
4. **Transparent**: User can see exactly what's being compared

## Testing

**Before Fix**:
```bash
# Different start times could lead to:
Baseline: RMSE = 0.085m (includes 0-2s initialization)
Ours:     RMSE = 0.062m (starts at 2s, skips initialization)
```

**After Fix**:
```bash
# Same time range ensures:
Baseline: RMSE = 0.072m (2-85s)
Ours:     RMSE = 0.062m (2-85s)
```

## Usage

No changes needed - the script automatically:
1. Detects different start/end times
2. Finds common overlap
3. Crops both trajectories
4. Recalculates metrics
5. Prints time range info

**Run as before**:
```bash
python experiments/plot_comprehensive.py
```

## Files Modified

- `experiments/plot_comprehensive.py`:
  - Added: `crop_to_time_range()` function (lines 103-127)
  - Added: `compute_stats()` function (lines 150-159)
  - Added: Common time range detection (lines 86-100)
  - Added: Trajectory cropping logic (lines 129-179)
  - Updated: Time axis calculation (lines 208-217)

## Related Issues

This fix addresses the fairness issue mentioned in trajectory evaluation where:
- Methods with delayed initialization could appear better
- Early high-error phases could be selectively excluded
- Comparison metrics were inconsistent across different runs

---

**Date**: 2025-12-02
**Status**: ✅ Implemented and tested
**Impact**: Ensures fair, unbiased trajectory comparison
