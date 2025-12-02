import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from evo.tools import file_interface
from evo.core import sync, trajectory, metrics
from evo.core.geometry import umeyama_alignment

# ================= 1. 配置区域 (请修改这里) =================
# 路径配置
GT_FILE = "/home/linux/mxm/data/Euroc/MH_05_difficult/mav0/state_groundtruth_estimate0/data.tum"
BASELINE_FILE = "/home/linux/mxm/output/experiment_backend/MH05_test/1202/o-o/vins_closed_loop.tum" # 原始 VINS
OURS_FILE = "/home/linux/mxm/output/experiment_backend/MH05_test/1202/o-d/vins_closed_loop.tum" # 你的方法

# GT_FILE = "/home/linux/mxm/data/Euroc/V2_03_difficult/mav0/state_groundtruth_estimate0/data.tum"
# BASELINE_FILE = "/home/linux/mxm/output/experiment_backend/V203_test/1202/o-o/vins_closed_loop.tum" # 原始 VINS
# OURS_FILE = "/home/linux/mxm/output/experiment_backend/V203_test/1202/weight_5/o-d/vins_closed_loop.tum" # 你的方法

# 绘图美化配置 (符合学术出版标准)
config = {
    "font.family": 'serif',
    "font.serif": ['DejaVu Serif', 'Liberation Serif', 'Times', 'Times New Roman'], # 论文标准字体（优先级递减）
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "figure.dpi": 300,
    "mathtext.fontset": 'dejavuserif'  # 数学公式字体
}
rcParams.update(config)

# ================= 2. 核心计算函数 =================

def process_trajectory(gt_file, est_file, name):
    """读取、同步、对齐并计算APE/RPE"""
    # 1. 读取
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    
    # 2. 同步 (基于时间戳)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.01)
    
    # 3. SE(3) 对齐 (不改变尺度，符合 VIO 评估标准)
    r_a, t_a, s_a = umeyama_alignment(traj_est.positions_xyz.T, traj_ref.positions_xyz.T, with_scale=False)
    traj_est_aligned = trajectory.Trajectory(
        positions_xyz=np.dot(r_a, traj_est.positions_xyz.T).T + t_a.reshape(1, 3),
        orientations_quat_wxyz=traj_est.orientations_quat_wxyz,
        timestamps=traj_est.timestamps
    )
    
    # 4. 计算 APE (全局一致性)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ape_stats = ape_metric.get_all_statistics()
    ape_data = ape_metric.get_result().np_arrays['error_array']
    
    # 5. 计算 RPE (局部平滑度/漂移率) - delta=1 (逐帧)
    # RPE 需要原始未对齐的数据来计算增量，或者对齐后的也可以（相对变换不变）
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0, delta_unit=metrics.Unit.frames, all_pairs=False)
    rpe_metric.process_data((traj_ref, traj_est_aligned))
    rpe_stats = rpe_metric.get_all_statistics()
    rpe_data = rpe_metric.get_result().np_arrays['error_array']

    return {
        "name": name,
        "traj_ref": traj_ref,
        "traj_est": traj_est_aligned,
        "ape_stats": ape_stats,
        "ape_data": ape_data,
        "rpe_stats": rpe_stats,
        "rpe_data": rpe_data,
        "timestamps": traj_est_aligned.timestamps - traj_est_aligned.timestamps[0] # 相对时间
    }

# ================= 3. 执行处理 =================
print("正在处理数据...")
baseline_res = process_trajectory(GT_FILE, BASELINE_FILE, "Original VINS")
ours_res = process_trajectory(GT_FILE, OURS_FILE, "Ours (Deep Fusion)")
gt_ref = baseline_res["traj_ref"] # GT 用谁的都一样，取第一个

# ================= 3.5. 找到两个轨迹的共同时间范围并裁剪 =================
# 问题：如果新方法比原始方法晚开始，会跳过原始方法中RMSE较高的初始化阶段
# 解决：找到两个轨迹都有值的时间范围，只在这个范围内计算RMSE和绘图

# 1. 找到共同时间范围
baseline_start = baseline_res['traj_est'].timestamps[0]
baseline_end = baseline_res['traj_est'].timestamps[-1]
ours_start = ours_res['traj_est'].timestamps[0]
ours_end = ours_res['traj_est'].timestamps[-1]

# 共同起始时间：取较晚开始的那个
common_start = max(baseline_start, ours_start)
# 共同结束时间：取较早结束的那个
common_end = min(baseline_end, ours_end)

print(f"\n时间范围分析:")
print(f"  Baseline: {baseline_start:.3f}s - {baseline_end:.3f}s (duration: {baseline_end-baseline_start:.3f}s)")
print(f"  Ours:     {ours_start:.3f}s - {ours_end:.3f}s (duration: {ours_end-ours_start:.3f}s)")
print(f"  Common:   {common_start:.3f}s - {common_end:.3f}s (duration: {common_end-common_start:.3f}s)")

# 2. 裁剪到共同时间范围
def crop_to_time_range(traj, ape_data, rpe_data, start_time, end_time):
    """裁剪轨迹到指定时间范围"""
    # 找到时间范围内的索引
    mask = (traj.timestamps >= start_time) & (traj.timestamps <= end_time)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"No data in time range [{start_time}, {end_time}]")

    # 裁剪轨迹
    cropped_traj = trajectory.Trajectory(
        positions_xyz=traj.positions_xyz[mask],
        orientations_quat_wxyz=traj.orientations_quat_wxyz[mask],
        timestamps=traj.timestamps[mask]
    )

    # 裁剪APE数据（APE数组长度与轨迹点数相同）
    cropped_ape = ape_data[mask]

    # 裁剪RPE数据（RPE数组长度可能比轨迹点数少1，因为是相邻帧之间的误差）
    # 为了安全，取min确保不越界
    rpe_mask_len = min(len(rpe_data), len(mask))
    cropped_rpe = rpe_data[mask[:rpe_mask_len]]

    return cropped_traj, cropped_ape, cropped_rpe

# 裁剪baseline
baseline_traj_cropped, baseline_ape_cropped, baseline_rpe_cropped = crop_to_time_range(
    baseline_res['traj_est'], baseline_res['ape_data'], baseline_res['rpe_data'],
    common_start, common_end
)

# 裁剪ours
ours_traj_cropped, ours_ape_cropped, ours_rpe_cropped = crop_to_time_range(
    ours_res['traj_est'], ours_res['ape_data'], ours_res['rpe_data'],
    common_start, common_end
)

# 裁剪GT（对应到共同时间范围）
gt_mask = (gt_ref.timestamps >= common_start) & (gt_ref.timestamps <= common_end)
gt_cropped = trajectory.Trajectory(
    positions_xyz=gt_ref.positions_xyz[gt_mask],
    orientations_quat_wxyz=gt_ref.orientations_quat_wxyz[gt_mask],
    timestamps=gt_ref.timestamps[gt_mask]
)

# 3. 重新计算裁剪后的统计数据
def compute_stats(data):
    """计算统计数据"""
    return {
        'rmse': np.sqrt(np.mean(data**2)),
        'mean': np.mean(data),
        'max': np.max(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data)
    }

baseline_ape_stats_cropped = compute_stats(baseline_ape_cropped)
ours_ape_stats_cropped = compute_stats(ours_ape_cropped)
baseline_rpe_stats_cropped = compute_stats(baseline_rpe_cropped)
ours_rpe_stats_cropped = compute_stats(ours_rpe_cropped)

# 更新结果字典（用裁剪后的数据）
baseline_res['traj_est'] = baseline_traj_cropped
baseline_res['ape_data'] = baseline_ape_cropped
baseline_res['rpe_data'] = baseline_rpe_cropped
baseline_res['ape_stats'] = baseline_ape_stats_cropped
baseline_res['rpe_stats'] = baseline_rpe_stats_cropped

ours_res['traj_est'] = ours_traj_cropped
ours_res['ape_data'] = ours_ape_cropped
ours_res['rpe_data'] = ours_rpe_cropped
ours_res['ape_stats'] = ours_ape_stats_cropped
ours_res['rpe_stats'] = ours_rpe_stats_cropped

gt_ref = gt_cropped

print(f"\n裁剪后的数据点数:")
print(f"  Baseline: {len(baseline_ape_cropped)} points")
print(f"  Ours:     {len(ours_ape_cropped)} points")
print(f"  GT:       {len(gt_cropped.timestamps)} points")

# 打印数值结果供论文表格使用（基于裁剪后的共同时间范围）
print(f"\n{'Metric':<10} | {'Original':<15} | {'Ours':<15} | {'Improvement':<10}")
print("-" * 60)
print("基于共同时间范围 [{:.2f}s, {:.2f}s] 的统计:".format(
    common_start - common_start, common_end - common_start))
print("-" * 60)
metrics_to_show = ['rmse', 'mean', 'max', 'std']
for m in metrics_to_show:
    v1 = baseline_res['ape_stats'][m]
    v2 = ours_res['ape_stats'][m]
    imp = (v1 - v2) / v1 * 100
    print(f"APE {m:<6} | {v1:.4f} m        | {v2:.4f} m        | {imp:+.2f}%")
print("-" * 60)
for m in metrics_to_show:
    v1 = baseline_res['rpe_stats'][m]
    v2 = ours_res['rpe_stats'][m]
    imp = (v1 - v2) / v1 * 100
    print(f"RPE {m:<6} | {v1:.4f} m        | {v2:.4f} m        | {imp:+.2f}%")

# ================= 4. 绘图逻辑 =================
# 1. 获取全局统一的时间基准（使用共同时间范围的起始时间）
# 注意：gt_ref 已经是裁剪后的数据了
global_start_time = common_start

# 2. 计算绝对时间轴（都从共同起始时间开始，所以都从0开始）
time_baseline = baseline_res['traj_est'].timestamps - global_start_time
time_ours = ours_res['traj_est'].timestamps - global_start_time

print(f"\n绘图时间范围:")
print(f"  Time range: [0.0s, {common_end - common_start:.2f}s]")
print(f"  Baseline time: [{time_baseline[0]:.3f}s, {time_baseline[-1]:.3f}s]")
print(f"  Ours time:     [{time_ours[0]:.3f}s, {time_ours[-1]:.3f}s]")

# 3. 绘图
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3) # 2行3列布局

# --- 图1: 2D 轨迹对比 (Main Trajectory) ---
ax_traj = fig.add_subplot(gs[0, :2]) # 占第一行的前两列
ax_traj.plot(gt_ref.positions_xyz[:, 0], gt_ref.positions_xyz[:, 1], 'k--', label='Ground Truth', alpha=0.7)
ax_traj.plot(baseline_res['traj_est'].positions_xyz[:, 0], baseline_res['traj_est'].positions_xyz[:, 1], 
             '#1f77b4', label=f"Original (RMSE={baseline_res['ape_stats']['rmse']:.3f})")
ax_traj.plot(ours_res['traj_est'].positions_xyz[:, 0], ours_res['traj_est'].positions_xyz[:, 1], 
             '#d62728', label=f"Ours (RMSE={ours_res['ape_stats']['rmse']:.3f})")
ax_traj.set_title('(a) Trajectory Comparison (XY Plane)')
ax_traj.set_xlabel('x [m]')
ax_traj.set_ylabel('y [m]')
ax_traj.legend()
ax_traj.axis('equal')
ax_traj.grid(linestyle=':', alpha=0.5)

# --- 图2: 箱线图 (Boxplot) - 统计分布 ---
ax_box = fig.add_subplot(gs[0, 2]) # 占第一行第3列
data_to_plot = [baseline_res['ape_data'], ours_res['ape_data']]
# 自定义箱线图颜色
box = ax_box.boxplot(data_to_plot, patch_artist=True, labels=['Original', 'Ours'], widths=0.5, showfliers=False)
colors = ['#1f77b4', '#d62728']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax_box.set_title('(b) APE Distribution')
ax_box.set_ylabel('Translation Error [m]')
ax_box.grid(axis='y', linestyle=':', alpha=0.5)

# --- 图3: APE 随时间变化 (Error over Time) ---
ax_ape = fig.add_subplot(gs[1, :]) # 占整个第二行
ax_ape.plot(time_baseline, baseline_res['ape_data'], '#1f77b4', alpha=0.6, label='Original Error')
ax_ape.plot(time_ours, ours_res['ape_data'], '#d62728', alpha=0.9, linewidth=1.2, label='Ours Error')
ax_ape.set_title('(c) Absolute Pose Error (APE) Evolution')
ax_ape.set_xlabel('Time [s]')
ax_ape.set_ylabel('Error [m]')
ax_ape.legend()
ax_ape.grid(linestyle=':', alpha=0.5)
ax_ape.set_xlim(left=0)

plt.tight_layout()
plt.savefig("/home/linux/mxm/proj/VINS-Mo/src/VINS-Mo/experiments/thesis_evaluation_result.png", bbox_inches='tight')
print("\n图表已保存为 thesis_evaluation_result.png")
plt.show()