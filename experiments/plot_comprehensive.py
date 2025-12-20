import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from evo.tools import file_interface
from evo.core import sync, trajectory, metrics
from evo.core.geometry import umeyama_alignment

# ================= 1. 配置区域 (请修改这里) =================
# 路径配置
# GT_FILE = "/home/linux/mxm/data/TUM/room2/data.tum"
# BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/room2_test/1216/o-o/0/vins_closed_loop.tum"
# OURS_PATH = "/home/linux/mxm/output/experiments_backend/room2_test/1216/o-d/k_10/0/"

# GT_FILE = "/home/linux/mxm/data/EuRoC/MH_04_difficult/mav0/state_groundtruth_estimate0/data.tum"
# BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/MH04_test/1216/new/o-o/0/vins_closed_loop.tum"
# OURS_PATH = "/home/linux/mxm/output/experiments_backend/MH04_test/1216/new/o-d/k_15/0/"

GT_FILE = "/home/linux/mxm/data/groundtruth/per-sequence/corridor1-2/groundtruth.txt"
BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/corridor1-2/1219/o-o/test/vins_closed_loop.tum"
OURS_PATH = "/home/linux/mxm/output/experiments_backend/corridor1-2/1219/o-d/test/"

# GT_FILE = "/home/linux/mxm/data/groundtruth/per-sequence/market1-1/groundtruth.txt"
# BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/market1-1/1219/o-o/0/vins_closed_loop.tum"
# OURS_PATH = "/home/linux/mxm/output/experiments_backend/market1-1/1219/o-d/k_5/1/"

# GT_FILE = "/home/linux/mxm/data/groundtruth/per-sequence/office1-5/groundtruth.txt"
# BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/office1-5/1219/o-o/0/vins_closed_loop.tum"
# OURS_PATH = "/home/linux/mxm/output/experiments_backend/office1-5/1219/o-d/k_10/0/"

# GT_FILE = "/home/linux/mxm/data/groundtruth/per-sequence/cafe1-2/groundtruth.txt"
# BASELINE_FILE = "/home/linux/mxm/output/experiments_backend/cafe1-2/1220/o-o/0/vins_closed_loop.tum"
# OURS_PATH = "/home/linux/mxm/output/experiments_backend/cafe1-2/1220/o-d/k_5/0/"

OURS_FILE = OURS_PATH + "vins_closed_loop.tum"

# 输出配置
OUTPUT_PATH = OURS_PATH  # 保存结果的路径
OUTPUT_PREFIX = "evaluation_result"  # 输出文件前缀

# 绘图美化配置 (符合 IEEE/ACM 学术出版标准)
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times'],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.labelweight": 'normal',
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": 'png',
    "mathtext.fontset": 'stix',  # 更接近 Times New Roman 的数学字体
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ':',
}
rcParams.update(config)

# 颜色方案 (色盲友好 + 高对比度)
COLORS = {
    'gt': '#000000',        # 黑色 - Ground Truth
    'baseline': '#0173B2',  # 蓝色 - Baseline
    'ours': '#D62728',      # 橙色 - Ours
    'error': '#CC78BC',     # 紫色 - Error
}

# ================= 2. 文件检测函数 =================

def check_files_existence():
    """检测哪些文件存在，返回文件状态字典"""
    file_status = {
        'gt': os.path.exists(GT_FILE) if GT_FILE else False,
        'baseline': os.path.exists(BASELINE_FILE) if BASELINE_FILE else False,
        'ours': os.path.exists(OURS_FILE) if OURS_FILE else False,
    }

    print("=" * 70)
    print("文件状态检测:")
    print("-" * 70)
    print(f"  Ground Truth:  {'✓ 存在' if file_status['gt'] else '✗ 不存在'}")
    if file_status['gt']:
        print(f"    路径: {GT_FILE}")
    print(f"  Baseline:      {'✓ 存在' if file_status['baseline'] else '✗ 不存在'}")
    if file_status['baseline']:
        print(f"    路径: {BASELINE_FILE}")
    print(f"  Ours:          {'✓ 存在' if file_status['ours'] else '✗ 不存在'}")
    if file_status['ours']:
        print(f"    路径: {OURS_FILE}")
    print("=" * 70)

    # 检查是否至少有一个估计轨迹
    if not file_status['baseline'] and not file_status['ours']:
        raise ValueError("错误：至少需要一个估计轨迹文件 (Baseline 或 Ours)")

    return file_status

# ================= 3. 核心计算函数 =================

def process_trajectory(gt_file, est_file, name, align=True):
    """读取、同步、对齐并计算APE/RPE"""
    # 1. 读取
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    result = {
        "name": name,
        "traj_est_original": traj_est,
        "traj_est": None,
        "traj_ref": None,
        "ape_stats": None,
        "ape_data": None,
        "rpe_stats": None,
        "rpe_data": None,
        "timestamps": traj_est.timestamps - traj_est.timestamps[0]  # 相对时间
    }

    # 如果没有 GT，直接返回估计轨迹
    if gt_file is None or not os.path.exists(gt_file):
        result["traj_est"] = traj_est
        return result

    # 2. 读取 GT
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    # 3. 同步 (基于时间戳)
    try:
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.01)
    except Exception as e:
        print(f"警告：{name} 时间同步失败: {e}")
        result["traj_est"] = traj_est
        return result

    result["traj_ref"] = traj_ref

    # 4. SE(3) 对齐 (不改变尺度，符合 VIO 评估标准)
    if align:
        try:
            r_a, t_a, _ = umeyama_alignment(traj_est.positions_xyz.T, traj_ref.positions_xyz.T, with_scale=False)
            traj_est_aligned = trajectory.Trajectory(
                positions_xyz=np.dot(r_a, traj_est.positions_xyz.T).T + t_a.reshape(1, 3),
                orientations_quat_wxyz=traj_est.orientations_quat_wxyz,
                timestamps=traj_est.timestamps
            )
        except Exception as e:
            print(f"警告：{name} 对齐失败: {e}")
            traj_est_aligned = traj_est
    else:
        traj_est_aligned = traj_est

    result["traj_est"] = traj_est_aligned

    # 5. 计算 APE (全局一致性)
    try:
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref, traj_est_aligned))
        result["ape_stats"] = ape_metric.get_all_statistics()
        result["ape_data"] = ape_metric.get_result().np_arrays['error_array']
    except Exception as e:
        print(f"警告：{name} APE 计算失败: {e}")

    # 6. 计算 RPE (局部平滑度/漂移率) - delta=1 (逐帧)
    try:
        rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0,
                                 delta_unit=metrics.Unit.frames, all_pairs=False)
        rpe_metric.process_data((traj_ref, traj_est_aligned))
        result["rpe_stats"] = rpe_metric.get_all_statistics()
        result["rpe_data"] = rpe_metric.get_result().np_arrays['error_array']
    except Exception as e:
        print(f"警告：{name} RPE 计算失败: {e}")

    return result

# ================= 4. 时间对齐函数 =================

def crop_to_common_time_range(results_dict, gt_ref):
    """将所有轨迹裁剪到共同时间范围"""
    available_trajs = [v for v in results_dict.values() if v is not None and v['traj_est'] is not None]

    if len(available_trajs) == 0:
        return results_dict, gt_ref

    # 找到共同时间范围
    starts = [r['traj_est'].timestamps[0] for r in available_trajs]
    ends = [r['traj_est'].timestamps[-1] for r in available_trajs]

    common_start = max(starts)
    common_end = min(ends)

    print(f"\n时间范围分析:")
    for name, res in results_dict.items():
        if res is not None and res['traj_est'] is not None:
            start = res['traj_est'].timestamps[0]
            end = res['traj_est'].timestamps[-1]
            print(f"  {name:<12}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)")
    print(f"  {'Common':<12}: {common_start:.3f}s - {common_end:.3f}s (duration: {common_end-common_start:.3f}s)")

    # 裁剪函数
    def crop_single_result(result, start_time, end_time):
        if result is None or result['traj_est'] is None:
            return result

        traj = result['traj_est']
        mask = (traj.timestamps >= start_time) & (traj.timestamps <= end_time)

        if np.sum(mask) == 0:
            print(f"警告：{result['name']} 在共同时间范围内无数据")
            return result

        # 裁剪轨迹
        result['traj_est'] = trajectory.Trajectory(
            positions_xyz=traj.positions_xyz[mask],
            orientations_quat_wxyz=traj.orientations_quat_wxyz[mask],
            timestamps=traj.timestamps[mask]
        )

        # 更新相对时间
        result['timestamps'] = result['traj_est'].timestamps - result['traj_est'].timestamps[0]

        # 裁剪误差数据
        if result['ape_data'] is not None:
            result['ape_data'] = result['ape_data'][mask]
            result['ape_stats'] = compute_stats(result['ape_data'])

        if result['rpe_data'] is not None:
            rpe_mask_len = min(len(result['rpe_data']), len(mask))
            result['rpe_data'] = result['rpe_data'][mask[:rpe_mask_len]]
            result['rpe_stats'] = compute_stats(result['rpe_data'])

        return result

    # 裁剪所有结果
    for name in results_dict:
        results_dict[name] = crop_single_result(results_dict[name], common_start, common_end)

    # 裁剪 GT
    if gt_ref is not None:
        gt_mask = (gt_ref.timestamps >= common_start) & (gt_ref.timestamps <= common_end)
        if np.sum(gt_mask) > 0:
            gt_ref = trajectory.Trajectory(
                positions_xyz=gt_ref.positions_xyz[gt_mask],
                orientations_quat_wxyz=gt_ref.orientations_quat_wxyz[gt_mask],
                timestamps=gt_ref.timestamps[gt_mask]
            )

    return results_dict, gt_ref, common_start

def compute_stats(data):
    """计算统计数据"""
    if data is None or len(data) == 0:
        return None
    return {
        'rmse': np.sqrt(np.mean(data**2)),
        'mean': np.mean(data),
        'max': np.max(data),
        'std': np.std(data),
        'median': np.median(data),
        'min': np.min(data)
    }

# ================= 5. 绘图函数 =================

def plot_trajectory_only(results_dict, output_path):
    """仅绘制轨迹图（无GT情况）"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for name, res in results_dict.items():
        if res is not None and res['traj_est'] is not None:
            traj = res['traj_est']
            color = COLORS.get(name, '#333333')
            ax.plot(traj.positions_xyz[:, 0], traj.positions_xyz[:, 1],
                   color=color, label=name.replace('_', ' ').title(), linewidth=2.5, alpha=0.8)

            # 标记起点和终点
            ax.scatter(traj.positions_xyz[0, 0], traj.positions_xyz[0, 1],
                      color=color, marker='o', s=150, zorder=5, edgecolors='white', linewidths=2,
                      label=f'{name.title()} Start')
            ax.scatter(traj.positions_xyz[-1, 0], traj.positions_xyz[-1, 1],
                      color=color, marker='s', s=150, zorder=5, edgecolors='white', linewidths=2,
                      label=f'{name.title()} End')

    ax.set_xlabel('X Position [m]', fontweight='bold', fontsize=13)
    ax.set_ylabel('Y Position [m]', fontweight='bold', fontsize=13)
    ax.set_title('Estimated Trajectories (XY Plane)', fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='best', framealpha=0.95, fontsize=11)
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.4)

    # 添加方向箭头
    for name, res in results_dict.items():
        if res is not None and res['traj_est'] is not None:
            traj = res['traj_est']
            # 添加多个方向箭头以显示轨迹方向
            n_arrows = 3
            indices = np.linspace(len(traj.positions_xyz) // 4,
                                 3 * len(traj.positions_xyz) // 4,
                                 n_arrows, dtype=int)
            for idx in indices:
                if idx > 0 and idx < len(traj.positions_xyz):
                    dx = traj.positions_xyz[idx, 0] - traj.positions_xyz[idx-1, 0]
                    dy = traj.positions_xyz[idx, 1] - traj.positions_xyz[idx-1, 1]
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 0:
                        ax.arrow(traj.positions_xyz[idx-1, 0], traj.positions_xyz[idx-1, 1],
                                dx, dy, head_width=0.2, head_length=0.15,
                                fc=COLORS.get(name, '#333333'),
                                ec=COLORS.get(name, '#333333'), alpha=0.6, zorder=4,
                                width=0.02)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_trajectory.png"),
                bbox_inches='tight', dpi=300)
    print(f"✓ 轨迹图已保存: {OUTPUT_PREFIX}_trajectory.png")
    plt.close()

def plot_single_method_with_gt(result, gt_ref, output_path, common_start):
    """绘制单个方法与GT的对比（分成多个独立图表）"""
    has_errors = result['ape_data'] is not None
    method_name = result['name'].lower().replace(' ', '_')

    # === 图1：轨迹对比 ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # GT轨迹
    ax.plot(gt_ref.positions_xyz[:, 0], gt_ref.positions_xyz[:, 1],
            color=COLORS['gt'], linestyle='--', label='Ground Truth',
            linewidth=3, alpha=0.8, zorder=1)

    # 估计轨迹
    traj = result['traj_est']
    color = COLORS.get(result['name'].lower(), COLORS['ours'])
    ax.plot(traj.positions_xyz[:, 0], traj.positions_xyz[:, 1],
            color=color, label=result['name'], linewidth=2.5, zorder=2)

    # 起点标记
    ax.scatter(gt_ref.positions_xyz[0, 0], gt_ref.positions_xyz[0, 1],
               color=COLORS['gt'], marker='o', s=200, zorder=5,
               edgecolors='white', linewidths=2.5, label='Start')

    # 终点标记
    ax.scatter(gt_ref.positions_xyz[-1, 0], gt_ref.positions_xyz[-1, 1],
               color=COLORS['gt'], marker='s', s=200, zorder=5,
               edgecolors='white', linewidths=2.5, label='End (GT)')
    ax.scatter(traj.positions_xyz[-1, 0], traj.positions_xyz[-1, 1],
               color=color, marker='s', s=200, zorder=5,
               edgecolors='white', linewidths=2.5, label=f'End ({result["name"]})')

    # 添加RMSE信息
    if has_errors:
        rmse_text = f"APE RMSE: {result['ape_stats']['rmse']:.4f} m"
        ax.text(0.02, 0.98, rmse_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))

    ax.set_xlabel('X Position [m]', fontweight='bold', fontsize=13)
    ax.set_ylabel('Y Position [m]', fontweight='bold', fontsize=13)
    ax.set_title(f'Trajectory Comparison: {result["name"]} vs Ground Truth',
                fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='best', framealpha=0.95, fontsize=11)
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_{method_name}_trajectory.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ 轨迹图: {OUTPUT_PREFIX}_{method_name}_trajectory.png")
    plt.close()

    if not has_errors:
        return

    # === 图2：APE 时序图（大图，细节清晰）===
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    time_axis = result['traj_est'].timestamps - common_start

    # 绘制APE曲线
    ax.plot(time_axis, result['ape_data'], color=color,
           alpha=0.8, linewidth=2, label='APE', zorder=2)

    # 均值线
    ax.axhline(y=result['ape_stats']['mean'], color='red',
              linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {result["ape_stats"]["mean"]:.4f} m',
              zorder=1)

    # 标准差区域
    ax.fill_between(time_axis,
                   result['ape_stats']['mean'] - result['ape_stats']['std'],
                   result['ape_stats']['mean'] + result['ape_stats']['std'],
                   color=color, alpha=0.25, label=f'±1σ: {result["ape_stats"]["std"]:.4f} m',
                   zorder=0)

    # 添加统计信息框
    stats_text = f"RMSE:   {result['ape_stats']['rmse']:.4f} m\n"
    stats_text += f"Mean:   {result['ape_stats']['mean']:.4f} m\n"
    stats_text += f"Median: {result['ape_stats']['median']:.4f} m\n"
    stats_text += f"Std:    {result['ape_stats']['std']:.4f} m\n"
    stats_text += f"Max:    {result['ape_stats']['max']:.4f} m\n"
    stats_text += f"Min:    {result['ape_stats']['min']:.4f} m"

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8),
           family='monospace')

    ax.set_xlabel('Time [s]', fontweight='bold', fontsize=13)
    ax.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=13)
    ax.set_title(f'Absolute Pose Error (APE) - {result["name"]}',
                fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_{method_name}_ape_evolution.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ APE时序图: {OUTPUT_PREFIX}_{method_name}_ape_evolution.png")
    plt.close()

    # === 图3：APE 分布箱线图 + 直方图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：箱线图
    box = ax1.boxplot([result['ape_data']], patch_artist=True,
                      labels=[result['name']], widths=0.5, showfliers=True,
                      flierprops=dict(marker='o', markerfacecolor=color, markersize=4, alpha=0.5))
    box['boxes'][0].set_facecolor(color)
    box['boxes'][0].set_alpha(0.6)

    # 添加均值点
    ax1.scatter([1], [result['ape_stats']['mean']], color='red', s=100,
               zorder=3, marker='D', edgecolors='white', linewidths=1.5,
               label='Mean')

    ax1.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=13)
    ax1.set_title('APE Distribution (Boxplot)', fontweight='bold', pad=15, fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)

    # 子图2：直方图
    ax2.hist(result['ape_data'], bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(result['ape_stats']['mean'], color='red', linestyle='--',
               linewidth=2, label=f'Mean: {result["ape_stats"]["mean"]:.4f} m')
    ax2.axvline(result['ape_stats']['median'], color='green', linestyle='--',
               linewidth=2, label=f'Median: {result["ape_stats"]["median"]:.4f} m')

    ax2.set_xlabel('Translation Error [m]', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=13)
    ax2.set_title('APE Distribution (Histogram)', fontweight='bold', pad=15, fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_{method_name}_ape_distribution.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ APE分布图: {OUTPUT_PREFIX}_{method_name}_ape_distribution.png")
    plt.close()

    # === 图4：RPE 分析（如果有）===
    if result['rpe_data'] is not None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        time_axis_rpe = result['traj_est'].timestamps[:len(result['rpe_data'])] - common_start

        ax.plot(time_axis_rpe, result['rpe_data'], color=color,
               alpha=0.8, linewidth=2, label='RPE')
        ax.axhline(y=result['rpe_stats']['mean'], color='red',
                  linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Mean: {result["rpe_stats"]["mean"]:.4f} m')

        # 统计信息
        stats_text = f"RMSE:   {result['rpe_stats']['rmse']:.4f} m\n"
        stats_text += f"Mean:   {result['rpe_stats']['mean']:.4f} m\n"
        stats_text += f"Median: {result['rpe_stats']['median']:.4f} m\n"
        stats_text += f"Std:    {result['rpe_stats']['std']:.4f} m"

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8),
               family='monospace')

        ax.set_xlabel('Time [s]', fontweight='bold', fontsize=13)
        ax.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=13)
        ax.set_title(f'Relative Pose Error (RPE) - {result["name"]}',
                    fontweight='bold', pad=20, fontsize=15)
        ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_{method_name}_rpe_evolution.png"),
                    bbox_inches='tight', dpi=300)
        print(f"  ✓ RPE时序图: {OUTPUT_PREFIX}_{method_name}_rpe_evolution.png")
        plt.close()

def plot_comparison_with_gt(baseline_res, ours_res, gt_ref, output_path, common_start):
    """绘制完整的对比图（Baseline + Ours + GT）- 分成多个独立图表"""

    # === 图1：轨迹对比（大图） ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # GT轨迹
    ax.plot(gt_ref.positions_xyz[:, 0], gt_ref.positions_xyz[:, 1],
            color=COLORS['gt'], linestyle='--', label='Ground Truth',
            linewidth=3.5, alpha=0.9, zorder=1)

    # Baseline轨迹
    if baseline_res is not None:
        traj_b = baseline_res['traj_est']
        label_b = "Baseline"
        if baseline_res['ape_stats']:
            label_b += f" (RMSE: {baseline_res['ape_stats']['rmse']:.4f} m)"
        ax.plot(traj_b.positions_xyz[:, 0], traj_b.positions_xyz[:, 1],
                color=COLORS['baseline'], label=label_b, linewidth=2.5, alpha=0.75, zorder=2)

    # Ours轨迹
    if ours_res is not None:
        traj_o = ours_res['traj_est']
        label_o = "Ours"
        if ours_res['ape_stats']:
            label_o += f" (RMSE: {ours_res['ape_stats']['rmse']:.4f} m)"
        ax.plot(traj_o.positions_xyz[:, 0], traj_o.positions_xyz[:, 1],
                color=COLORS['ours'], label=label_o, linewidth=2.5, alpha=0.9, zorder=3)

    # 起点和终点标记
    ax.scatter(gt_ref.positions_xyz[0, 0], gt_ref.positions_xyz[0, 1],
               color='green', marker='o', s=250, zorder=6,
               edgecolors='white', linewidths=3, label='Start', alpha=0.9)
    ax.scatter(gt_ref.positions_xyz[-1, 0], gt_ref.positions_xyz[-1, 1],
               color='red', marker='s', s=250, zorder=6,
               edgecolors='white', linewidths=3, label='End', alpha=0.9)

    ax.set_xlabel('X Position [m]', fontweight='bold', fontsize=14)
    ax.set_ylabel('Y Position [m]', fontweight='bold', fontsize=14)
    ax.set_title('Trajectory Comparison (XY Plane)', fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', framealpha=0.95, fontsize=12, shadow=True)
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_trajectory.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ 轨迹对比图: {OUTPUT_PREFIX}_comparison_trajectory.png")
    plt.close()

    # === 图2：APE时序对比（双曲线大图）===
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    if baseline_res and baseline_res['ape_data'] is not None:
        time_b = baseline_res['traj_est'].timestamps - common_start
        ax.plot(time_b, baseline_res['ape_data'], color=COLORS['baseline'],
               alpha=0.7, linewidth=2.5, label=f'Baseline (RMSE: {baseline_res["ape_stats"]["rmse"]:.4f} m)',
               zorder=2)
        ax.axhline(y=baseline_res['ape_stats']['mean'], color=COLORS['baseline'],
                  linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    if ours_res and ours_res['ape_data'] is not None:
        time_o = ours_res['traj_est'].timestamps - common_start
        ax.plot(time_o, ours_res['ape_data'], color=COLORS['ours'],
               alpha=0.9, linewidth=2.5, label=f'Ours (RMSE: {ours_res["ape_stats"]["rmse"]:.4f} m)',
               zorder=3)
        ax.axhline(y=ours_res['ape_stats']['mean'], color=COLORS['ours'],
                  linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

    # 计算改进
    if baseline_res and ours_res and baseline_res['ape_stats'] and ours_res['ape_stats']:
        improvement = (baseline_res['ape_stats']['rmse'] - ours_res['ape_stats']['rmse']) / \
                     baseline_res['ape_stats']['rmse'] * 100
        improvement_text = f"Improvement: {improvement:+.2f}%"
        color_imp = 'green' if improvement > 0 else 'red'
        ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
               fontsize=13, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color_imp, alpha=0.3, pad=0.8),
               color=color_imp)

    ax.set_xlabel('Time [s]', fontweight='bold', fontsize=14)
    ax.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=14)
    ax.set_title('Absolute Pose Error (APE) Comparison', fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=12, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_ape_evolution.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ APE时序对比: {OUTPUT_PREFIX}_comparison_ape_evolution.png")
    plt.close()

    # === 图3：APE箱线图对比 ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    data_to_plot = []
    labels = []
    colors_list = []

    if baseline_res and baseline_res['ape_data'] is not None:
        data_to_plot.append(baseline_res['ape_data'])
        labels.append('Baseline')
        colors_list.append(COLORS['baseline'])

    if ours_res and ours_res['ape_data'] is not None:
        data_to_plot.append(ours_res['ape_data'])
        labels.append('Ours')
        colors_list.append(COLORS['ours'])

    if len(data_to_plot) > 0:
        positions = range(1, len(data_to_plot) + 1)
        box = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                        positions=positions, widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=5, alpha=0.4))

        for patch, color in zip(box['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(2)

        # 添加均值点
        for i, data in enumerate(data_to_plot):
            ax.scatter([positions[i]], [np.mean(data)], color='red', s=150,
                      zorder=5, marker='D', edgecolors='white', linewidths=2,
                      label='Mean' if i == 0 else '')

    ax.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=14)
    ax.set_title('APE Distribution Comparison (Boxplot)', fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_ape_boxplot.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ APE箱线图对比: {OUTPUT_PREFIX}_comparison_ape_boxplot.png")
    plt.close()

    # === 图4：APE直方图对比（叠加）===
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    if baseline_res and baseline_res['ape_data'] is not None:
        ax.hist(baseline_res['ape_data'], bins=60, color=COLORS['baseline'], alpha=0.6,
               edgecolor='black', linewidth=0.5, label=f'Baseline (Mean: {baseline_res["ape_stats"]["mean"]:.4f} m)')

    if ours_res and ours_res['ape_data'] is not None:
        ax.hist(ours_res['ape_data'], bins=60, color=COLORS['ours'], alpha=0.7,
               edgecolor='black', linewidth=0.5, label=f'Ours (Mean: {ours_res["ape_stats"]["mean"]:.4f} m)')

    # 添加均值线
    if baseline_res and baseline_res['ape_stats']:
        ax.axvline(baseline_res['ape_stats']['mean'], color=COLORS['baseline'],
                  linestyle='--', linewidth=2.5, alpha=0.9)
    if ours_res and ours_res['ape_stats']:
        ax.axvline(ours_res['ape_stats']['mean'], color=COLORS['ours'],
                  linestyle='--', linewidth=2.5, alpha=0.9)

    ax.set_xlabel('Translation Error [m]', fontweight='bold', fontsize=14)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=14)
    ax.set_title('APE Distribution Comparison (Histogram)', fontweight='bold', pad=20, fontsize=16)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_ape_histogram.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ APE直方图对比: {OUTPUT_PREFIX}_comparison_ape_histogram.png")
    plt.close()

    # === 图5：RPE时序对比 ===
    if (baseline_res and baseline_res['rpe_data'] is not None) or \
       (ours_res and ours_res['rpe_data'] is not None):

        fig, ax = plt.subplots(1, 1, figsize=(16, 7))

        if baseline_res and baseline_res['rpe_data'] is not None:
            time_b_rpe = baseline_res['traj_est'].timestamps[:len(baseline_res['rpe_data'])] - common_start
            ax.plot(time_b_rpe, baseline_res['rpe_data'], color=COLORS['baseline'],
                   alpha=0.7, linewidth=2.5, label=f'Baseline (RMSE: {baseline_res["rpe_stats"]["rmse"]:.4f} m)')
            ax.axhline(y=baseline_res['rpe_stats']['mean'], color=COLORS['baseline'],
                      linestyle='--', linewidth=1.5, alpha=0.6)

        if ours_res and ours_res['rpe_data'] is not None:
            time_o_rpe = ours_res['traj_est'].timestamps[:len(ours_res['rpe_data'])] - common_start
            ax.plot(time_o_rpe, ours_res['rpe_data'], color=COLORS['ours'],
                   alpha=0.9, linewidth=2.5, label=f'Ours (RMSE: {ours_res["rpe_stats"]["rmse"]:.4f} m)')
            ax.axhline(y=ours_res['rpe_stats']['mean'], color=COLORS['ours'],
                      linestyle='--', linewidth=1.5, alpha=0.6)

        # 计算改进
        if baseline_res and ours_res and baseline_res['rpe_stats'] and ours_res['rpe_stats']:
            improvement_rpe = (baseline_res['rpe_stats']['rmse'] - ours_res['rpe_stats']['rmse']) / \
                             baseline_res['rpe_stats']['rmse'] * 100
            improvement_text = f"Improvement: {improvement_rpe:+.2f}%"
            color_imp = 'green' if improvement_rpe > 0 else 'red'
            ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
                   fontsize=13, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color_imp, alpha=0.3, pad=0.8),
                   color=color_imp)

        ax.set_xlabel('Time [s]', fontweight='bold', fontsize=14)
        ax.set_ylabel('Translation Error [m]', fontweight='bold', fontsize=14)
        ax.set_title('Relative Pose Error (RPE) Comparison', fontweight='bold', pad=20, fontsize=16)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=12, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_rpe_evolution.png"),
                    bbox_inches='tight', dpi=300)
        print(f"  ✓ RPE时序对比: {OUTPUT_PREFIX}_comparison_rpe_evolution.png")
        plt.close()

    # === 图6：统计对比表格（独立大图）===
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')

    # 构建详细的统计表格
    table_data = []
    table_data.append(['Metric', 'Baseline', 'Ours', 'Improvement', 'Status'])

    metrics_to_show = ['rmse', 'mean', 'median', 'std', 'max', 'min']
    metric_display = {
        'rmse': 'APE RMSE',
        'mean': 'APE Mean',
        'median': 'APE Median',
        'std': 'APE Std Dev',
        'max': 'APE Max',
        'min': 'APE Min'
    }

    for m in metrics_to_show:
        row = [metric_display[m]]

        if baseline_res and baseline_res['ape_stats']:
            v1 = baseline_res['ape_stats'][m]
            row.append(f"{v1:.5f} m")
        else:
            row.append('-')
            v1 = None

        if ours_res and ours_res['ape_stats']:
            v2 = ours_res['ape_stats'][m]
            row.append(f"{v2:.5f} m")
        else:
            row.append('-')
            v2 = None

        if v1 is not None and v2 is not None:
            imp = (v1 - v2) / v1 * 100
            row.append(f"{imp:+.2f}%")
            row.append('Better' if imp > 0 else 'Worse')
        else:
            row.append('-')
            row.append('-')

        table_data.append(row)

    # 添加RPE统计
    if (baseline_res and baseline_res['rpe_stats']) or (ours_res and ours_res['rpe_stats']):
        table_data.append(['', '', '', '', ''])  # 空行分隔

        rpe_metrics = ['rmse', 'mean', 'std']
        for m in rpe_metrics:
            row = [f'RPE {m.upper()}' if m == 'rmse' else f'RPE {m.capitalize()}']

            if baseline_res and baseline_res['rpe_stats']:
                v1 = baseline_res['rpe_stats'][m]
                row.append(f"{v1:.5f} m")
            else:
                row.append('-')
                v1 = None

            if ours_res and ours_res['rpe_stats']:
                v2 = ours_res['rpe_stats'][m]
                row.append(f"{v2:.5f} m")
            else:
                row.append('-')
                v2 = None

            if v1 is not None and v2 is not None:
                imp = (v1 - v2) / v1 * 100
                row.append(f"{imp:+.2f}%")
                row.append('Better' if imp > 0 else 'Worse')
            else:
                row.append('-')
                row.append('-')

            table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 美化表格
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1.5)

            if i == 0:  # 表头
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
            elif table_data[i][0] == '':  # 空行
                cell.set_facecolor('#F0F0F0')
            else:
                if j == 1:  # Baseline列
                    cell.set_facecolor('#E2EFDA')
                elif j == 2:  # Ours列
                    cell.set_facecolor('#FCE4D6')
                elif j == 3:  # Improvement列
                    if '-' not in table_data[i][j]:
                        try:
                            val = float(table_data[i][j].replace('%', ''))
                            if val > 0:
                                cell.set_facecolor('#C6EFCE')  # 绿色 - 改进
                            else:
                                cell.set_facecolor('#FFC7CE')  # 红色 - 退化
                        except:
                            pass
                elif j == 4:  # Status列
                    if table_data[i][j] == 'Better':
                        cell.set_facecolor('#C6EFCE')
                        cell.set_text_props(weight='bold', color='darkgreen')
                    elif table_data[i][j] == 'Worse':
                        cell.set_facecolor('#FFC7CE')
                        cell.set_text_props(weight='bold', color='darkred')

    ax.set_title('Performance Metrics Comparison', fontweight='bold',
                pad=20, fontsize=18, y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{OUTPUT_PREFIX}_comparison_statistics.png"),
                bbox_inches='tight', dpi=300)
    print(f"  ✓ 统计对比表: {OUTPUT_PREFIX}_comparison_statistics.png")
    plt.close()

# ================= 6. 统计信息打印 =================

def print_statistics(baseline_res, ours_res, common_start, common_end):
    """打印详细的统计信息"""
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    if common_start is not None and common_end is not None:
        print(f"共同时间范围: [0.00s, {common_end - common_start:.2f}s]")
        print("-" * 80)

    # 表头
    print(f"{'Metric':<15} | {'Baseline':<18} | {'Ours':<18} | {'Improvement':<12}")
    print("-" * 80)

    metrics_to_show = ['rmse', 'mean', 'std', 'max', 'median']

    # APE 指标
    print("APE (Absolute Pose Error):")
    for m in metrics_to_show:
        v1 = baseline_res['ape_stats'][m] if baseline_res and baseline_res['ape_stats'] else None
        v2 = ours_res['ape_stats'][m] if ours_res and ours_res['ape_stats'] else None

        v1_str = f"{v1:.6f} m" if v1 is not None else "N/A"
        v2_str = f"{v2:.6f} m" if v2 is not None else "N/A"

        if v1 is not None and v2 is not None:
            imp = (v1 - v2) / v1 * 100
            imp_str = f"{imp:+.2f}%"
        else:
            imp_str = "N/A"

        print(f"  {m:<13} | {v1_str:<18} | {v2_str:<18} | {imp_str:<12}")

    print("-" * 80)

    # RPE 指标
    print("RPE (Relative Pose Error):")
    for m in metrics_to_show:
        v1 = baseline_res['rpe_stats'][m] if baseline_res and baseline_res['rpe_stats'] else None
        v2 = ours_res['rpe_stats'][m] if ours_res and ours_res['rpe_stats'] else None

        v1_str = f"{v1:.6f} m" if v1 is not None else "N/A"
        v2_str = f"{v2:.6f} m" if v2 is not None else "N/A"

        if v1 is not None and v2 is not None:
            imp = (v1 - v2) / v1 * 100
            imp_str = f"{imp:+.2f}%"
        else:
            imp_str = "N/A"

        print(f"  {m:<13} | {v1_str:<18} | {v2_str:<18} | {imp_str:<12}")

    print("=" * 80)

# ================= 7. 主流程 =================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" " * 20 + "VINS-Mo 轨迹评估工具")
    print(" " * 15 + "(学术出版级可视化 + 自适应文件处理)")
    print("=" * 80)

    # 1. 检测文件
    file_status = check_files_existence()

    # 2. 创建输出目录
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 3. 处理数据
    print("\n正在处理轨迹数据...")
    print("-" * 70)

    results = {
        'baseline': None,
        'ours': None
    }

    gt_file = GT_FILE if file_status['gt'] else None

    if file_status['baseline']:
        print("  [1/2] 处理 Baseline 轨迹...")
        results['baseline'] = process_trajectory(gt_file, BASELINE_FILE, "Baseline")

    if file_status['ours']:
        idx = 2 if file_status['baseline'] else 1
        total = 2 if file_status['baseline'] else 1
        print(f"  [{idx}/{total}] 处理 Ours 轨迹...")
        results['ours'] = process_trajectory(gt_file, OURS_FILE, "Ours")

    # 4. 获取GT参考（如果存在）
    gt_ref = None
    if file_status['gt']:
        if results['baseline'] and results['baseline']['traj_ref'] is not None:
            gt_ref = results['baseline']['traj_ref']
        elif results['ours'] and results['ours']['traj_ref'] is not None:
            gt_ref = results['ours']['traj_ref']

    # 5. 时间对齐
    common_start = None
    if file_status['gt'] and (file_status['baseline'] or file_status['ours']):
        print("\n正在对齐时间范围...")
        results, gt_ref, common_start = crop_to_common_time_range(results, gt_ref)

    # 6. 打印统计信息
    if file_status['gt'] and file_status['baseline'] and file_status['ours']:
        print_statistics(results['baseline'], results['ours'], common_start,
                        results['ours']['traj_est'].timestamps[-1] if results['ours'] else None)

    # 7. 绘图
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("-" * 70)

    if not file_status['gt']:
        # 情况1：无GT，仅绘制轨迹
        print("  模式: 无真值 - 仅绘制轨迹图")
        plot_trajectory_only(results, OUTPUT_PATH)

    elif file_status['baseline'] and file_status['ours']:
        # 情况2：有GT + Baseline + Ours - 完整对比
        print("  模式: 完整对比 (GT + Baseline + Ours)")
        plot_comparison_with_gt(results['baseline'], results['ours'], gt_ref,
                               OUTPUT_PATH, common_start)

    elif file_status['baseline']:
        # 情况3：有GT + 仅Baseline
        print("  模式: 单方法分析 (GT + Baseline)")
        plot_single_method_with_gt(results['baseline'], gt_ref, OUTPUT_PATH, common_start)

    elif file_status['ours']:
        # 情况4：有GT + 仅Ours
        print("  模式: 单方法分析 (GT + Ours)")
        plot_single_method_with_gt(results['ours'], gt_ref, OUTPUT_PATH, common_start)

    print("=" * 80)
    print("✓ 所有任务完成!")
    print(f"  结果保存在: {OUTPUT_PATH}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
        plt.show()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
