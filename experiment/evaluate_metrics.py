import os
import numpy as np
import glob
from datetime import datetime
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

# ================= 配置区域 =================
# GT_FILE = "/home/linux/mxm/data/EuRoC/V1_01_easy/mav0/state_groundtruth_estimate0/data.tum" # 真值文件路径 (TUM格式)
# GT_FILE = "/home/linux/mxm/data/EuRoC/MH_05_difficult/mav0/state_groundtruth_estimate0/data.tum" # 真值文件路径 (TUM格式)
GT_FILE = "/home/linux/mxm/data/EuRoC/V1_03_difficult/mav0/state_groundtruth_estimate0/data.tum"
# GT_FILE = "/home/linux/mxm/data/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.tum"

# RESULT_DIR = "/home/linux/mxm/output/experiments_initial/Depth_anything/V101_test/1209/window_0_5s/deep/0" 
# RESULT_DIR = "/home/linux/mxm/output/experiments_initial/MiDaS/MH05_test/window_0_3s/deep" 
# RESULT_DIR = "/home/linux/mxm/output/experiments_initial/Depth_anything/MH05_test/1209/window_0_3s/deep" 
RESULT_DIR = "/home/linux/mxm/output/experiments_initial/Depth_anything/V103_test/1209/window_0.3s/original/1" 



OUTPUT_MD = os.path.join(RESULT_DIR, "evaluation_report.md")  # Markdown 输出路径

# 阈值设置
MIN_GT_MOTION = 0.15   # (米) 真值移动小于此值，视为静止/微动片段，不计入误差统计
DIVERGENCE_THRES = 5.0 # (米) ATE 大于此值，视为跟踪发散 (虽然初始化成功但跑飞了)
# ===========================================

def get_matrix_from_pose(traj, index=0):
    """提取 4x4 位姿矩阵"""
    t = traj.positions_xyz[index]
    q_wxyz = traj.orientations_quat_wxyz[index]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    R = Rotation.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def align_to_first_frame(traj_est, traj_ref):
    """手动第一帧对齐"""
    T_est_0 = get_matrix_from_pose(traj_est, 0)
    T_ref_0 = get_matrix_from_pose(traj_ref, 0)
    T_align = T_ref_0 @ np.linalg.inv(T_est_0)
    traj_est.transform(T_align)

def calculate_metrics(traj_est, traj_ref):
    """计算单段 Scale Error 和 ATE"""
    # 1. 检查真值运动激励
    dist_ref = np.sum(np.linalg.norm(traj_ref.positions_xyz[1:] - traj_ref.positions_xyz[:-1], axis=1))
    
    # 如果真值几乎没动，返回特殊标记
    if dist_ref < MIN_GT_MOTION:
        return None, None, True # is_static = True

    # 2. 对齐
    align_to_first_frame(traj_est, traj_ref)
    
    # 3. 计算尺度误差
    dist_est = np.sum(np.linalg.norm(traj_est.positions_xyz[1:] - traj_est.positions_xyz[:-1], axis=1))
    scale_error = abs(dist_est / dist_ref - 1.0) * 100
    
    # 4. 计算 ATE
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    ate_rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    
    return scale_error, ate_rmse, False

def generate_markdown_report(count_total, count_success, count_static, count_diverged,
                             stats_init_scale, stats_run_ate, detail_rows):
    """生成 Markdown 格式的实验报告"""

    # 提取实验配置信息
    exp_name = os.path.basename(os.path.dirname(RESULT_DIR))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(RESULT_DIR)))
    method_name = os.path.basename(RESULT_DIR)

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        # 标题和元数据
        f.write(f"# VINS Initialization Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 实验配置
        f.write("## Experiment Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Dataset | {dataset_name} |\n")
        f.write(f"| Experiment | {exp_name} |\n")
        f.write(f"| Method | {method_name} |\n")
        f.write(f"| Ground Truth | `{os.path.basename(GT_FILE)}` |\n")
        f.write(f"| Min Motion Threshold | {MIN_GT_MOTION} m |\n")
        f.write(f"| Divergence Threshold | {DIVERGENCE_THRES} m |\n\n")

        # 总体统计
        success_rate = count_success / count_total * 100 if count_total > 0 else 0
        valid_samples = len(stats_init_scale)

        f.write("## Overall Statistics\n\n")
        f.write("| Metric | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        f.write(f"| **Total Test Segments** | {count_total} | 100.0% |\n")
        f.write(f"| Initialization Success | {count_success} | {success_rate:.1f}% |\n")
        f.write(f"| - Static/Micro-motion (excluded) | {count_static} | {count_static/count_total*100:.1f}% |\n")
        f.write(f"| - Tracking Diverged (excluded) | {count_diverged} | {count_diverged/count_total*100:.1f}% |\n")
        f.write(f"| **Valid Evaluation Samples** | {valid_samples} | {valid_samples/count_total*100:.1f}% |\n\n")

        # 性能指标汇总
        if len(stats_init_scale) > 0:
            f.write("## Performance Metrics Summary\n\n")
            f.write("### Initialization Scale Error\n\n")
            f.write("*Measures geometric alignment accuracy during initialization phase*\n\n")

            scale_arr = np.array(stats_init_scale)
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Mean | {np.mean(scale_arr):.2f}% |\n")
            f.write(f"| **Median** | **{np.median(scale_arr):.2f}%** |\n")
            f.write(f"| Std Dev | {np.std(scale_arr):.2f}% |\n")
            f.write(f"| Min | {np.min(scale_arr):.2f}% |\n")
            f.write(f"| Max | {np.max(scale_arr):.2f}% |\n")
            f.write(f"| 25th Percentile | {np.percentile(scale_arr, 25):.2f}% |\n")
            f.write(f"| 75th Percentile | {np.percentile(scale_arr, 75):.2f}% |\n\n")

            f.write("### Tracking Accuracy (ATE over 10s window)\n\n")
            f.write("*Measures long-term tracking drift after initialization*\n\n")

            ate_arr = np.array(stats_run_ate)
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Mean | {np.mean(ate_arr):.3f} m |\n")
            f.write(f"| **Median** | **{np.median(ate_arr):.3f} m** |\n")
            f.write(f"| Std Dev | {np.std(ate_arr):.3f} m |\n")
            f.write(f"| Min | {np.min(ate_arr):.3f} m |\n")
            f.write(f"| Max | {np.max(ate_arr):.3f} m |\n")
            f.write(f"| 25th Percentile | {np.percentile(ate_arr, 25):.3f} m |\n")
            f.write(f"| 75th Percentile | {np.percentile(ate_arr, 75):.3f} m |\n\n")

            # 性能评估分析
            f.write("## Performance Analysis\n\n")
            f.write("### Scale Error Interpretation\n\n")
            median_scale = np.median(scale_arr)
            if median_scale < 5.0:
                f.write(f"✅ **Excellent**: Scale error median of {median_scale:.2f}% indicates very accurate ")
                f.write("scale recovery. The initialization provides a strong geometric foundation.\n\n")
            elif median_scale < 10.0:
                f.write(f"✓ **Good**: Scale error median of {median_scale:.2f}% shows reliable scale estimation ")
                f.write("with minor deviations acceptable for most applications.\n\n")
            elif median_scale < 20.0:
                f.write(f"⚠ **Moderate**: Scale error median of {median_scale:.2f}% indicates noticeable scale drift. ")
                f.write("May require additional constraints or longer initialization duration.\n\n")
            else:
                f.write(f"❌ **Poor**: Scale error median of {median_scale:.2f}% suggests significant scale ")
                f.write("estimation issues. Initialization conditions may be insufficient.\n\n")

            f.write("### Tracking Accuracy Interpretation\n\n")
            median_ate = np.median(ate_arr)
            if median_ate < 0.1:
                f.write(f"✅ **Excellent**: ATE median of {median_ate:.3f}m over 10s window demonstrates ")
                f.write("exceptional tracking accuracy with minimal drift.\n\n")
            elif median_ate < 0.5:
                f.write(f"✓ **Good**: ATE median of {median_ate:.3f}m indicates solid tracking performance ")
                f.write("suitable for precision applications.\n\n")
            elif median_ate < 1.0:
                f.write(f"⚠ **Moderate**: ATE median of {median_ate:.3f}m shows acceptable tracking but with ")
                f.write("noticeable drift accumulation over time.\n\n")
            else:
                f.write(f"❌ **Poor**: ATE median of {median_ate:.3f}m indicates significant tracking drift. ")
                f.write("System may struggle with long-term consistency.\n\n")

            # 成功率分析
            f.write("### Initialization Success Rate Analysis\n\n")
            if success_rate >= 90:
                f.write(f"✅ **Highly Robust**: {success_rate:.1f}% success rate demonstrates excellent ")
                f.write("robustness across diverse motion patterns and conditions.\n\n")
            elif success_rate >= 70:
                f.write(f"✓ **Robust**: {success_rate:.1f}% success rate shows good reliability with ")
                f.write("occasional failures in challenging scenarios.\n\n")
            elif success_rate >= 50:
                f.write(f"⚠ **Moderate Reliability**: {success_rate:.1f}% success rate indicates moderate ")
                f.write("robustness. May require motion constraints or improved conditions.\n\n")
            else:
                f.write(f"❌ **Low Reliability**: {success_rate:.1f}% success rate suggests fundamental ")
                f.write("challenges with current initialization approach or test conditions.\n\n")

        # 详细结果表格
        f.write("## Detailed Results Per Segment\n\n")
        f.write("| Time (s) | Status | Init Scale Error | Run ATE | Notes |\n")
        f.write("|----------|--------|------------------|---------|-------|\n")

        for row in detail_rows:
            status_emoji = {
                'OK': '✅',
                'Failed': '❌',
                'Static': '⏸️',
                'Diverged': '⚠️',
                'Error': '❗'
            }.get(row['status'], '')

            f.write(f"| {row['time']} | {status_emoji} {row['status']} | {row['init_scale']} | ")
            f.write(f"{row['run_ate']} | {row['note']} |\n")

        f.write("\n---\n\n")
        f.write("**Note**: For research publications, prefer reporting **Median** values as they are ")
        f.write("more robust to outliers than Mean values.\n\n")
        f.write("**Legend**:\n")
        f.write("- ✅ OK: Successfully initialized and tracked\n")
        f.write("- ❌ Failed: Initialization failed\n")
        f.write("- ⏸️ Static: Insufficient motion (excluded from statistics)\n")
        f.write("- ⚠️ Diverged: Initialized but tracking diverged (excluded from statistics)\n")
        f.write("- ❗ Error: Processing error\n")

def evaluate():
    print(f"Loading Ground Truth: {GT_FILE}")
    traj_ref_full = file_interface.read_tum_trajectory_file(GT_FILE)

    # 获取所有初始化的时间戳 ID
    all_files = glob.glob(os.path.join(RESULT_DIR, "traj_*"))
    unique_ids = set()
    for f in all_files:
        fname = os.path.basename(f)
        if "traj_" in fname:
            # 提取时间戳，兼容 _init.txt, _run.txt, _fail.mark
            parts = fname.split("_") # ['traj', '10.0', 'init.txt']
            if len(parts) >= 2:
                unique_ids.add(parts[1])

    sorted_ids = sorted(list(unique_ids), key=lambda x: float(x))

    # 统计容器
    stats_init_scale = []
    stats_run_ate = []
    detail_rows = []  # 存储详细结果（用于Markdown表格）

    count_total = 0
    count_success = 0
    count_static = 0    # 静止片段数
    count_diverged = 0  # 发散片段数

    print(f"\n{'Time':<6} | {'Status':<8} | {'Init Scale':<10} | {'Run ATE':<10} | {'Note'}")
    print("-" * 65)

    for t_id in sorted_ids:
        count_total += 1
        f_init = os.path.join(RESULT_DIR, f"traj_{t_id}_init.txt")
        f_run  = os.path.join(RESULT_DIR, f"traj_{t_id}_run.txt")

        # 1. 检查是否初始化成功 (文件存在)
        if not (os.path.exists(f_init) and os.path.exists(f_run)):
            print(f"{t_id:<6} | \033[91mFailed\033[0m   | {'-':<10} | {'-':<10} | Init Failed")
            detail_rows.append({
                'time': t_id,
                'status': 'Failed',
                'init_scale': '-',
                'run_ate': '-',
                'note': 'Init Failed'
            })
            continue

        count_success += 1

        # 2. 读取轨迹
        traj_init = file_interface.read_tum_trajectory_file(f_init)
        traj_run  = file_interface.read_tum_trajectory_file(f_run)

        # 3. 关联真值
        ref_init, est_init = sync.associate_trajectories(traj_ref_full, traj_init, max_diff=0.02)
        ref_run,  est_run  = sync.associate_trajectories(traj_ref_full, traj_run, max_diff=0.02)

        if len(est_init.timestamps) < 5:
            print(f"{t_id:<6} | Error    | Too few points")
            detail_rows.append({
                'time': t_id,
                'status': 'Error',
                'init_scale': '-',
                'run_ate': '-',
                'note': 'Too few points'
            })
            continue

        # 4. 计算指标
        # 只关心 Init 的 Scale (几何对齐能力) 和 Run 的 ATE (长期跟踪能力)
        i_scale, _, is_static = calculate_metrics(est_init, ref_init)
        _, r_ate, _           = calculate_metrics(est_run, ref_run)

        # 5. 分类统计
        if is_static:
            count_static += 1
            print(f"{t_id:<6} | \033[93mStatic\033[0m   | {'Skip':<10} | {'Skip':<10} | Motion < {MIN_GT_MOTION}m")
            detail_rows.append({
                'time': t_id,
                'status': 'Static',
                'init_scale': 'Skip',
                'run_ate': 'Skip',
                'note': f'Motion < {MIN_GT_MOTION}m'
            })
        else:
            # 如果运行段是静止的 (r_ate is None)，特殊处理
            if r_ate is None:
                stats_init_scale.append(i_scale)
                print(f"{t_id:<6} | \033[92mOK\033[0m       | {i_scale:6.2f}%    | {'-':<10} | Run segment static")
                detail_rows.append({
                    'time': t_id,
                    'status': 'OK',
                    'init_scale': f'{i_scale:.2f}%',
                    'run_ate': '-',
                    'note': 'Run segment static'
                })
            # 检查是否发散
            elif r_ate > DIVERGENCE_THRES:
                count_diverged += 1
                print(f"{t_id:<6} | \033[91mDiverged\033[0m | {i_scale:6.2f}%    | {r_ate:6.2f}m    | ATE > {DIVERGENCE_THRES}m")
                detail_rows.append({
                    'time': t_id,
                    'status': 'Diverged',
                    'init_scale': f'{i_scale:.2f}%',
                    'run_ate': f'{r_ate:.2f}m',
                    'note': f'ATE > {DIVERGENCE_THRES}m'
                })
            else:
                # 正常数据，计入统计
                stats_init_scale.append(i_scale)
                stats_run_ate.append(r_ate)
                print(f"{t_id:<6} | \033[92mOK\033[0m       | {i_scale:6.2f}%    | {r_ate:6.3f}m    |")
                detail_rows.append({
                    'time': t_id,
                    'status': 'OK',
                    'init_scale': f'{i_scale:.2f}%',
                    'run_ate': f'{r_ate:.3f}m',
                    'note': ''
                })

    # === 生成 Markdown 报告 ===
    generate_markdown_report(
        count_total, count_success, count_static, count_diverged,
        stats_init_scale, stats_run_ate, detail_rows
    )

    # === 终端最终报告 ===
    print("\n" + "="*30 + " 最终评估报告 " + "="*30)
    print(f"总测试片段: {count_total}")
    print(f"初始化成功: {count_success} (成功率: {count_success/count_total*100:.1f}%)")
    print(f"  - 静止/微动片段: {count_static} (已剔除)")
    print(f"  - 跟踪发散片段: {count_diverged} (已剔除)")
    print(f"  - 有效评估样本: {len(stats_init_scale)}")

    if len(stats_init_scale) > 0:
        print("-" * 60)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Median':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 60)

        # 尺度误差统计
        d = np.array(stats_init_scale)
        print(f"{'Init Scale Err':<15} | {np.mean(d):6.2f}%    | \033[92m{np.median(d):6.2f}%\033[0m    | {np.min(d):6.2f}%    | {np.max(d):6.2f}%")

        # 跟踪误差统计
        d = np.array(stats_run_ate)
        print(f"{'Run ATE (10s)':<15} | {np.mean(d):6.3f}m    | \033[92m{np.median(d):6.3f}m\033[0m    | {np.min(d):6.3f}m    | {np.max(d):6.3f}m")
        print("-" * 60)
        print("注意：论文汇报时优先使用 Median (中位数)，因为它能抵抗离群值影响。")
        print(f"\n✅ Markdown报告已保存至: {OUTPUT_MD}")
    else:
        print("没有有效的运动片段可供评估。")

if __name__ == "__main__":
    evaluate()



