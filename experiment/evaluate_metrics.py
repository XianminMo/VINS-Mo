import os
import numpy as np
import glob
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

# ================= 配置区域 =================
GT_FILE = "/home/linux/mxm/data/Euroc/V1_01_easy/mav0/state_groundtruth_estimate0/data.tum" # 真值文件路径 (TUM格式)
RESULT_DIR = "/home/linux/mxm/output/experiments_initial/V101_test/window_0_5s/deep" 

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
            continue

        # 4. 计算指标
        # 只关心 Init 的 Scale (几何对齐能力) 和 Run 的 ATE (长期跟踪能力)
        i_scale, _, is_static = calculate_metrics(est_init, ref_init)
        _, r_ate, _           = calculate_metrics(est_run, ref_run)
        
        # 5. 分类统计
        if is_static:
            count_static += 1
            print(f"{t_id:<6} | \033[93mStatic\033[0m   | {'Skip':<10} | {'Skip':<10} | Motion < {MIN_GT_MOTION}m")
        else:
            # 检查是否发散
            if r_ate > DIVERGENCE_THRES:
                count_diverged += 1
                print(f"{t_id:<6} | \033[91mDiverged\033[0m | {i_scale:6.2f}%    | {r_ate:6.2f}m    | ATE > {DIVERGENCE_THRES}m")
            else:
                # 正常数据，计入统计
                stats_init_scale.append(i_scale)
                stats_run_ate.append(r_ate)
                print(f"{t_id:<6} | \033[92mOK\033[0m       | {i_scale:6.2f}%    | {r_ate:6.3f}m    |")

    # === 最终报告 ===
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
    else:
        print("没有有效的运动片段可供评估。")

if __name__ == "__main__":
    evaluate()



