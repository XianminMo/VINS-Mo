#!/usr/bin/env python3
"""
深度融合指标可视化脚本

读取 depth_fusion_metrics.csv 文件，绘制以下趋势图：
1. 权重(weight)随时间变化
2. IMU数据(gyro_norm, acc_disturbance)随时间变化
3. 不稳定性评分(raw_score, smoothed_score)随时间变化
4. Huber阈值随时间变化
5. 深度参数(scale_a, shift_b)随时间变化

使用方法:
    python3 plot_depth_fusion_metrics.py <csv_file_path> [--output <output_dir>]

示例:
    python3 plot_depth_fusion_metrics.py /path/to/depth_fusion_metrics.csv
    python3 plot_depth_fusion_metrics.py /path/to/depth_fusion_metrics.csv --output ./figures
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_depth_fusion_metrics(csv_path, output_dir=None):
    """
    绘制深度融合指标图表

    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录（可选）
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取数据: {len(df)} 行")
        print(f"列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 检查必需的列
    required_cols = ['frame_id', 'weight', 'scale_a', 'shift_b']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必需的列: {missing_cols}")
        return

    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(csv_path).parent

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取权重模式
    weight_mode = df['weight_mode'].iloc[0] if 'weight_mode' in df.columns else 1
    mode_str = "自适应权重" if weight_mode == 1 else "固定权重"

    # ========================================================================
    # 图1: 权重和Huber阈值随时间变化
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 权重
    ax1.plot(df['frame_id'], df['weight'], 'b-', linewidth=2, label='Weight')
    ax1.set_xlabel('Frame ID', fontsize=12)
    ax1.set_ylabel('Weight', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Depth Fusion Weight ({mode_str})', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')

    # Huber阈值
    ax2.plot(df['frame_id'], df['huber_threshold'], 'r-', linewidth=2, label='Huber Threshold')
    ax2.set_xlabel('Frame ID', fontsize=12)
    ax2.set_ylabel('Huber Threshold', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Huber Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig_path = output_path / 'depth_fusion_weight_huber.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {fig_path}")
    plt.close()

    # ========================================================================
    # 图2: IMU数据（仅自适应模式）
    # ========================================================================
    if weight_mode == 1 and 'gyro_norm' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 陀螺仪范数
        ax1.plot(df['frame_id'], df['gyro_norm'], 'g-', linewidth=1.5, label='Gyro Norm')
        ax1.set_xlabel('Frame ID', fontsize=12)
        ax1.set_ylabel('Gyro Norm (rad/s)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('IMU Gyroscope Norm', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')

        # 加速度扰动
        ax2.plot(df['frame_id'], df['acc_disturbance'], 'm-', linewidth=1.5, label='Acc Disturbance')
        ax2.set_xlabel('Frame ID', fontsize=12)
        ax2.set_ylabel('Acc Disturbance (m/s²)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('IMU Acceleration Disturbance', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        fig_path = output_path / 'depth_fusion_imu.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {fig_path}")
        plt.close()

    # ========================================================================
    # 图3: 不稳定性评分（仅自适应模式）
    # ========================================================================
    if weight_mode == 1 and 'raw_score' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['frame_id'], df['raw_score'], 'c-', linewidth=1, alpha=0.5, label='Raw Score')
        ax.plot(df['frame_id'], df['smoothed_score'], 'b-', linewidth=2, label='Smoothed Score')
        ax.set_xlabel('Frame ID', fontsize=12)
        ax.set_ylabel('Instability Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title('Motion Instability Score (Raw vs Smoothed)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()
        fig_path = output_path / 'depth_fusion_instability_score.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {fig_path}")
        plt.close()

    # ========================================================================
    # 图4: 深度参数(a, b)随时间变化
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 尺度参数 a
    ax1.plot(df['frame_id'], df['scale_a'], 'orange', linewidth=2, label='Scale a')
    ax1.set_xlabel('Frame ID', fontsize=12)
    ax1.set_ylabel('Scale a', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Depth Scale Parameter (a)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')

    # 偏移参数 b
    ax2.plot(df['frame_id'], df['shift_b'], 'purple', linewidth=2, label='Shift b')
    ax2.set_xlabel('Frame ID', fontsize=12)
    ax2.set_ylabel('Shift b', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Depth Shift Parameter (b)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig_path = output_path / 'depth_fusion_parameters.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {fig_path}")
    plt.close()

    # ========================================================================
    # 图5: 综合视图（权重、评分、参数）
    # ========================================================================
    if weight_mode == 1 and 'smoothed_score' in df.columns:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 子图1: 权重
        axes[0].plot(df['frame_id'], df['weight'], 'b-', linewidth=2)
        axes[0].set_ylabel('Weight', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Comprehensive View: Weight, Instability Score, and Depth Parameters',
                         fontsize=14, fontweight='bold')

        # 子图2: 平滑后的不稳定性评分
        axes[1].plot(df['frame_id'], df['smoothed_score'], 'r-', linewidth=2)
        axes[1].set_ylabel('Smoothed Score', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 子图3: 深度参数
        ax3_twin = axes[2].twinx()
        line1 = axes[2].plot(df['frame_id'], df['scale_a'], 'orange', linewidth=2, label='Scale a')
        line2 = ax3_twin.plot(df['frame_id'], df['shift_b'], 'purple', linewidth=2, label='Shift b')
        axes[2].set_xlabel('Frame ID', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Scale a', fontsize=11, fontweight='bold', color='orange')
        ax3_twin.set_ylabel('Shift b', fontsize=11, fontweight='bold', color='purple')
        axes[2].tick_params(axis='y', labelcolor='orange')
        ax3_twin.tick_params(axis='y', labelcolor='purple')
        axes[2].grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc='upper right')

        plt.tight_layout()
        fig_path = output_path / 'depth_fusion_comprehensive.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {fig_path}")
        plt.close()

    # ========================================================================
    # 打印统计信息
    # ========================================================================
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"总帧数: {len(df)}")
    print(f"权重模式: {mode_str}")
    print(f"\n权重统计:")
    print(f"  平均值: {df['weight'].mean():.4f}")
    print(f"  最小值: {df['weight'].min():.4f}")
    print(f"  最大值: {df['weight'].max():.4f}")
    print(f"  标准差: {df['weight'].std():.4f}")

    print(f"\n深度参数统计:")
    print(f"  Scale a - 平均: {df['scale_a'].mean():.6f}, 标准差: {df['scale_a'].std():.6f}")
    print(f"  Shift b - 平均: {df['shift_b'].mean():.6f}, 标准差: {df['shift_b'].std():.6f}")

    if weight_mode == 1 and 'smoothed_score' in df.columns:
        print(f"\n不稳定性评分统计:")
        print(f"  平均值: {df['smoothed_score'].mean():.4f}")
        print(f"  最小值: {df['smoothed_score'].min():.4f}")
        print(f"  最大值: {df['smoothed_score'].max():.4f}")

    print("="*60)
    print(f"\n所有图表已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='深度融合指标可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_file', type=str, help='CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出目录（默认为CSV文件所在目录）')

    args = parser.parse_args()

    # 检查文件是否存在
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"错误: 文件不存在: {csv_path}")
        return

    # 绘制图表
    plot_depth_fusion_metrics(str(csv_path), args.output)


if __name__ == '__main__':
    main()
