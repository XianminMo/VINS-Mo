import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
path = "/home/linux/mxm/output/experiments_backend/corridor1_1/1219/o-d/k_5/0.5_speed/"
df = pd.read_csv(path + 'para_a_global.csv')

# 创建图表：3行2列布局
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ========================================
# 左上: 所有帧的 para_a (scale) 值
# ========================================
ax1 = fig.add_subplot(gs[0, 0])

# 绘制每帧的尺度参数曲线
for i in range(21):  # WINDOW_SIZE + 1 = 21
    col_name = f'frame_{i}_para_a'
    if col_name in df.columns:
        ax1.plot(df['global_frame_id'], df[col_name],
                alpha=0.3, linewidth=0.5, label=f'Frame {i}' if i % 5 == 0 else '')

# 绘制平均值
ax1.plot(df['global_frame_id'], df['avg_para_a'],
        'k-', linewidth=2.5, label='Average a', zorder=10)

# 添加参考线和稳定区域
ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5,
            label='Expected (1.0)', zorder=5)
ax1.axhspan(0.85, 1.15, alpha=0.1, color='green',
            label='Stable range [0.85, 1.15]')

ax1.set_ylabel('Scale Parameter (a)', fontsize=11)
ax1.set_title('Scale Parameter (a) Evolution - Per Frame', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.8, 1.2])

# ========================================
# 右上: 所有帧的 para_b (shift) 值
# ========================================
ax2 = fig.add_subplot(gs[0, 1])

# 绘制每帧的偏移参数曲线
for i in range(21):
    col_name = f'frame_{i}_para_b'
    if col_name in df.columns:
        ax2.plot(df['global_frame_id'], df[col_name],
                alpha=0.3, linewidth=0.5, label=f'Frame {i}' if i % 5 == 0 else '')

# 绘制平均值
ax2.plot(df['global_frame_id'], df['avg_para_b'],
        'k-', linewidth=2.5, label='Average b', zorder=10)

# 添加参考线和稳定区域
ax2.axhline(y=0.0, color='r', linestyle='--', linewidth=1.5,
            label='Expected (0.0)', zorder=5)
ax2.axhspan(-0.01, 0.01, alpha=0.1, color='green',
            label='Stable range [-0.01, 0.01]')

ax2.set_ylabel('Shift Parameter (b)', fontsize=11)
ax2.set_title('Shift Parameter (b) Evolution - Per Frame', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, ncol=3)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.02, 0.02])

# ========================================
# 左中: 尺度参数统计（平均值和波动）
# ========================================
ax3 = fig.add_subplot(gs[1, 0])

# 绘制平均值和标准差
ax3.plot(df['global_frame_id'], df['avg_para_a'],
        'b-', linewidth=2, label='Average a')

# 绘制范围
ax3.fill_between(df['global_frame_id'],
                df['min_para_a'],
                df['max_para_a'],
                alpha=0.2, color='blue', label='Min-Max Range')

# 绘制 spread
ax3.plot(df['global_frame_id'], df['spread_a'],
        'c--', linewidth=1, alpha=0.7, label='Spread (max-min)')

# 参考线
ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=1,
            label='Expected (1.0)')
ax3.axhline(y=0.85, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax3.axhline(y=1.15, color='orange', linestyle=':', linewidth=1, alpha=0.5)

ax3.set_ylabel('Scale Parameter (a)', fontsize=11)
ax3.set_title('Scale Parameter (a) Statistics', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.8, 1.2])

# ========================================
# 右中: 偏移参数统计
# ========================================
ax4 = fig.add_subplot(gs[1, 1])

# 绘制平均值
ax4.plot(df['global_frame_id'], df['avg_para_b'],
        'b-', linewidth=2, label='Average b')

# 绘制范围
ax4.fill_between(df['global_frame_id'],
                df['min_para_b'],
                df['max_para_b'],
                alpha=0.2, color='blue', label='Min-Max Range')

# 绘制 spread
ax4.plot(df['global_frame_id'], df['spread_b'],
        'c--', linewidth=1, alpha=0.7, label='Spread (max-min)')

# 参考线
ax4.axhline(y=0.0, color='r', linestyle='--', linewidth=1,
            label='Expected (0.0)')
ax4.axhline(y=-0.01, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax4.axhline(y=0.01, color='orange', linestyle=':', linewidth=1, alpha=0.5)

ax4.set_ylabel('Shift Parameter (b)', fontsize=11)
ax4.set_title('Shift Parameter (b) Statistics', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-0.02, 0.02])

# ========================================
# 底部: 双参数对比（归一化显示）
# ========================================
ax5 = fig.add_subplot(gs[2, :])

# 归一化处理以便在同一图中比较
a_normalized = (df['avg_para_a'] - 1.0) * 100  # 转换为百分比偏差
b_normalized = df['avg_para_b'] * 1000  # 转换为毫米级

ax5.plot(df['global_frame_id'], a_normalized,
        'b-', linewidth=2, label='Scale (a): Deviation from 1.0 [%]', alpha=0.8)
ax5.plot(df['global_frame_id'], b_normalized,
        'r-', linewidth=2, label='Shift (b): Value × 1000', alpha=0.8)

# 零线
ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

# 稳定区域
ax5.axhspan(-5, 5, alpha=0.05, color='green', label='±5% stable zone')

ax5.set_xlabel('Global Frame ID', fontsize=11)
ax5.set_ylabel('Normalized Values', fontsize=11)
ax5.set_title('Dual-Parameter Evolution Comparison (Normalized)', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=10)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path + 'dual_param_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved figure: {path}dual_param_analysis.png")

# ========================================
# 打印统计摘要
# ========================================
print("\n" + "="*60)
print("DUAL-PARAMETER DEPTH FACTOR STATISTICS")
print("="*60)

print("\n📊 SCALE PARAMETER (a) - Expected: 1.0")
print("-" * 60)
print(f"  Total frames:        {len(df)}")
print(f"  Average (overall):   {df['avg_para_a'].mean():.6f}")
print(f"  Std deviation:       {df['avg_para_a'].std():.6f}")
print(f"  Min value:           {df['min_para_a'].min():.6f}")
print(f"  Max value:           {df['max_para_a'].max():.6f}")
print(f"  Average spread:      {df['spread_a'].mean():.6f}")
print(f"  Max spread:          {df['spread_a'].max():.6f}")
print(f"  Deviation from 1.0:  {(df['avg_para_a'].mean() - 1.0)*100:+.2f}%")

print("\n📊 SHIFT PARAMETER (b) - Expected: 0.0")
print("-" * 60)
print(f"  Total frames:        {len(df)}")
print(f"  Average (overall):   {df['avg_para_b'].mean():.8f}")
print(f"  Std deviation:       {df['avg_para_b'].std():.8f}")
print(f"  Min value:           {df['min_para_b'].min():.8f}")
print(f"  Max value:           {df['max_para_b'].max():.8f}")
print(f"  Average spread:      {df['spread_b'].mean():.8f}")
print(f"  Max spread:          {df['spread_b'].max():.8f}")

# 健康评估
print("\n🏥 SYSTEM HEALTH ASSESSMENT")
print("-" * 60)

# Scale (a) 健康评估
stable_a_frames = ((df['avg_para_a'] > 0.85) & (df['avg_para_a'] < 1.15) &
                   (df['spread_a'] < 0.2)).sum()
print(f"  Scale (a) stable:    {stable_a_frames}/{len(df)} ({100*stable_a_frames/len(df):.1f}%)")

drift_a_frames = ((df['avg_para_a'] < 0.8) | (df['avg_para_a'] > 1.2)).sum()
print(f"  Scale (a) drift:     {drift_a_frames}/{len(df)} ({100*drift_a_frames/len(df):.1f}%)")

# Shift (b) 健康评估
stable_b_frames = ((df['avg_para_b'] > -0.01) & (df['avg_para_b'] < 0.01) &
                   (df['spread_b'] < 0.01)).sum()
print(f"  Shift (b) stable:    {stable_b_frames}/{len(df)} ({100*stable_b_frames/len(df):.1f}%)")

drift_b_frames = ((df['avg_para_b'] < -0.05) | (df['avg_para_b'] > 0.05)).sum()
print(f"  Shift (b) drift:     {drift_b_frames}/{len(df)} ({100*drift_b_frames/len(df):.1f}%)")

# 总体评估
both_stable = ((df['avg_para_a'] > 0.85) & (df['avg_para_a'] < 1.15) &
               (df['avg_para_b'] > -0.01) & (df['avg_para_b'] < 0.01)).sum()
print(f"\n  ✅ Both stable:      {both_stable}/{len(df)} ({100*both_stable/len(df):.1f}%)")

print("\n" + "="*60)

# 检测异常
print("\n⚠️  ANOMALY DETECTION")
print("-" * 60)

# Scale 异常
large_a_changes = (df['spread_a'] > 0.2).sum()
if large_a_changes > 0:
    print(f"  ⚠️  Large scale jumps: {large_a_changes} frames")
else:
    print(f"  ✅ No large scale jumps detected")

# Shift 异常
large_b_changes = (df['spread_b'] > 0.02).sum()
if large_b_changes > 0:
    print(f"  ⚠️  Large shift jumps: {large_b_changes} frames")
else:
    print(f"  ✅ No large shift jumps detected")

# Shift 是否真正 "lazy"
b_very_small = (abs(df['avg_para_b']) < 0.001).sum()
print(f"\n  Shift near-zero:     {b_very_small}/{len(df)} ({100*b_very_small/len(df):.1f}%)")
if 100*b_very_small/len(df) > 80:
    print(f"  ✅ Shift parameter is properly 'lazy' (stays near 0)")
else:
    print(f"  ⚠️  Shift parameter may be too active")

print("\n" + "="*60)
print("Analysis complete! Check 'dual_param_analysis.png' for visualizations.")
print("="*60 + "\n")

# 显示图表
plt.show()
