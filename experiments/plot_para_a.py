import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
path = "/home/linux/mxm/output/experiments_backend/V201_test/1216/o-d/k_10/0/"
df = pd.read_csv(path + 'para_a_global.csv')

# 创建图表2
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ========================================
# 子图1: 所有帧的 para_a 值
# ========================================
ax1 = axes[0]

# 绘制每帧的曲线（透明度降低）
for i in range(11):
    col_name = f'frame_{i}_para_a'
    if col_name in df.columns:
        ax1.plot(df['global_frame_id'], df[col_name],
                alpha=0.3, linewidth=0.5, label=f'Frame {i}')

# 绘制平均值（加粗）
ax1.plot(df['global_frame_id'], df['avg_para_a'],
        'k-', linewidth=2, label='Average', zorder=10)

# 添加参考线（期望值 1.0）
ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1,
            label='Expected (1.0)', zorder=5)

# 填充稳定范围 [0.8, 1.2]
ax1.axhspan(0.8, 1.2, alpha=0.1, color='green',
            label='Stable range [0.8, 1.2]')

ax1.set_ylabel('para_a_global value')
ax1.set_title('Per-frame para_a_global Evolution')
ax1.legend(loc='upper right', fontsize=8, ncol=4)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.5])  # 调整 Y 轴范围

# ========================================
# 子图2: 统计指标（平均值和波动范围）
# ========================================
ax2 = axes[1]

# 绘制平均值
ax2.plot(df['global_frame_id'], df['avg_para_a'],
        'b-', linewidth=2, label='Average para_a')

# 绘制波动范围（spread）
ax2.fill_between(df['global_frame_id'],
                df['avg_para_a'] - df['spread']/2,
                df['avg_para_a'] + df['spread']/2,
                alpha=0.3, color='blue', label='Spread')

# 添加参考线
ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1,
            label='Expected (1.0)')

# 添加稳定阈值线
ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax2.axhline(y=1.2, color='orange', linestyle=':', linewidth=1, alpha=0.5)

ax2.set_xlabel('Global Frame ID')
ax2.set_ylabel('para_a_global statistics')
ax2.set_title('Average para_a_global and Spread')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.5])

plt.tight_layout()
plt.savefig(path + 'para_a_global.png', dpi=300)
plt.show()

# ========================================
# 打印统计摘要
# ========================================
print("\n=== para_a_global Statistics ===")
print(f"Total frames: {len(df)}")
print(f"Average para_a (overall): {df['avg_para_a'].mean():.4f}")
print(f"Std deviation:  {df['avg_para_a'].std():.4f}")
print(f"Min value:      {df['min_para_a'].min():.4f}")
print(f"Max value:      {df['max_para_a'].max():.4f}")
print(f"Average spread: {df['spread'].mean():.4f}")
print(f"Max spread:     {df['spread'].max():.4f}")

# 健康评估
stable_frames = ((df['avg_para_a'] > 0.8) & (df['avg_para_a'] < 1.2) & (df['spread'] < 0.5)).sum()
print(f"\nStable frames: {stable_frames}/{len(df)} ({100*stable_frames/len(df):.1f}%)")

drift_frames = ((df['avg_para_a'] < 0.5) | (df['avg_para_a'] > 1.5)).sum()
print(f"Drift frames:  {drift_frames}/{len(df)} ({100*drift_frames/len(df):.1f}%)")