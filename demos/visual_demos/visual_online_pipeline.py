# -*- coding: utf-8 -*-
"""
在线处理管道可视化 - 实时数据流与环形缓冲区
展示滑动窗口、环形缓冲区、实时特征计算、分类结果输出
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ===== (1) 环形缓冲区可视化 =====
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
buffer_size = 32
write_pos = 22
read_pos = 8
# 画环形缓冲区
theta = np.linspace(0, 2*np.pi, buffer_size, endpoint=False)
# 已写入数据
for i in range(buffer_size):
    angle = theta[i]
    if i < write_pos:
        # 已写入
        r_outer = 1.0
        color = '#2ecc71'
    else:
        # 空闲
        r_outer = 0.9
        color = '#ecf0f1'
    ax1.bar(angle, r_outer, width=2*np.pi/buffer_size*0.9, bottom=0.3,
            color=color, edgecolor='white', linewidth=0.5, alpha=0.8)

# 写指针
ax1.annotate('', xy=(theta[write_pos], 1.4), xytext=(theta[write_pos], 1.6),
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
ax1.text(theta[write_pos], 1.75, '写指针', fontsize=9, color='#e74c3c', fontweight='bold',
         ha='center', va='center')

# 读指针
ax1.annotate('', xy=(theta[read_pos], 1.4), xytext=(theta[read_pos], 1.6),
             arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
ax1.text(theta[read_pos], 1.75, '读指针', fontsize=9, color='#3498db', fontweight='bold',
         ha='center', va='center')

# 中心文字
ax1.text(0, 0, 'Ring\nBuffer\n32 slots', ha='center', va='center',
         fontsize=12, fontweight='bold', color='#2c3e50')
ax1.set_title('高效环形缓冲区 (Ring Buffer)', fontsize=13, fontweight='bold', pad=20)
ax1.set_ylim(0, 2)
ax1.set_yticks([])
ax1.set_xticks([])

# ===== (2) 实时数据流时间轴 =====
ax2 = fig.add_subplot(gs[0, 1])
fs = 500
t = np.arange(0, 3, 1/fs)
# 模拟实时EEG流
signal = np.sin(2*np.pi*10*t) * 0.5 + 0.2 * np.random.randn(len(t))

# 滑动窗口
window_samples = 250  # 0.5s
step_samples = 50     # 0.1s

ax2.plot(t, signal, color='#3498db', linewidth=0.5, alpha=0.6)
# 画几个窗口
for i, start in enumerate(range(0, len(t) - window_samples, step_samples)):
    if i > 20:
        break
    end = start + window_samples
    alpha = 0.2 if i % 3 == 0 else 0.08
    color = '#e74c3c' if i % 3 == 0 else '#2ecc71'
    ax2.axvspan(t[start], t[end], alpha=alpha, color=color)

# 标注特征提取点
feat_times = t[range(0, len(t) - window_samples, step_samples * 3)][:8]
feat_vals = np.random.uniform(-0.8, 0.8, len(feat_times))
ax2.scatter(feat_times, feat_vals, color='#e74c3c', s=50, zorder=5, marker='D',
           label='特征提取点')

ax2.set_xlabel('时间 (s)', fontsize=10)
ax2.set_ylabel('幅值 (μV)', fontsize=10)
ax2.set_title('在线滑动窗口实时特征提取', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 3)
ax2.grid(True, alpha=0.3)

# ===== (3) 实时分类输出时间线 =====
ax3 = fig.add_subplot(gs[1, :])
np.random.seed(42)
n_windows = 30
window_times = np.arange(n_windows) * 0.1  # 每0.1s一个窗口

# 模拟实时分类结果
classes = ['HC', 'D-CI', 'D-NCI']
class_colors = {'HC': '#2ecc71', 'D-CI': '#e74c3c', 'D-NCI': '#f39c12'}

# 前段HC，中段D-CI，后段D-NCI
preds = ['HC']*10 + ['D-CI']*8 + ['D-NCI']*12
# 加点噪声
for i in range(n_windows):
    if np.random.random() < 0.15:
        preds[i] = np.random.choice(classes)

# 画分类结果时间线
for i, pred in enumerate(preds):
    ax3.barh(0, 0.1, left=window_times[i], height=0.6, 
             color=class_colors[pred], alpha=0.85, edgecolor='white', linewidth=0.5)

# 置信度曲线
confidence = np.random.uniform(0.6, 0.95, n_windows)
ax3_twin = ax3.twinx()
ax3_twin.plot(window_times + 0.05, confidence, 'ko-', linewidth=2, markersize=4, label='分类置信度')
ax3_twin.set_ylabel('置信度', fontsize=10, color='black')
ax3_twin.set_ylim(0, 1.1)

# 图例
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in class_colors.items()]
legend_patches.append(plt.Line2D([0], [0], color='black', marker='o', label='分类置信度'))
ax3.legend(handles=legend_patches, fontsize=9, loc='upper right', ncol=4)

ax3.set_xlabel('时间 (s)', fontsize=10)
ax3.set_ylabel('分类结果', fontsize=10)
ax3.set_title('在线实时分类输出时间线 (TabPFN三分类)', fontsize=13, fontweight='bold')
ax3.set_xlim(-0.05, window_times[-1] + 0.15)
ax3.set_ylim(-0.5, 0.5)
ax3.set_yticks([])
ax3.grid(True, alpha=0.2, axis='x')

# 标注分类切换点
ax3.axvline(x=window_times[10], color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=window_times[18], color='gray', linestyle='--', alpha=0.5)
ax3.text(window_times[5], 0.35, 'HC阶段', ha='center', fontsize=10, fontweight='bold', color='#2ecc71')
ax3.text(window_times[14], 0.35, 'D-CI阶段', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')
ax3.text(window_times[24], 0.35, 'D-NCI阶段', ha='center', fontsize=10, fontweight='bold', color='#f39c12')

plt.suptitle('MetaBCI 在线处理管道: 环形缓冲区 → 滑动窗口 → 实时分类', 
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('visual_online_pipeline.png', dpi=200, bbox_inches='tight', facecolor='white')
print('✓ 在线管道可视化已保存: visual_online_pipeline.png')
plt.show()
