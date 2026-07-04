# -*- coding: utf-8 -*-
"""
特征提取可视化 - 135维特征工程展示
展示前额六导联、5频率带、PLI/RP/FE三类特征的提取过程
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
fs = 250  # 采样率
t = np.arange(0, 2, 1/fs)  # 2秒

# ===== 合成6通道前额EEG信号 =====
channels = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8']
freqs = [6, 9, 11.5, 16, 25]  # θ, α_low, α_high, β_low, β_high
band_names = [r'$\theta$(4-8Hz)', r'$\alpha_L$(8-10Hz)', r'$\alpha_H$(10-13Hz)', r'$\beta_L$(13-20Hz)', r'$\beta_H$(20-30Hz)']

eeg = np.zeros((6, len(t)))
for ch in range(6):
    for f in freqs:
        eeg[ch] += np.sin(2*np.pi*f*t + np.random.uniform(0, 2*np.pi)) * np.random.uniform(0.3, 1.0)
    eeg[ch] += 0.1 * np.random.randn(len(t))

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# ===== (1) 原始EEG信号 =====
ax1 = fig.add_subplot(gs[0, :])
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i in range(6):
    ax1.plot(t, eeg[i] + i*8, color=colors[i], linewidth=0.8, label=channels[i])
ax1.set_ylabel('幅值 (μV)', fontsize=10)
ax1.set_xlabel('时间 (s)', fontsize=10)
ax1.set_title('前额六导联原始EEG信号 (FP1, FP2, F3, F4, F7, F8)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', ncol=6, fontsize=8)
ax1.set_xlim(0, 2)
ax1.grid(True, alpha=0.3)

# ===== (2) 频率分解 =====
ax2 = fig.add_subplot(gs[1, 0])
for i, (bn, c) in enumerate(zip(band_names, colors)):
    # 模拟带通滤波后的信号
    from scipy.signal import butter, filtfilt
    low = [4, 8, 10, 13, 20][i]
    high = [8, 10, 13, 20, 30][i]
    nyq = fs / 2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    filtered = filtfilt(b, a, eeg[0])
    ax2.plot(t, filtered + i*3, color=c, linewidth=0.9, label=bn)
ax2.set_ylabel('幅值', fontsize=9)
ax2.set_xlabel('时间 (s)', fontsize=9)
ax2.set_title('5频率带分解 (FP1)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=7, loc='upper right')
ax2.set_xlim(0, 2)
ax2.grid(True, alpha=0.3)

# ===== (3) PLI连接矩阵 =====
ax3 = fig.add_subplot(gs[1, 1])
# 模拟6×6 PLI矩阵（15个唯一对）
pli_matrix = np.abs(np.random.randn(6, 6)) * 0.3 + 0.2
pli_matrix = (pli_matrix + pli_matrix.T) / 2
np.fill_diagonal(pli_matrix, 0)
im = ax3.imshow(pli_matrix, cmap='YlOrRd', vmin=0, vmax=0.8, aspect='equal')
ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels(channels, fontsize=8)
ax3.set_yticklabels(channels, fontsize=8)
ax3.set_title('PLI相位连接矩阵 (单频带)', fontsize=11, fontweight='bold')
for i in range(6):
    for j in range(6):
        ax3.text(j, i, f'{pli_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if pli_matrix[i,j] > 0.4 else 'black')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='PLI值')
ax3.text(0.5, -0.15, '15导联对 × 5频带 = 75维', transform=ax3.transAxes,
         ha='center', fontsize=8, color='#e74c3c', fontweight='bold')

# ===== (4) RP相对功率 =====
ax4 = fig.add_subplot(gs[1, 2])
rp_data = np.random.uniform(0.1, 0.9, (6, 5))
rp_data = rp_data / rp_data.sum(axis=1, keepdims=True)  # 归一化
x = np.arange(6)
width = 0.15
for i, (bn, c) in enumerate(zip(band_names, colors)):
    ax4.bar(x + i*width, rp_data[:, i], width, color=c, label=bn, alpha=0.85)
ax4.set_xticks(x + width*2)
ax4.set_xticklabels(channels, fontsize=8)
ax4.set_ylabel('相对功率', fontsize=9)
ax4.set_title('RP相对功率分布', fontsize=11, fontweight='bold')
ax4.legend(fontsize=6, ncol=2, loc='upper right')
ax4.text(0.5, -0.15, '6导联 × 5频带 = 30维', transform=ax4.transAxes,
         ha='center', fontsize=8, color='#e74c3c', fontweight='bold')

# ===== (5) 135维特征总览热力图 =====
ax5 = fig.add_subplot(gs[2, :])
features_135 = np.random.randn(10, 135) * np.random.uniform(0.5, 2.0, 135)
# 分三段着色
im2 = ax5.imshow(features_135, cmap='RdYlBu_r', aspect='auto', vmin=-3, vmax=3)
ax5.set_ylabel('样本', fontsize=9)
ax5.set_xlabel('特征维度', fontsize=9)
ax5.set_title('135维特征矩阵总览 (10个样本示例)', fontsize=12, fontweight='bold')

# 分段标注
ax5.axvline(x=74.5, color='white', linewidth=2, linestyle='--')
ax5.axvline(x=104.5, color='white', linewidth=2, linestyle='--')
ax5.text(37, -1.2, 'PLI (75维)', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')
ax5.text(89.5, -1.2, 'RP (30维)', ha='center', fontsize=10, fontweight='bold', color='#2ecc71')
ax5.text(119.5, -1.2, 'FE (30维)', ha='center', fontsize=10, fontweight='bold', color='#9b59b6')
plt.colorbar(im2, ax=ax5, fraction=0.02, pad=0.02, label='标准化值')

plt.suptitle('MetaBCI 前额六导联135维特征提取引擎', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('visual_features.png', dpi=200, bbox_inches='tight', facecolor='white')
print('✓ 特征提取可视化已保存: visual_features.png')
plt.show()
