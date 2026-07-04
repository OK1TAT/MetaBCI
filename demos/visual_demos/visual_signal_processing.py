# -*- coding: utf-8 -*-
"""
信号处理可视化 - EEG预处理管道展示
展示原始信号→滤波→频谱分析→微状态全过程
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import butter, filtfilt, welch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
fs = 250
t = np.arange(0, 4, 1/fs)

# 合成含噪声的EEG信号
clean = (np.sin(2*np.pi*10*t) * 0.5 +  # alpha
         np.sin(2*np.pi*6*t) * 0.3 +    # theta
         np.sin(2*np.pi*20*t) * 0.2)     # beta
noise_50 = np.sin(2*np.pi*50*t) * 0.8    # 工频干扰
noise_high = np.random.randn(len(t)) * 0.4  # 高频噪声
raw = clean + noise_50 + noise_high

# 带通滤波 0.5-45Hz
nyq = fs / 2
b_bp, a_bp = butter(4, [0.5/nyq, 45/nyq], btype='band')
filtered = filtfilt(b_bp, a_bp, raw)

# 陷波滤波 50Hz
b_notch, a_notch = butter(4, [48/nyq, 52/nyq], btype='bandstop')
filtered = filtfilt(b_notch, a_notch, filtered)

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3)

# ===== (1) 原始信号 vs 滤波后 =====
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, raw, color='#e74c3c', linewidth=0.6, alpha=0.7, label='原始信号(含50Hz工频+高频噪声)')
ax1.plot(t, filtered, color='#2ecc71', linewidth=1.2, label='滤波后信号(0.5-45Hz带通+50Hz陷波)')
ax1.set_xlabel('时间 (s)', fontsize=10)
ax1.set_ylabel('幅值 (μV)', fontsize=10)
ax1.set_title('EEG信号预处理: 带通滤波 + 陷波滤波', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(0, 4)
ax1.grid(True, alpha=0.3)

# ===== (2) 功率谱密度对比 =====
ax2 = fig.add_subplot(gs[1, 0])
f_raw, psd_raw = welch(raw, fs, nperseg=512)
f_filt, psd_filt = welch(filtered, fs, nperseg=512)
ax2.semilogy(f_raw, psd_raw, color='#e74c3c', linewidth=1, alpha=0.7, label='原始信号PSD')
ax2.semilogy(f_filt, psd_filt, color='#2ecc71', linewidth=1.5, label='滤波后PSD')
ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50Hz工频')
# 频率带标注
band_ranges = [(4, 8, 'θ'), (8, 10, 'αL'), (10, 13, 'αH'), (13, 20, 'βL'), (20, 30, 'βH')]
band_colors = ['#3498db', '#2ecc71', '#1abc9c', '#f39c12', '#e74c3c']
for (lo, hi, name), c in zip(band_ranges, band_colors):
    ax2.axvspan(lo, hi, alpha=0.1, color=c)
    ax2.text((lo+hi)/2, ax2.get_ylim()[1]*0.5, name, ha='center', fontsize=7, color=c, fontweight='bold')
ax2.set_xlabel('频率 (Hz)', fontsize=10)
ax2.set_ylabel('PSD (μV²/Hz)', fontsize=10)
ax2.set_title('功率谱密度对比', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_xlim(0, 60)
ax2.grid(True, alpha=0.3)

# ===== (3) 5频率带功率分布 =====
ax3 = fig.add_subplot(gs[1, 1])
band_powers = []
for lo, hi, _ in band_ranges:
    idx = (f_filt >= lo) & (f_filt < hi)
    band_powers.append(np.trapezoid(psd_filt[idx], f_filt[idx]))
bar_colors = ['#3498db', '#2ecc71', '#1abc9c', '#f39c12', '#e74c3c']
band_labels = ['θ\n(4-8Hz)', 'αL\n(8-10Hz)', 'αH\n(10-13Hz)', 'βL\n(13-20Hz)', 'βH\n(20-30Hz)']
bars = ax3.bar(band_labels, band_powers, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, band_powers):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_ylabel('绝对功率 (μV²)', fontsize=10)
ax3.set_title('5频率带功率分布', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2, axis='y')

# ===== (4) 多通道滤波后信号 =====
ax4 = fig.add_subplot(gs[2, :])
channels = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8']
colors_ch = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (ch, c) in enumerate(zip(channels, colors_ch)):
    sig = (np.sin(2*np.pi*np.random.uniform(6, 25)*t) * 0.5 + 
           0.2 * np.random.randn(len(t)))
    b, a = butter(4, [0.5/nyq, 45/nyq], btype='band')
    sig = filtfilt(b, a, sig)
    ax4.plot(t, sig + i*4, color=c, linewidth=0.8, label=ch)
ax4.set_xlabel('时间 (s)', fontsize=10)
ax4.set_ylabel('幅值 (μV)', fontsize=10)
ax4.set_title('前额六导联滤波后信号', fontsize=13, fontweight='bold')
ax4.legend(fontsize=8, ncol=6, loc='upper right')
ax4.set_xlim(0, 4)
ax4.grid(True, alpha=0.3)

# ===== (5) 在线滑动窗口示意 =====
ax5 = fig.add_subplot(gs[3, :])
sig_online = np.sin(2*np.pi*10*t[:500]) * 0.5 + 0.2 * np.random.randn(500)
ax5.plot(t[:500], sig_online, color='#3498db', linewidth=1, alpha=0.8)
# 滑动窗口
window_size = 125  # 0.5秒
step = 25  # 0.1秒
for start in range(0, 500 - window_size, step):
    end = start + window_size
    color = plt.cm.RdYlGn_r(start / 500)
    ax5.axvspan(t[start], t[end], alpha=0.15, color=color)
    if start % (step*5) == 0:
        ax5.axvline(x=t[start], color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        ax5.annotate(f'窗口{start//step + 1}', xy=(t[start], 1.8), fontsize=7, color='red')
ax5.set_xlabel('时间 (s)', fontsize=10)
ax5.set_ylabel('幅值 (μV)', fontsize=10)
ax5.set_title('在线滑动窗口特征提取 (窗口=0.5s, 步长=0.1s, 重叠率=80%)', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 2)
ax5.grid(True, alpha=0.3)

plt.suptitle('MetaBCI EEG信号处理与在线分析管道', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('visual_signal_processing.png', dpi=200, bbox_inches='tight', facecolor='white')
print('✓ 信号处理可视化已保存: visual_signal_processing.png')
plt.show()
