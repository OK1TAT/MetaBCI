# -*- coding: utf-8 -*-
"""
系统架构可视化 - MetaBCI抑郁症认知障碍评估系统
生成系统总体架构图，展示数据流和模块关系
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 颜色方案
C_DEVICE = '#3498db'
C_FLOW = '#2ecc71'
C_FEATURE = '#e74c3c'
C_MODEL = '#9b59b6'
C_GUI = '#f39c12'
C_STIM = '#1abc9c'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->', mutation_scale=20,
                             color=color, linewidth=2)
    ax.add_patch(arrow)

# ===== 标题 =====
ax.text(8, 9.5, 'MetaBCI 抑郁症认知障碍评估系统架构', 
        ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

# ===== 第一层：数据采集 =====
ax.text(0.3, 8.3, '数据采集层', fontsize=11, fontweight='bold', color='#555')
draw_box(ax, 1, 7.2, 2.8, 0.9, 'WiFi Shield\n16通道EEG设备', C_DEVICE, 9)
draw_box(ax, 4.5, 7.2, 2.8, 0.9, 'LSL实时\n数据流适配器', C_DEVICE, 9)
draw_box(ax, 8, 7.2, 2.8, 0.9, 'EDF文件\n离线数据加载', C_DEVICE, 9)
draw_box(ax, 11.5, 7.2, 2.8, 0.9, '静息态范式\nBrainstim刺激', C_STIM, 9)

# ===== 第二层：预处理 =====
ax.text(0.3, 6.3, '预处理层', fontsize=11, fontweight='bold', color='#555')
draw_box(ax, 2, 5.2, 3, 0.9, '带通滤波 0.5-45Hz\n陷波滤波 50Hz', C_FLOW, 9)
draw_box(ax, 6, 5.2, 3, 0.9, '通道名智能清洗\n前额六导联选取', C_FLOW, 9)
draw_box(ax, 10, 5.2, 3, 0.9, '重采样 250Hz\n坏段剔除', C_FLOW, 9)

# ===== 第三层：特征提取 =====
ax.text(0.3, 4.3, '特征提取层 (135维)', fontsize=11, fontweight='bold', color='#555')
draw_box(ax, 1.5, 3.2, 3.5, 0.9, 'PLI相位锁定指数\n5频带×15导联对 = 75维', C_FEATURE, 9)
draw_box(ax, 5.8, 3.2, 3.5, 0.9, 'RP相对功率\n5频带×6导联 = 30维', C_FEATURE, 9)
draw_box(ax, 10.1, 3.2, 3.5, 0.9, 'FE模糊熵特征\n5频带×6导联 = 30维', C_FEATURE, 9)

# ===== 第四层：分类模型 =====
ax.text(0.3, 2.3, '分类模型层', fontsize=11, fontweight='bold', color='#555')
draw_box(ax, 3, 1.2, 3.5, 0.9, 'TabPFN三分类\nHC / D-CI / D-NCI', C_MODEL, 9)
draw_box(ax, 7.5, 1.2, 3.5, 0.9, 'GroupKFold\n5折交叉验证', C_MODEL, 9)
draw_box(ax, 12, 1.2, 2.3, 0.9, '模型蒸馏\n导出JSON', C_MODEL, 8)

# ===== 第五层：展示层 =====
ax.text(0.3, 0.3, '展示层', fontsize=11, fontweight='bold', color='#555')
draw_box(ax, 4, -0.5, 4, 0.7, 'Java Swing EEG实时监控GUI', C_GUI, 9)
draw_box(ax, 9, -0.5, 4, 0.7, '在线滑动窗口特征管道', C_GUI, 9)

# ===== 箭头连接 =====
# 设备层 → 预处理层
draw_arrow(ax, 2.4, 7.2, 3.5, 6.1)
draw_arrow(ax, 5.9, 7.2, 7.5, 6.1)
draw_arrow(ax, 9.4, 7.2, 11.5, 6.1)
draw_arrow(ax, 12.9, 7.2, 11.5, 6.1)

# 预处理层 → 特征提取层
draw_arrow(ax, 3.5, 5.2, 3.2, 4.1)
draw_arrow(ax, 7.5, 5.2, 7.5, 4.1)
draw_arrow(ax, 11.5, 5.2, 11.8, 4.1)

# 特征提取层 → 分类模型层
draw_arrow(ax, 3.2, 3.2, 4.7, 2.1)
draw_arrow(ax, 7.5, 3.2, 6.5, 2.1)
draw_arrow(ax, 11.8, 3.2, 9.2, 2.1)

# 分类模型层 → 展示层
draw_arrow(ax, 4.7, 1.2, 6, 0.2)
draw_arrow(ax, 9.2, 1.2, 10, 0.2)
draw_arrow(ax, 13.1, 1.2, 11, 0.2)

# MetaBCI框架标注
ax.text(15.5, 5.2, 'MetaBCI\n框架', ha='center', va='center',
        fontsize=9, color='#bbbdc0', style='italic',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

plt.tight_layout()
plt.savefig('visual_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
print('✓ 系统架构图已保存: visual_architecture.png')
plt.show()
