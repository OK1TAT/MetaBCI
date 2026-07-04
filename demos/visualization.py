#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五维认知特征雷达图
维度: theta-PLI, alpha-PLI, theta-RP, alpha-RP, FE
分组: HC(深蓝) / D-NCI(中蓝) / D-CI(浅蓝)
用法: python radar_chart.py features.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_features(df):
    """计算五维特征各标签均值，并做 min-max 归一化到 [0, 1]"""
    # 确定列
    pli_theta_cols = [c for c in df.columns if c.startswith('PLI_theta_')]
    pli_alpha_lo_cols = [c for c in df.columns if c.startswith('PLI_alpha_low_')]
    pli_alpha_hi_cols = [c for c in df.columns if c.startswith('PLI_alpha_high_')]
    pli_alpha_cols = pli_alpha_lo_cols + pli_alpha_hi_cols

    rp_theta_cols = [c for c in df.columns if c.startswith('RP_theta_')]
    rp_alpha_lo_cols = [c for c in df.columns if c.startswith('RP_alpha_low_')]
    rp_alpha_hi_cols = [c for c in df.columns if c.startswith('RP_alpha_high_')]
    rp_alpha_cols = rp_alpha_lo_cols + rp_alpha_hi_cols

    fe_cols = [c for c in df.columns if c.startswith('FE_')]

    dims = {
        'theta-PLI': pli_theta_cols,
        'alpha-PLI': pli_alpha_cols,
        'theta-RP':  rp_theta_cols,
        'alpha-RP':  rp_alpha_cols,
        'FE':        fe_cols,
    }

    # 检查列是否存在
    for name, cols in dims.items():
        if not cols:
            print(f"⚠️ 未找到 {name} 对应的列，请检查特征文件")
            return None, None

    # 计算各标签均值
    labels = {0: 'HC', 1: 'D-NCI', 2: 'D-CI'}
    raw_values = {}
    for lbl, name in labels.items():
        sub = df[df['label'] == lbl]
        raw_values[name] = [sub[cols].mean().mean() for cols in dims.values()]

    # Min-max 归一化（按维度）
    raw_matrix = np.array(list(raw_values.values()))  # 3×5
    mins = raw_matrix.min(axis=0)
    maxs = raw_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # 避免除零
    norm_matrix = (raw_matrix - mins) / ranges

    dim_names = list(dims.keys())
    return raw_values, norm_matrix, dim_names


def draw_radar(raw_values, norm_matrix, dim_names, output_path):
    """绘制五边形雷达图"""
    n_dims = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # 配色：深蓝→中蓝→浅蓝
    groups = [
        ('HC',    '#1a3a5c', 0.25),   # 深蓝
        ('D-NCI', '#3b7dd8', 0.20),   # 中蓝
        ('D-CI',  '#8fb8e8', 0.18),   # 浅蓝
    ]

    for label, color, alpha in groups:
        idx = list({'HC': 0, 'D-NCI': 1, 'D-CI': 2}.keys()).index(label)
        values = norm_matrix[idx].tolist()
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2.5, color=color,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                label=label, zorder=3)
        ax.fill(angles, values, color=color, alpha=alpha, zorder=2)

        # 标注原始数值
        for angle, val, raw in zip(angles[:-1], values, norm_matrix[idx]):
            # 在数据点外侧标注原始值
            offset = 0.08
            ax.annotate(f'{raw_values[label][angles[:-1].index(angle)]:.4f}',
                        xy=(angle, val),
                        xytext=(angle, val + offset),
                        ha='center', va='center',
                        fontsize=8.5, color=color, fontweight='bold')

    # 维度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=13, fontweight='bold')

    # 设置径向刻度
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
    ax.yaxis.grid(True, color='#cccccc', linestyle='--', linewidth=0.8)
    ax.xaxis.grid(True, color='#cccccc', linestyle='--', linewidth=0.8)

    # 背景色
    ax.set_facecolor('#f8f9fb')
    fig.patch.set_facecolor('#f8f9fb')

    # 标题
    ax.set_title('抑郁症认知障碍 EEG 五维特征雷达图',
                 fontsize=16, fontweight='bold', pad=25, color='#1a1a1a')

    # 图例
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15),
                       fontsize=12, framealpha=0.9, edgecolor='#cccccc')

    # 添加说明文字
    fig.text(0.5, 0.02,
             '数值为各维度 min-max 归一化结果 (0~1)，括号内为原始均值\n'
             'PLI = 相位锁指数 | RP = 相对功率 | FE = 模糊熵',
             ha='center', fontsize=9.5, color='#666666', style='italic')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#f8f9fb')
    plt.close()
    print(f"✅ 雷达图已保存: {output_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'features.csv'
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        print(f"用法: python {sys.argv[0]} <features.csv路径>")
        sys.exit(1)

    output_path = os.path.join(os.path.dirname(csv_path) or '.', 'radar_chart.png')

    print(f"📂 加载数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"   样本数: {len(df)}, 特征列: {len(df.columns) - 2}")

    raw_values, norm_matrix, dim_names = compute_features(df)
    if raw_values is None:
        sys.exit(1)

    print("\n📊 各维度原始均值:")
    print(f"{'维度':<12} {'HC':>10} {'D-NCI':>10} {'D-CI':>10}")
    print("-" * 45)
    for i, dim in enumerate(dim_names):
        print(f"{dim:<12} {raw_values['HC'][i]:>10.4f} {raw_values['D-NCI'][i]:>10.4f} {raw_values['D-CI'][i]:>10.4f}")

    print("\n🎨 生成雷达图...")
    draw_radar(raw_values, norm_matrix, dim_names, output_path)


if __name__ == '__main__':
    main()
