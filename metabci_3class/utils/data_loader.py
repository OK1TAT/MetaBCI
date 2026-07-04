# -*- coding: utf-8 -*-
"""
数据加载工具

负责:
- 扫描 EDF 文件目录
- 读取名单 Excel
- 按姓名匹配 EDF 与标签
- 构建统一的 subject_info 字典
"""

import os
import pandas as pd

from metabci_3class.config import MOCA_THRESHOLD


def build_subject_info(
    hc_dir: str,
    dep_dir: str,
    label_excel: str,
    moca_threshold: float = MOCA_THRESHOLD,
) -> dict:
    """
    构建被试信息字典

    Args:
        hc_dir: HC 组 EDF 文件目录
        dep_dir: 抑郁症组 EDF 文件目录
        label_excel: 抑郁症名单 Excel 路径
        moca_threshold: MoCA 阈值

    Returns:
        subject_info: {
            subject_id: {
                'edf_path': str,
                'label': int,      # 0=D-CI, 1=D-NCI, 2=HC
                'moca': float,
                'group': str       # 'HC' 或 'DEP'
            }
        }
    """
    subject_info = {}

    # HC 组
    if hc_dir and os.path.exists(hc_dir):
        edf_files = sorted([
            f for f in os.listdir(hc_dir)
            if f.lower().endswith('.edf')
        ])
        print(f"HC组: {len(edf_files)} 个EDF文件")

        for i, fname in enumerate(edf_files):
            sid = f"HC_{i+1:03d}"
            subject_info[sid] = {
                'edf_path': os.path.join(hc_dir, fname),
                'label': 2,        # HC = 2
                'moca': 30.0,      # 默认满分
                'group': 'HC',
            }

    # 抑郁症组
    if dep_dir and os.path.exists(dep_dir):
        # 加载名单
        df_label = pd.read_excel(label_excel)
        print(f"抑郁症名单: {len(df_label)} 人")

        # 列出 EDF 文件
        edf_files = sorted([
            f for f in os.listdir(dep_dir)
            if f.lower().endswith('.edf')
        ])

        matched = 0
        for _, row in df_label.iterrows():
            name = str(row['姓名']).strip()
            moca = float(row['MOCA评分'])
            label = 0 if moca < moca_threshold else 1  # D-CI=0, D-NCI=1

            # 匹配 EDF（文件名包含姓名）
            matched_edf = None
            for f in edf_files:
                if name in f:
                    matched_edf = f
                    break

            if matched_edf:
                sid = f"DEP_{int(row['用户指定顺序']):03d}"
                subject_info[sid] = {
                    'edf_path': os.path.join(dep_dir, matched_edf),
                    'label': label,
                    'moca': moca,
                    'group': 'DEP',
                }
                matched += 1

        print(f"抑郁症匹配: {matched}/{len(df_label)}")

    # 统计
    labels = [v['label'] for v in subject_info.values()]
    print(f"\n总被试: {len(subject_info)}")
    print(f"  HC:   {labels.count(2)}")
    print(f"  D-CI: {labels.count(0)}")
    print(f"  D-NCI:{labels.count(1)}")

    return subject_info
