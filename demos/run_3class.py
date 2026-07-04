# -*- coding: utf-8 -*-
"""
三分类 EEG 认知障碍检测系统 - 主程序

完全基于 MetaBCI 平台 (brainda + brainstim)

运行流程:
    1. 构建被试信息
    2. 加载数据
    3. 提取特征
    4. 训练模型
    5. 保存结果

使用 MetaBCI 功能:

Brainda (6 个):
    1. BaseDataset         - 数据加载接口
    2. upper_ch_names      - 通道名标准化
    3. generate_filterbank - 多频段滤波器组
    4. set_random_seeds    - 全局随机种子
    5. EnhancedStratifiedKFold - 增强分层交叉验证
    6. Covariance          - 协方差估计（可选）

Brainstim (4 个):
    1. Experiment    - 实验框架
    2. VisualStim    - 视觉刺激基类
    3. paradigm      - 流程控制函数
    4. NeuroScanPort - 标签通信
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metabci_3class.config import (
    HC_DIR, DEP_DIR, LABEL_EXCEL, OUTPUT_DIR
)
from metabci_3class.utils.data_loader import build_subject_info
from metabci_3class.brainda.datasets import FrontalEEGDataset
from metabci_3class.brainda.paradigms import RestingStateParadigm
from metabci_3class.brainda.algorithms.feature_extraction import extract_frontal_features
from metabci_3class.brainda.algorithms.training import ThreeClassTrainer

def _save_features_csv(features, feature_names, y, meta, output_dir):
    import csv
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'features.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = feature_names + ['label', 'subject']
        writer.writerow(header)

        subject_list = meta['subject'] if 'subject' in meta else [''] * len(y)
        for i in range(len(features)):
            row = features[i].tolist() + [y[i], subject_list[i]]
            writer.writerow(row)

    print(f"[保存] 特征已保存: {csv_path}")
    print(f"  样本数: {len(features)}, 特征数: {len(feature_names)}")

def main():
    """主程序入口"""
    print("="*60)
    print("三分类 EEG 认知障碍检测系统")
    print("基于 MetaBCI 平台 (brainda + brainstim)")
    print("="*60)

    # Step 1: 构建被试信息
    print("\n[Step 1/5] 构建被试信息...")
    subject_info = build_subject_info(
        hc_dir=HC_DIR,
        dep_dir=DEP_DIR,
        label_excel=LABEL_EXCEL
    )

    if not subject_info:
        print("错误: 没有找到任何被试数据")
        return 1

    # Step 2: 创建数据集
    print("\n[Step 2/5] 创建数据集...")
    dataset = FrontalEEGDataset(subject_info)

    # Step 3: 加载数据
    print("\n[Step 3/5] 加载并预处理数据...")
    paradigm = RestingStateParadigm(dataset)
    X, y, meta = paradigm.get_data()

    if X is None:
        print("错误: 数据加载失败")
        return 1

    print(f"\n数据形状: X={X.shape}, y={y.shape}")

    # Step 4: 特征提取
    print("\n[Step 4/5] 提取特征...")
    features, feature_names = extract_frontal_features(X)
    print(f"特征形状: {features.shape}")
    print(f"特征数: {len(feature_names)}")
    _save_features_csv(features, feature_names, y, meta, OUTPUT_DIR)

    # Step 5: 训练模型
    print("\n[Step 5/5] 训练模型...")
    subject_ids = meta['subject'].tolist()
    trainer = ThreeClassTrainer(output_dir=OUTPUT_DIR)
    result = trainer.train(features, y, subject_ids)

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"结果保存在: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
