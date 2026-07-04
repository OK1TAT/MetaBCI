# -*- coding: utf-8 -*-
"""
export_tabpfn.py - 将 TabPFN 三分类模型蒸馏导出为 Java GUI 可用的 JSON 格式

流程:
  1. 加载 tabpfn_results.joblib (含 fold_models + scaler + feature_names)
  2. 加载 features.csv (135维)
  3. 用 scaler 标准化特征
  4. 用 5折 TabPFN 模型集成预测全部样本
  5. 用 LogisticRegression 蒸馏 (拟合集成预测结果)
  6. 导出 weights/biases/labels JSON

用法:
  python export_tabpfn.py
  python export_tabpfn.py results_3class/models/tabpfn_results.joblib
  python export_tabpfn.py results_3class/models/tabpfn_results.joblib output.json data/features.csv
"""
import sys
import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = {0: "D-CI", 1: "D-NCI", 2: "HC"}


def load_features_csv(csv_path):
    """加载特征CSV，排除非数值列和标签列"""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"  CSV原始列数: {len(df.columns)}")

    # 排除非数值列
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass
    df = df[numeric_cols]

    # 排除标签列
    last_col = df.columns[-1].lower()
    if last_col in ('label', 'target', 'class', 'y', 'diagnosis', 'group'):
        df = df.iloc[:, :-1]

    X = df.values.astype(np.float64)
    print(f"  特征维度: {X.shape}")
    return X, list(df.columns)


def main():
    # 解析参数
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results_3class/models/tabpfn_results.joblib"
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith('.csv') else "tabpfn_model.json"
    csv_path = None
    for arg in sys.argv[2:]:
        if arg.endswith('.csv'):
            csv_path = arg

    # 默认CSV路径
    if csv_path is None:
        for p in ["data/features.csv", "../data/features.csv", "features.csv"]:
            if os.path.exists(p):
                csv_path = p
                break

    print(f"加载 TabPFN 结果: {results_path}")
    if not os.path.exists(results_path):
        print(f"❌ 文件不存在: {results_path}")
        print("请先运行 python demos/run_tabpfn.py 训练模型")
        sys.exit(1)

    results = joblib.load(results_path)

    fold_models = results['fold_models']
    scaler = results['scaler']
    feature_names_stored = results.get('feature_names', None)
    n_models = len(fold_models)
    print(f"  折模型数: {n_models}")
    print(f"  类别: {list(CLASS_NAMES.values())}")

    # 加载特征
    if csv_path is None:
        print("❌ 找不到 features.csv，请指定路径")
        print("用法: python export_tabpfn.py model.joblib output.json path/to/features.csv")
        sys.exit(1)

    print(f"\n加载特征: {csv_path}")
    X, feat_names = load_features_csv(csv_path)

    # 标准化
    X_scaled = scaler.transform(X)
    print(f"  标准化后: {X_scaled.shape}")

    # 集成预测：所有fold模型预测全部样本，取平均概率
    print(f"\n集成预测 ({n_models}个模型)...")
    all_proba = np.zeros((X.shape[0], 3))
    for i, model in enumerate(fold_models):
        proba = model.predict_proba(X_scaled)
        # 确保概率矩阵列数=3
        if proba.shape[1] != 3:
            # 补齐缺失类别
            full = np.zeros((X.shape[0], 3))
            for j, c in enumerate(model.classes_):
                full[:, c] = proba[:, j]
            proba = full
        all_proba += proba
        print(f"  Fold {i+1}/{n_models} 完成")

    all_proba /= n_models
    y_pred = np.argmax(all_proba, axis=1)

    print(f"\n集成预测类别分布:")
    for c in range(3):
        count = np.sum(y_pred == c)
        print(f"  {CLASS_NAMES[c]}: {count} ({count/len(y_pred)*100:.1f}%)")

    # 蒸馏：LogisticRegression 拟合集成预测
    print(f"\n蒸馏训练 (LogisticRegression)...")
    lr = LogisticRegression(max_iter=5000, C=1.0, random_state=42, multi_class='multinomial')
    lr.fit(X_scaled, y_pred)

    train_acc = lr.score(X_scaled, y_pred)
    print(f"  蒸馏准确率: {train_acc:.4f}")
    print(f"  蒸馏 vs 集成一致率: {np.mean(lr.predict(X_scaled) == y_pred):.4f}")

    # 提取权重
    weights = np.array(lr.coef_, dtype=np.float64)   # (3, 135)
    biases = np.array(lr.intercept_, dtype=np.float64)  # (3,)
    labels = [CLASS_NAMES[c] for c in lr.classes_]

    print(f"\n导出结果: {len(labels)} 类, {weights.shape[-1]} 维特征")
    print(f"  类别: {labels}")
    print(f"  权重矩阵: {weights.shape}")
    print(f"  偏置: {biases.shape}")

    # 保存JSON
    result = {
        "weights": weights.tolist(),
        "biases": biases.tolist(),
        "labels": labels
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 导出完成: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"\nJava GUI 载入此 JSON 即可实现三分类预测")


if __name__ == '__main__':
    main()
