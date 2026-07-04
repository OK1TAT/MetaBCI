"""
model_export.py v2 - 将 joblib/pickle 模型导出为 Java GUI 可用的 JSON 格式
支持:
  1. sklearn 线性模型 (SVC/Ridge/LogisticRegression/SGDClassifier 等) → 直接提取 coef_
  2. Pipeline → 自动解包取 final estimator
  3. 树模型/集成模型 (RandomForest/GradientBoosting/XGBoost 等) → 蒸馏为线性模型

用法:
  python model_export.py model.pkl
  python model_export.py model.pkl output.json
  python model_export.py model.pkl output.json features.csv

如果模型是非线性的(树/集成)，需要提供训练特征CSV用于蒸馏。
脚本会自动在同目录/上级目录查找 features.csv。
"""
import sys
import os
import json
import numpy as np


def find_features_csv(model_path):
    """在模型文件附近自动查找 features.csv"""
    candidates = []
    model_dir = os.path.dirname(os.path.abspath(model_path))
    # 同目录
    candidates.append(os.path.join(model_dir, "features.csv"))
    # 上级 data 目录
    parent = os.path.dirname(model_dir)
    candidates.append(os.path.join(parent, "data", "features.csv"))
    candidates.append(os.path.join(parent, "features.csv"))
    # 上上级 data 目录
    grandparent = os.path.dirname(parent)
    candidates.append(os.path.join(grandparent, "data", "features.csv"))
    candidates.append(os.path.join(grandparent, "features.csv"))
    # eeg_gui 同级 data
    candidates.append(os.path.join(parent, "..", "data", "features.csv"))

    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p
    return None


def load_features_csv(csv_path):
    """加载特征CSV，返回 (X, feature_names)
    自动排除：
      1. 非数值列（如受试者ID 'HC_001'）
      2. 最后一列如果是 label/target/class 等
    """
    import pandas as pd
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"  CSV原始列数: {len(df.columns)}, 列名: {list(df.columns[:5])}...")

    # 排除非数值列（如受试者ID）
    numeric_cols = []
    dropped_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            dropped_cols.append(col)
    if dropped_cols:
        print(f"  排除非数值列: {dropped_cols}")
        df = df[numeric_cols]

    # 最后一列如果是 label/target 则排除
    last_col = df.columns[-1].lower()
    if last_col in ('label', 'target', 'class', 'y', 'diagnosis', 'group'):
        print(f"  排除标签列: {df.columns[-1]}")
        df = df.iloc[:, :-1]

    X = df.values.astype(np.float64)
    return X, list(df.columns)


def unwrap_pipeline(model):
    """如果是 Pipeline，返回 (pipeline, final_estimator)"""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        return model, final
    return model, model


def extract_linear(model):
    """从线性模型提取 weights/biases/labels"""
    weights = None
    biases = None
    labels = None

    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        weights = np.array(model.coef_, dtype=np.float64)
        biases = np.array(model.intercept_, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
    else:
        raise ValueError(f"模型 {type(model).__name__} 不支持 coef_/intercept_ 提取")

    if hasattr(model, 'classes_'):
        labels = [str(c) for c in model.classes_]
    elif biases.shape[0] > 1:
        labels = [f"class_{i}" for i in range(biases.shape[0])]
    else:
        labels = ["class_0", "class_1"]

    n_classes = len(labels)
    n_features = weights.shape[-1]

    # 二分类: coef_ 是 (1, n_features) 但有 2 个类
    if weights.shape[0] == 1 and n_classes == 2:
        weights = np.vstack([-weights, weights])
        biases = np.array([-biases[0], biases[0]])

    return weights, biases, labels


def distill_to_linear(pipeline_or_model, X):
    """
    用模型的预测结果训练一个 LogisticRegression 来近似它（蒸馏）
    返回 weights, biases, labels
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print(f"  蒸馏: 用 {X.shape[0]} 个样本, {X.shape[1]} 维特征")

    # 用完整 pipeline 预测（包含 scaler 等预处理）
    y_pred = pipeline_or_model.predict(X)
    unique_classes = sorted(set(y_pred))
    print(f"  模型预测类别: {unique_classes}")

    # 先标准化特征，再训练线性模型
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练 LogisticRegression 拟合预测结果
    lr = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    lr.fit(X_scaled, y_pred)

    train_acc = lr.score(X_scaled, y_pred)
    print(f"  蒸馏模型训练准确率: {train_acc:.4f}")

    weights = np.array(lr.coef_, dtype=np.float64)
    biases = np.array(lr.intercept_, dtype=np.float64)
    labels = [str(c) for c in lr.classes_]

    # 二分类展开
    if weights.shape[0] == 1 and len(labels) == 2:
        weights = np.vstack([-weights, weights])
        biases = np.array([-biases[0], biases[0]])

    return weights, biases, labels


def convert(input_path, output_path=None, features_path=None):
    """转换 joblib/pickle 模型为 JSON"""
    import joblib

    print(f"加载模型: {input_path}")
    model = joblib.load(input_path)
    print(f"模型类型: {type(model).__name__}")

    # 解包 Pipeline
    full_model, final_estimator = unwrap_pipeline(model)
    print(f"最终估计器: {type(final_estimator).__name__}")

    if hasattr(final_estimator, 'classes_'):
        print(f"类别: {final_estimator.classes_}")

    # 尝试直接提取线性权重
    try:
        weights, biases, labels = extract_linear(final_estimator)
        print(f"直接提取: {len(labels)} 类, {weights.shape[-1]} 维特征")
    except ValueError:
        print(f"  {type(final_estimator).__name__} 非线性模型，使用蒸馏方法...")

        # 查找特征文件
        if features_path is None:
            features_path = find_features_csv(input_path)

        if features_path is None:
            print("错误: 非线性模型需要训练特征数据进行蒸馏。")
            print("用法: python model_export.py model.pkl output.json features.csv")
            sys.exit(1)

        print(f"  使用特征文件: {features_path}")
        X, feat_names = load_features_csv(features_path)
        print(f"  特征维度: {X.shape}")

        weights, biases, labels = distill_to_linear(full_model, X)

    print(f"\n导出结果: {len(labels)} 类, {weights.shape[-1]} 维特征")
    print(f"类别标签: {labels}")

    # 构建 JSON
    result = {
        "weights": weights.tolist(),
        "biases": biases.tolist(),
        "labels": labels
    }

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"导出完成: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python model_export.py <model.pkl|model.joblib> [output.json] [features.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith('.csv') else None
    features_file = None
    for arg in sys.argv[2:]:
        if arg.endswith('.csv'):
            features_file = arg

    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        sys.exit(1)

    convert(input_file, output_file, features_file)
