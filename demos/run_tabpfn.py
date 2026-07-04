# -*- coding: utf-8 -*-
"""
TabPFN 三分类训练脚本
从 feature.csv 直接加载特征 → TabPFN训练 → 输出模型和评估结果

用法:
    python demos/run_tabpfn.py                          # 默认读取 ./results_3class/features.csv
    python demos/run_tabpfn.py path/to/features.csv     # 指定CSV路径

CSV格式:
    - 特征列: PLI_*, RP_*, FE_* (135维)
    - label列: 0=D-CI, 1=D-NCI, 2=HC
    - subject列: 被试ID

环境要求:
    pip install tabpfn
    # HuggingFace认证（gated model）:
    #   1. 访问 https://huggingface.co/Prior-Labs/tabpfn-v2-classifier 接受条款
    #   2. 创建 token: https://huggingface.co/settings/tokens
    #   3. set HF_TOKEN=hf_xxxx  或  huggingface-cli login
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ============================================================
# 配置
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42
N_FOLDS = 5
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results_3class")

CLASS_NAMES = {0: "D-CI", 1: "D-NCI", 2: "HC"}
CLASS_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

# TabPFN 参数
TABPFN_N_estimators = 128    # 内部集成数
MAX_TRAIN_SAMPLES = 3000     # TabPFN 单次最大训练样本数

# 调参网格
PARAM_GRID = {
    'n_estimators': [64, 128, 256],        # 内部集成数
    'max_train_samples': [1500, 3000],      # 子采样上限
}


def detect_device():
    """自动检测GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] {name}")
            return 'cuda'
        else:
            print("[CPU] 无GPU，使用CPU")
            return 'cpu'
    except ImportError:
        print("[CPU] 未安装torch")
        return 'cpu'


# ============================================================
# 数据加载
# ============================================================
def load_features(csv_path):
    """
    加载 feature.csv

    Returns:
        X: ndarray (n_samples, n_features)
        y: ndarray (n_samples,)
        subjects: list of str
        feature_names: list of str
    """
    print(f"\n{'='*60}")
    print(f"加载数据: {csv_path}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"数据形状: {df.shape}")

    # 分离特征和标签
    meta_cols = ['label', 'subject']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float64)
    y = df['label'].values.astype(int)
    subjects = df['subject'].astype(str).tolist()
    feature_names = feature_cols

    print(f"特征维度: {X.shape}")
    print(f"特征列数: {len(feature_names)}")
    print(f"\n类别分布:")
    for label in sorted(CLASS_NAMES.keys()):
        n = np.sum(y == label)
        print(f"  {CLASS_NAMES[label]}: {n} ({n/len(y)*100:.1f}%)")
    print(f"总样本: {len(y)}")
    print(f"被试数: {len(set(subjects))}")

    # 检查缺失值
    n_nan = np.isnan(X).sum()
    if n_nan > 0:
        print(f"\n[警告] 发现 {n_nan} 个NaN，用0填充")
        X = np.nan_to_num(X, nan=0.0)

    return X, y, subjects, feature_names


# ============================================================
# TabPFN 集成下采样
# ============================================================
def balance_classes(X, y, rng):
    """类别均衡下采样"""
    classes, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    if len(set(counts)) == 1:
        return X, y

    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        selected = rng.choice(cls_idx, size=min_count, replace=False)
        indices.extend(selected)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_tabpfn_ensemble(X_train, y_train, X_test, n_estimators=1,
                          max_samples=MAX_TRAIN_SAMPLES, rng=None,
                          tabpfn_n_estimators=TABPFN_N_estimators,
                          device='cpu'):
    """
    TabPFN 集成训练

    当训练样本超过 max_samples 时，多次子采样集成。
    """
    from tabpfn import TabPFNClassifier

    if rng is None:
        rng = np.random.RandomState(RANDOM_SEED)

    all_proba = []

    for i in range(n_estimators):
        # 子采样
        if len(X_train) > max_samples:
            idx = rng.choice(len(X_train), size=max_samples, replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]
            # 类别均衡
            X_sub, y_sub = balance_classes(X_sub, y_sub, rng)
        else:
            X_sub, y_sub = balance_classes(X_train.copy(), y_train.copy(), rng)

        # TabPFN
        clf = TabPFNClassifier(
            n_estimators=tabpfn_n_estimators,
            device=device,
            random_state=RANDOM_SEED + i
        )
        clf.fit(X_sub, y_sub)
        proba = clf.predict_proba(X_test)
        all_proba.append(proba)

    # 集成平均
    avg_proba = np.mean(all_proba, axis=0)
    return avg_proba, clf


# ============================================================
# 交叉验证训练
# ============================================================
def cross_validate_tabpfn(X, y, subjects, feature_names, output_dir, best_params=None):
    """
    GroupKFold 交叉验证 + TabPFN

    Returns:
        dict: 完整训练结果
    """
    print(f"\n{'='*60}")
    print(f"TabPFN 交叉验证训练")
    print(f"{'='*60}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 交叉验证
    subject_ids = np.array(subjects)
    gkf = GroupKFold(n_splits=N_FOLDS)
    rng = np.random.RandomState(RANDOM_SEED)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=subject_ids)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = subject_ids[train_idx]

        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        print(f"  训练: {len(train_idx)} 样本, {len(np.unique(g_train))} 被试")
        print(f"  测试: {len(test_idx)} 样本, {len(np.unique(subject_ids[test_idx]))} 被试")

        # 训练集样本量
        unique_train_subjects = len(np.unique(g_train))
        n_ensemble = max(1, unique_train_subjects // 50)  # 按被试数决定集成数

        # 使用调参结果或默认值
        tabpfn_n_est = TABPFN_N_estimators
        max_samples = MAX_TRAIN_SAMPLES
        if best_params:
            tabpfn_n_est = best_params.get('n_estimators', TABPFN_N_estimators)
            max_samples = best_params.get('max_train_samples', MAX_TRAIN_SAMPLES)

        # TabPFN 训练 + 预测
        y_proba, final_clf = train_tabpfn_ensemble(
            X_train, y_train, X_test,
            n_estimators=n_ensemble,
            max_samples=max_samples,
            rng=rng,
            tabpfn_n_estimators=tabpfn_n_est,
            device=detect_device()
        )
        y_pred = np.argmax(y_proba, axis=1)

        # 评估
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        print(f"\n  混淆矩阵:")
        print(f"            D-CI  D-NCI  HC")
        for i, name in enumerate(CLASS_NAMES.values()):
            print(f"  {name:<8} {cm[i]}")

        # 分类报告
        print(f"\n  分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
                                    zero_division=0))

        # 收集
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        fold_models.append(final_clf)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        })

    # 总体指标
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    overall = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
        'recall': recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
        'f1': f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    }

    print(f"\n{'='*60}")
    print(f"总体结果")
    print(f"{'='*60}")
    print(f"Accuracy:  {overall['accuracy']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1:        {overall['f1']:.4f}")

    # 保存
    results = {
        'fold_results': fold_results,
        'overall': overall,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_proba': all_y_proba,
        'scaler': scaler,
        'fold_models': fold_models,
        'feature_names': feature_names
    }

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    joblib.dump(results, os.path.join(output_dir, 'models', 'tabpfn_results.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'models', 'scaler.joblib'))
    joblib.dump(fold_models, os.path.join(output_dir, 'models', 'fold_models.joblib'))
    print(f"\n[保存] 模型已保存到: {output_dir}/models/")

    return results


# ============================================================
# 超参数调优
# ============================================================
def tune_hyperparameters(X, y, subjects, param_grid=PARAM_GRID, n_folds=3):
    """
    TabPFN 超参数网格搜索

    对每组参数做 n_folds 折 GroupKFold 交叉验证，选 F1 最高的组合。

    Args:
        X: ndarray (n_samples, n_features) 原始特征（未标准化）
        y: ndarray (n_samples,)
        subjects: list
        param_grid: dict, 参数网格
        n_folds: int, 调参用折数（默认3，比正式训练快）

    Returns:
        dict: {'best_params': {...}, 'best_f1': float, 'all_results': [...]}
    """
    from tabpfn import TabPFNClassifier
    from itertools import product

    print(f"\n{'='*60}")
    print(f"超参数调优 (网格搜索)")
    print(f"{'='*60}")

    # 生成参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    print(f"参数网格: {len(combos)} 组组合")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    subject_ids = np.array(subjects)
    gkf = GroupKFold(n_splits=n_folds)
    rng = np.random.RandomState(RANDOM_SEED)
    device = detect_device()

    all_results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        n_est = params.get('n_estimators', TABPFN_N_estimators)
        max_samples = params.get('max_train_samples', MAX_TRAIN_SAMPLES)

        print(f"\n--- 组合 {idx+1}/{len(combos)}: {params} ---")

        fold_f1s = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=subject_ids)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train = subject_ids[train_idx]

            # 子采样 + 类别均衡
            if len(X_train) > max_samples:
                sub_idx = rng.choice(len(X_train), size=max_samples, replace=False)
                X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
                X_sub, y_sub = balance_classes(X_sub, y_sub, rng)
            else:
                X_sub, y_sub = balance_classes(X_train.copy(), y_train.copy(), rng)

            # TabPFN
            clf = TabPFNClassifier(
                n_estimators=n_est,
                device=device,
                random_state=RANDOM_SEED
            )
            clf.fit(X_sub, y_sub)
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            fold_f1s.append(f1)
            print(f"  Fold {fold+1}: F1={f1:.4f}")

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        print(f"  >>> 平均 F1: {mean_f1:.4f} ± {std_f1:.4f}")

        all_results.append({
            'params': params,
            'fold_f1s': fold_f1s,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        })

    # 排序找最优
    all_results.sort(key=lambda x: x['mean_f1'], reverse=True)

    print(f"\n{'='*60}")
    print(f"调参结果排名")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'F1(mean±std)':<20} {'参数'}")
    print(f"{'-'*60}")
    for i, r in enumerate(all_results):
        print(f"#{i+1:<4} {r['mean_f1']:.4f}±{r['std_f1']:.4f}    {r['params']}")

    best = all_results[0]
    print(f"\n最优参数: {best['params']}")
    print(f"最优 F1:  {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")

    # 保存调参结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump({
        'best_params': best['params'],
        'best_f1': best['mean_f1'],
        'all_results': all_results
    }, os.path.join(OUTPUT_DIR, 'models', 'tuning_results.joblib'))
    print(f"[保存] 调参结果: {OUTPUT_DIR}/models/tuning_results.joblib")

    return best['params']


# ============================================================
# 可视化
# ============================================================
def plot_results(results, output_dir):
    """绘制混淆矩阵 + ROC曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    # ---- 混淆矩阵 ----
    cm = results['overall']['confusion_matrix']
    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (TabPFN)')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'figures', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[图片] 混淆矩阵: {cm_path}")

    # ---- ROC曲线 (one-vs-rest) ----
    y_true = results['all_y_true']
    y_proba = results['all_y_proba']

    fig, ax = plt.subplots(figsize=(8, 6))

    for cls_id, cls_name in CLASS_NAMES.items():
        # 二值化
        y_bin = (y_true == cls_id).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, cls_id])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[cls_id],
                lw=2, label=f'{cls_name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (TabPFN, One-vs-Rest)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'figures', 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[图片] ROC曲线: {roc_path}")

    # ---- 各折指标柱状图 ----
    fold_results = results['fold_results']
    folds = [r['fold'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#3498db')
    ax.bar(x + width/2, f1s, width, label='F1-macro', color='#e74c3c')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('TabPFN Cross-Validation Results')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'figures', 'fold_metrics.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[图片] 各折指标: {bar_path}")


# ============================================================
# 报告生成
# ============================================================
def save_report(results, output_dir):
    """保存文本报告"""
    report_path = os.path.join(output_dir, 'tabpfn_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TabPFN 三分类训练报告\n")
        f.write("=" * 60 + "\n\n")

        # 总体
        ov = results['overall']
        f.write("总体指标:\n")
        f.write(f"  Accuracy:  {ov['accuracy']:.4f}\n")
        f.write(f"  Precision: {ov['precision']:.4f}\n")
        f.write(f"  Recall:    {ov['recall']:.4f}\n")
        f.write(f"  F1:        {ov['f1']:.4f}\n\n")

        # 各折
        f.write("各折结果:\n")
        f.write(f"  {'Fold':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}\n")
        f.write(f"  {'-'*38}\n")
        for r in results['fold_results']:
            f.write(f"  {r['fold']:<6} {r['accuracy']:.4f}   {r['precision']:.4f}   "
                    f"{r['recall']:.4f}   {r['f1']:.4f}\n")

        accs = [r['accuracy'] for r in results['fold_results']]
        f1s = [r['f1'] for r in results['fold_results']]
        f.write(f"\n  Mean±Std:\n")
        f.write(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"  F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")

        # 混淆矩阵
        f.write(f"\n总体混淆矩阵:\n")
        f.write(f"            D-CI  D-NCI  HC\n")
        cm = ov['confusion_matrix']
        for i, name in enumerate(CLASS_NAMES.values()):
            f.write(f"  {name:<8} {cm[i]}\n")

        # 分类报告
        f.write(f"\n详细分类报告:\n")
        f.write(classification_report(
            results['all_y_true'], results['all_y_pred'],
            target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
            zero_division=0
        ))

    print(f"[报告] {report_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='TabPFN 三分类训练')
    parser.add_argument('csv_path', nargs='?', default=None,
                        help='feature.csv 路径 (默认 ./results_3class/features.csv)')
    parser.add_argument('--no-tune', action='store_true',
                        help='跳过调参，直接用默认参数训练')
    parser.add_argument('--tune-folds', type=int, default=3,
                        help='调参用折数 (默认3)')
    parser.add_argument('--use-tuned', action='store_true',
                        help='直接加载已有调参结果训练（不重新搜索）')
    args = parser.parse_args()

    # CSV路径
    csv_path = args.csv_path or os.path.join(OUTPUT_DIR, 'features.csv')

    if not os.path.exists(csv_path):
        print(f"[错误] 文件不存在: {csv_path}")
        print(f"用法: python demos/run_tabpfn.py [feature_csv_path] [--tune]")
        print(f"默认路径: {os.path.join(OUTPUT_DIR, 'features.csv')}")
        return 1

    print("=" * 60)
    print("TabPFN 三分类训练 (HC / D-CI / D-NCI)")
    print("=" * 60)

    # 检测GPU
    device = detect_device()

    # 加载数据
    X, y, subjects, feature_names = load_features(csv_path)

    # 调参
    best_params = None
    if args.use_tuned:
        # 加载已有调参结果
        tuned_path = os.path.join(OUTPUT_DIR, 'models', 'tuning_results.joblib')
        if os.path.exists(tuned_path):
            tuned = joblib.load(tuned_path)
            best_params = tuned['best_params']
            print(f"\n[加载] 已有调参结果: {best_params} (F1={tuned['best_f1']:.4f})")
        else:
            print(f"[警告] 未找到调参结果: {tuned_path}，使用默认参数")
    elif not args.no_tune:
        # 默认行为：调参
        best_params = tune_hyperparameters(X, y, subjects, n_folds=args.tune_folds)
        print(f"\n将使用最优参数进行正式训练...")

    # 交叉验证训练
    results = cross_validate_tabpfn(X, y, subjects, feature_names, OUTPUT_DIR,
                                    best_params=best_params)

    # 可视化
    print(f"\n{'='*60}")
    print("生成可视化...")
    print(f"{'='*60}")
    plot_results(results, OUTPUT_DIR)

    # 文本报告
    save_report(results, OUTPUT_DIR)

    # 完成
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  models/          - 模型文件 (scaler + fold_models)")
    print(f"  models/tuning_results.joblib - 调参结果 (如执行了--tune)")
    print(f"  figures/         - 可视化 (混淆矩阵, ROC, 各折指标)")
    print(f"  tabpfn_report.txt - 文本报告")
    print(f"  tabpfn_results.joblib - 完整结果")

    return 0


if __name__ == '__main__':
    sys.exit(main())
