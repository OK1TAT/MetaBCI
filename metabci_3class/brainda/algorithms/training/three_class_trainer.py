# -*- coding: utf-8 -*-
"""
三分类训练器 (HC / D-CI / D-NCI)

分类器: TabPFN (集成下采样)
策略: GroupKFold 按被试分组 + 集成下采样 + 类别均衡

Brainda 功能使用:
    1. set_random_seeds       - 全局随机种子
    2. EnhancedStratifiedKFold - 增强分层交叉验证
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# TabPFN
try:
    from tabpfn_client import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

# Brainda 组件
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds
)

from metabci_3class.config import (
    RANDOM_SEED, N_FOLDS, CLASS_NAMES, OUTPUT_DIR
)

# TabPFN 集成下采样参数
ENSEMBLE_N_SAMPLES = 5   # 集成采样次数
MAX_TRAIN_SUBJECTS = 200 # 单次训练最大被试数


def detect_device():
    """自动检测GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] 检测到: {name}")
            return 'cuda'
    except ImportError:
        pass
    print("[CPU] 未检测到 GPU，使用 CPU 模式")
    return 'cpu'


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


def downsample_subjects(X, y, groups, target_n, rng):
    """按被试下采样，保持被试完整性"""
    unique_subjects = np.unique(groups)
    if len(unique_subjects) <= target_n:
        return X, y, groups
    selected = rng.choice(unique_subjects, size=target_n, replace=False)
    mask = np.isin(groups, selected)
    return X[mask], y[mask], groups[mask]


class ThreeClassTrainer:
    """
    三分类训练器 (HC / D-CI / D-NCI)

    策略:
        - GroupKFold 按被试分组（防数据泄露）
        - TabPFN 集成下采样
        - 类别均衡
    """

    def __init__(self, n_folds=N_FOLDS, random_seed=RANDOM_SEED, output_dir=OUTPUT_DIR):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.output_dir = output_dir

        # Brainda: 设置随机种子
        set_random_seeds(self.random_seed)

        # 检测GPU
        self.device = detect_device()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

        # 模型和 scaler
        self.scaler = None
        self.fold_models = []
        self.fold_results = []

        print(f"[ThreeClassTrainer] 初始化: {n_folds}折, seed={random_seed}, device={self.device}")

    def train(self, X, y, subject_ids):
        """
        训练模型

        Args:
            X: ndarray, shape (n_samples, n_features)
            y: ndarray, shape (n_samples,)
            subject_ids: list, 被试ID列表（用于 GroupKFold）

        Returns:
            dict: 训练结果
        """
        if not HAS_TABPFN:
            raise ImportError(
                "TabPFN 未安装！\n"
                "安装方法: pip install tabpfn"
            )

        print(f"\n{'='*60}")
        print(f"开始训练 (TabPFN)")
        print(f"{'='*60}")
        print(f"样本数: {X.shape[0]}")
        print(f"特征数: {X.shape[1]}")
        print(f"类别分布:")
        for label, count in zip(*np.unique(y, return_counts=True)):
            print(f"  {CLASS_NAMES[label]}: {count}")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 交叉验证
        self._cross_validate(X_scaled, y, subject_ids)

        # 保存模型
        self._save_models()

        # 输出汇总
        self._print_summary()

        return {
            'fold_results': self.fold_results,
            'fold_models': self.fold_models,
            'scaler': self.scaler
        }

    def _cross_validate(self, X, y, subject_ids):
        """GroupKFold 交叉验证 + TabPFN 集成下采样"""
        subject_ids = np.array(subject_ids)
        gkf = GroupKFold(n_splits=self.n_folds)
        rng = np.random.RandomState(self.random_seed)

        print(f"\n--- {self.n_folds}折交叉验证 (TabPFN 集成下采样) ---")

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train = subject_ids[train_idx]

            print(f"\nFold {fold + 1}/{self.n_folds}")
            print(f"  训练集: {len(train_idx)} (被试: {len(np.unique(g_train))})")
            print(f"  测试集: {len(test_idx)} (被试: {len(np.unique(subject_ids[test_idx]))})")

            # TabPFN 集成下采样
            unique_train_subjects = len(np.unique(g_train))
            target_subjects = min(unique_train_subjects, MAX_TRAIN_SUBJECTS)

            if unique_train_subjects > target_subjects:
                # 多次采样集成
                all_pred_proba = []
                fold_model_list = []

                for s in range(ENSEMBLE_N_SAMPLES):
                    X_s, y_s, g_s = downsample_subjects(
                        X_train, y_train, g_train, target_subjects, rng)
                    X_s, y_s = balance_classes(X_s, y_s, rng)

                    clf = TabPFNClassifier()
                    clf.fit(X_s, y_s)
                    proba = clf.predict_proba(X_test)
                    all_pred_proba.append(proba)
                    fold_model_list.append(clf)
                    print(f"    采样 {s+1}/{ENSEMBLE_N_SAMPLES} (N={len(X_s)})")

                y_pred_proba = np.mean(all_pred_proba, axis=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # 被试数不多，直接类别均衡
                X_bal, y_bal = balance_classes(X_train, y_train, rng)
                clf = TabPFNClassifier()
                clf.fit(X_bal, y_bal)
                y_pred = clf.predict(X_test)
                fold_model_list = [clf]
                print(f"    直接训练 (N={len(X_bal)})")

            # 评估
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            print(f"  Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            print(f"  混淆矩阵:")
            print(f"    {cm[0]}  (D-CI)")
            print(f"    {cm[1]}  (D-NCI)")
            print(f"    {cm[2]}  (HC)")

            # 保存
            self.fold_models.append(fold_model_list)
            self.fold_results.append({
                'fold': fold + 1,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm
            })

    def predict(self, X):
        """
        使用所有折模型集成预测

        Args:
            X: ndarray, shape (n_samples, n_features)

        Returns:
            y_pred: ndarray
        """
        if self.scaler is None:
            raise RuntimeError("模型未训练")

        X_scaled = self.scaler.transform(X)

        all_proba = []
        for fold_models in self.fold_models:
            for clf in fold_models:
                proba = clf.predict_proba(X_scaled)
                all_proba.append(proba)

        avg_proba = np.mean(all_proba, axis=0)
        y_pred = np.argmax(avg_proba, axis=1)
        return y_pred

    def _save_models(self):
        """保存模型"""
        model_dir = os.path.join(self.output_dir, 'models')

        # 保存 scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))

        # 保存每折模型（每折含多个子模型）
        joblib.dump(self.fold_models, os.path.join(model_dir, 'fold_models.joblib'))

        # 保存配置
        config = {
            'n_folds': self.n_folds,
            'random_seed': self.random_seed,
            'class_names': CLASS_NAMES,
            'classifier': 'TabPFN',
            'ensemble_n': ENSEMBLE_N_SAMPLES,
            'device': self.device
        }
        joblib.dump(config, os.path.join(model_dir, 'config.joblib'))

        print(f"\n[保存] 模型已保存到: {model_dir}")

    def _print_summary(self):
        """输出训练汇总"""
        print(f"\n{'='*60}")
        print(f"训练汇总 (TabPFN)")
        print(f"{'='*60}")

        accuracies = [r['accuracy'] for r in self.fold_results]
        precisions = [r['precision'] for r in self.fold_results]
        recalls = [r['recall'] for r in self.fold_results]
        f1s = [r['f1'] for r in self.fold_results]

        print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
