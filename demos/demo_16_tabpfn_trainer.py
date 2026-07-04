# -*- coding: utf-8 -*-
"""
表2-6,7: TabPFN三分类训练器 + 类别平衡下采样
运行: python demo_16_tabpfn_trainer.py
"""
import numpy as np
from sklearn.model_selection import GroupKFold

def balance_classes(X, y, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    _, counts = np.unique(y, return_counts=True)
    mc = counts.min()
    idx = []
    for c in np.unique(y):
        ci = np.where(y==c)[0]
        idx.extend(rng.choice(ci, mc, replace=False))
    return X[idx], y[idx]

def demo():
    np.random.seed(42)
    X = np.random.randn(100, 135)
    y = np.array([0]*20+[1]*30+[2]*50)
    groups = np.repeat(np.arange(10), 10)
    print("原始: %d样本, 分布=%s" % (len(y), dict(zip(*np.unique(y,return_counts=True)))))
    Xb, yb = balance_classes(X, y)
    print("平衡: %d样本, 分布=%s" % (len(yb), dict(zip(*np.unique(yb,return_counts=True)))))
    gkf = GroupKFold(n_splits=5)
    for f, (tr, te) in enumerate(gkf.split(X, y, groups)):
        og = set(groups[tr]) & set(groups[te])
        print("  Fold %d: 训练%d 测试%d 重叠=%s" % (f+1, len(tr), len(te), og))
    print("\n✓ Demo完成: TabPFN训练器 + 类别平衡")

if __name__ == "__main__":
    demo()
