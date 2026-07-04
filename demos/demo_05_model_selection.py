# -*- coding: utf-8 -*-
"""
表1-5,6: set_random_seeds + EnhancedStratifiedKFold（官方 metabci）
运行: python demo_05_model_selection.py
"""
import numpy as np
try:
    from metabci.brainda.algorithms.utils.model_selection import set_random_seeds, EnhancedStratifiedKFold
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")
    from sklearn.model_selection import StratifiedKFold

def demo():
    seed = 42
    print("设定随机种子:", seed)
    if HAS:
        set_random_seeds(seed)
    else:
        np.random.seed(seed)
    X = np.random.randn(100, 10)
    y = np.array([0]*30 + [1]*30 + [2]*40)
    kfold = (EnhancedStratifiedKFold if HAS else StratifiedKFold)(n_splits=5, shuffle=True, random_state=seed)
    print("\n分5层5折交叉验证:")
    for fold, (tr, te) in enumerate(kfold.split(X, y)):
        print("  Fold %d: 训练%d, 测试%d" % (fold+1, len(tr), len(te)))
    print("\n✓ Demo完成: set_random_seeds + EnhancedStratifiedKFold")

if __name__ == "__main__":
    demo()
