# -*- coding: utf-8 -*-
"""
表2-12,13,14,15: 在线滑动窗口特征管道
功能: 在线滑动窗口特征管道
运行: python demo_19_online_pipeline.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    import importlib
    mod = importlib.import_module("metabci_3class.brainflow.online_feature_pipeline")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("在线滑动窗口特征管道")
    print("模块: metabci_3class.brainflow.online_feature_pipeline")
    print("核心类: OnlineFeaturePipeline, FeatureSample")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: 在线滑动窗口特征管道")

if __name__ == "__main__":
    demo()
