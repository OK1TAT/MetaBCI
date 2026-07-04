# -*- coding: utf-8 -*-
"""
表1-14: RestingExperiment 静息态实验框架
运行: python demo_12_resting_experiment.py
"""
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    from metabci_3class.brainstim.framework.resting_framework import RestingExperiment
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("RestingExperiment: 封装Experiment, 重写warmup(strict=False)")
    if HAS:
        print("导入成功")
    print("\n✓ Demo完成: RestingExperiment")

if __name__ == "__main__":
    demo()
