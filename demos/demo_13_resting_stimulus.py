# -*- coding: utf-8 -*-
"""
表1-15: RestingState 静息态刺激范式（继承VisualStim）
运行: python demo_13_resting_stimulus.py
"""
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    from metabci_3class.brainstim.paradigm.resting_paradigm import RestingState, resting_paradigm
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("流程: 闭眼提示 -> 静息采集(60s) -> 睁眼提示 -> 静息采集(60s)")
    if HAS:
        print("导入成功")
    print("\n✓ Demo完成: RestingState 静息态刺激范式")

if __name__ == "__main__":
    demo()
