# -*- coding: utf-8 -*-
"""
表2-43: Python-Java数据桥接
功能: Python-Java数据桥接
运行: python demo_29_bridge.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(proj_root, "eeg_gui"))
try:
    import importlib
    mod = importlib.import_module("eeg_bridge")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("Python-Java数据桥接")
    print("模块: eeg_gui/eeg_bridge.py")
    print("核心函数: RealtimeEEGProcessor")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: Python-Java数据桥接")

if __name__ == "__main__":
    demo()
