# -*- coding: utf-8 -*-
"""
表2-9,10,11: LSL实时采集适配器
功能: LSL实时采集适配器
运行: python demo_18_lsl_adapter.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    import importlib
    mod = importlib.import_module("metabci_3class.brainflow.lsl_adapter")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("LSL实时采集适配器")
    print("模块: metabci_3class.brainflow.lsl_adapter")
    print("核心类: LSLAdapter, LSLStreamDiscovery, SyntheticLSLSource")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: LSL实时采集适配器")

if __name__ == "__main__":
    demo()
