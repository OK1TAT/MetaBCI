# -*- coding: utf-8 -*-
"""
表2-31: 数据异步保存线程
功能: 数据异步保存线程
运行: python demo_27_data_saver.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    import importlib
    mod = importlib.import_module("metabci_3class.brainflow.devices.data_saver")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("数据异步保存线程")
    print("模块: metabci_3class.brainflow.devices.data_saver")
    print("核心类: DataSaverThread, RawStreamSaverThread")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: 数据异步保存线程")

if __name__ == "__main__":
    demo()
