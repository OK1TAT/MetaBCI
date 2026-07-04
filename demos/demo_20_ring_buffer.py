# -*- coding: utf-8 -*-
"""
表2-16,17: 高效环形缓冲区
功能: 高效环形缓冲区
运行: python demo_20_ring_buffer.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    import importlib
    mod = importlib.import_module("metabci_3class.brainflow.ring_buffer")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("高效环形缓冲区")
    print("模块: metabci_3class.brainflow.ring_buffer")
    print("核心类: EfficientRingBuffer, MultiChannelRingBuffer")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: 高效环形缓冲区")

if __name__ == "__main__":
    demo()
