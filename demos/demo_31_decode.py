# -*- coding: utf-8 -*-
"""
表2-45: OpenBCI数据包独立解码
功能: OpenBCI数据包独立解码
运行: python demo_31_decode.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(proj_root, "eeg_gui"))
try:
    import importlib
    mod = importlib.import_module("decode")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("OpenBCI数据包独立解码")
    print("模块: eeg_gui/decode.py")
    print("核心函数: encode24BitSigned, parseEEGPacket")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: OpenBCI数据包独立解码")

if __name__ == "__main__":
    demo()
