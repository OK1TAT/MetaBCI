# -*- coding: utf-8 -*-
"""
表1-8,9: VisualStim + paradigm（官方 metabci）
运行: python demo_07_visualstim.py
"""
try:
    from metabci.brainstim.paradigm import VisualStim, paradigm
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")

def demo():
    print("VisualStim: content/duration/color 属性")
    print("paradigm(): VSObject/win/bg_color/fps 参数")
    if HAS:
        print("导入成功")
    print("\n✓ Demo完成: VisualStim + paradigm")

if __name__ == "__main__":
    demo()
