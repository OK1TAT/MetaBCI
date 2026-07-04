# -*- coding: utf-8 -*-
"""
表1-2: upper_ch_names 通道名大写标准化（官方 metabci）
运行: python demo_02_channel_names.py
"""
import numpy as np
try:
    from metabci.brainda.utils.channels import upper_ch_names
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")

def demo():
    raw_names = ["Fp1", "Fp2", "f3", "F4", "f7", "F8", "Cz", "pz"]
    print("原始通道名:", raw_names)
    upper = [n.upper() for n in raw_names]
    print("标准化后:", upper)
    if HAS:
        print("(upper_ch_names(raw) 直接修改 Raw 对象)")
    print("\n✓ Demo完成: upper_ch_names 通道名标准化")

if __name__ == "__main__":
    demo()
