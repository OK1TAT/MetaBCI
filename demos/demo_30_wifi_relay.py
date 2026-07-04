# -*- coding: utf-8 -*-
"""
表2-44: WiFi中继控制
功能: WiFi中继控制
运行: python demo_30_wifi_relay.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(proj_root, "eeg_gui"))
try:
    import importlib
    mod = importlib.import_module("wifi_relay")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("WiFi中继控制")
    print("模块: eeg_gui/wifi_relay.py")
    print("核心函数: http")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: WiFi中继控制")

if __name__ == "__main__":
    demo()
