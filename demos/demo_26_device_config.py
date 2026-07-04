# -*- coding: utf-8 -*-
"""
表2-30: 设备配置管理
功能: 设备配置管理
运行: python demo_26_device_config.py
"""
import sys, os, numpy as np
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    import importlib
    mod = importlib.import_module("metabci_3class.brainflow.devices.config")
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("设备配置管理")
    print("模块: metabci_3class.brainflow.devices.config")
    print("核心类: DeviceConfig, create_config")
    if HAS:
        comps = [n for n in dir(mod) if not n.startswith('_')]
        print("导入成功! 可用组件:", comps[:8])
    else:
        print("(请确保项目依赖已安装)")
    print("\n✓ Demo完成: 设备配置管理")

if __name__ == "__main__":
    demo()
