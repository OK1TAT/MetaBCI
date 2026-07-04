# -*- coding: utf-8 -*-
"""
表1-10: NeuroScanPort 并口标签通信（官方 metabci）
运行: python demo_08_neuroscan.py
"""
try:
    from metabci.brainstim.utils import NeuroScanPort
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")

def demo():
    print("NeuroScanPort: 并口发送/接收事件标记")
    if HAS:
        try:
            port = NeuroScanPort(port_addr=None)
            print("创建成功 (模拟模式)")
        except Exception as e:
            print("需要并口硬件:", e)
    print("\n✓ Demo完成: NeuroScanPort")

if __name__ == "__main__":
    demo()
