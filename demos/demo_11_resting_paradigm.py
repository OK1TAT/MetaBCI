# -*- coding: utf-8 -*-
"""
表1-13: RestingStateParadigm 静息态离线范式处理器
运行: python demo_11_resting_paradigm.py
"""
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    from metabci_3class.brainda.paradigms.resting_state import RestingStateParadigm
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    print("RestingStateParadigm: raw_hook/data_hook/get_data")
    if HAS:
        print("导入成功")
    print("\n✓ Demo完成: RestingStateParadigm")

if __name__ == "__main__":
    demo()
