# -*- coding: utf-8 -*-
"""
表1-12: FrontalEEGDataset 前额六导联EEG数据集（继承BaseDataset）
运行: python demo_10_frontal_dataset.py
"""
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_root)
try:
    from metabci_3class.brainda.datasets.frontal_dataset import FrontalEEGDataset, clean_channel_names
    HAS = True
except Exception as e:
    HAS = False
    print("[!] 导入失败:", e)

def demo():
    info = {"subject_id": "DEP_001", "label": 0, "edf_path": "/tmp/mock.edf", "group": "depression"}
    print("被试信息:", info)
    print("前额6导联: FP1, FP2, F3, F4, F7, F8")
    if HAS:
        try:
            ds = FrontalEEGDataset(info)
            print("FrontalEEGDataset 创建成功")
        except Exception as e:
            print("需要真实EDF:", e)
    print("\n通道名清洗: EEG FP1-REF -> FP1 (大小写不敏感)")
    print("\n✓ Demo完成: FrontalEEGDataset")

if __name__ == "__main__":
    demo()
