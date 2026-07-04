# -*- coding: utf-8 -*-
"""
表1-1: BaseDataset 基类用法（官方 metabci）
功能: MetaBCI BaseDataset 是所有数据集的基类，定义数据加载标准接口。
运行: python demo_01_basedataset.py
"""
import numpy as np

try:
    from metabci.brainda.datasets.base import BaseDataset
    HAS_METABCI = True
except ImportError:
    HAS_METABCI = False
    print("[!] metabci 未安装，请运行: pip install metabci")

def demo():
    if HAS_METABCI:
        class DummyDataset(BaseDataset):
            def __init__(self):
                super().__init__(
                    dataset_code="DummyEEG",
                    subjects=["sub1", "sub2"],
                    events={"rest": (0, (0.0, 10.0))},
                    channels=["FP1", "FP2", "F3", "F4", "F7", "F8"],
                    srate=250, paradigm="rest",
                )
            def data_path(self, subject, path=None, force_update=False,
                          update_path=None, proxies=None, verbose=None):
                return [["/tmp/" + subject + "_rest.edf"]]
            def _get_single_subject_data(self, subject, verbose=None):
                return None
        ds = DummyDataset()
        print("数据集代码:", ds.dataset_code)
        print("被试列表:", ds.subjects)
        print("通道列表:", ds.channels)
        print("采样率:", ds.srate)
    else:
        print("BaseDataset 构造参数:")
        print("  dataset_code, subjects, events, channels, srate, paradigm")
        print("子类必须实现: data_path(), _get_single_subject_data()")
    print("\n✓ Demo完成: BaseDataset 基类用法")

if __name__ == "__main__":
    demo()
