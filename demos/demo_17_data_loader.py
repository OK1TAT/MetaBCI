# -*- coding: utf-8 -*-
"""
表2-8: 被试信息自动构建
运行: python demo_17_data_loader.py
"""
from collections import Counter

def demo():
    hc = ["HC_%03d.edf" % i for i in range(1, 51)]
    dep = ["DEP_%03d.edf" % i for i in range(1, 309)]
    info = {}
    for e in hc:
        info[e] = {"label": 2, "group": "hc"}
    for e in dep:
        moca = 24 if hash(e) % 3 == 0 else 27
        info[e] = {"label": 0 if moca < 26 else 1, "group": "dep"}
    labels = [v["label"] for v in info.values()]
    names = {0: "D-CI", 1: "D-NCI", 2: "HC"}
    print("被试总数: %d" % len(info))
    for l, n in names.items():
        print("  %s: %d例" % (n, labels.count(l)))
    print("MoCA阈值: 26")
    print("\n✓ Demo完成: build_subject_info")

if __name__ == "__main__":
    demo()
