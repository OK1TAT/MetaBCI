# -*- coding: utf-8 -*-
"""
表1-3: generate_filterbank 多频段Chebyshev I型滤波器组（官方 metabci）
运行: python demo_03_filterbank.py
"""
import numpy as np
try:
    from metabci.brainda.algorithms.decomposition.base import generate_filterbank
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")

def demo():
    passbands = [[4, 8], [8, 10], [10, 13], [13, 20], [20, 30]]
    stopbands = [[2, 10], [6, 12], [8, 15], [11, 22], [18, 32]]
    srate, order, rp = 250, 4, 0.5
    print("频段:", [str(pb[0])+"-"+str(pb[1])+"Hz" for pb in passbands])
    print("采样率:", srate, "Hz, 阶数:", order, ", 纹波:", rp, "dB")
    if HAS:
        fb = generate_filterbank(passbands, stopbands, srate, order, rp)
        print("生成滤波器组:", len(fb), "个")
        t = np.arange(0, 2, 1/srate)
        sig = np.sin(2*np.pi*6*t) + np.sin(2*np.pi*20*t)
        for i, (f, pb) in enumerate(zip(fb, passbands)):
            out = f(sig)
            print("  频段%d-%dHz: 输出功率=%.2f" % (pb[0], pb[1], np.var(out)))
    else:
        print("将生成", len(passbands), "个Chebyshev I型IIR滤波器")
    print("\n✓ Demo完成: generate_filterbank 滤波器组生成")

if __name__ == "__main__":
    demo()
