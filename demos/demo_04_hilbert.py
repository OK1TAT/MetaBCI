# -*- coding: utf-8 -*-
"""
表1-4: TimeFrequencyAnalysis.fun_hilbert Hilbert变换（官方 metabci）
运行: python demo_04_hilbert.py
"""
import numpy as np
try:
    from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装, 使用scipy替代")
    from scipy.signal import hilbert as scipy_hilbert

def demo():
    fs = 250
    t = np.arange(0, 2, 1/fs)
    signal = np.sin(2*np.pi*10*t).reshape(1, -1)
    print("信号: 10Hz正弦波, 采样率%dHz" % fs)
    if HAS:
        tfa = TimeFrequencyAnalysis(fs)
        analytic = tfa.fun_hilbert(signal)
        phase = np.angle(analytic[0])
        env = np.abs(analytic[0])
    else:
        analytic = scipy_hilbert(signal[0])
        phase = np.angle(analytic)
        env = np.abs(analytic)
    print("瞬时相位范围: [%.3f, %.3f]" % (phase.min(), phase.max()))
    print("包络均值: %.4f (理论值≈1.0)" % env.mean())
    print("\n✓ Demo完成: TimeFrequencyAnalysis.fun_hilbert")

if __name__ == "__main__":
    demo()
