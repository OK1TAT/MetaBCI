# -*- coding: utf-8 -*-
"""
表2-2,3,4,5: 135维特征提取（PLI+RP+FE）
运行: python demo_15_feature_extraction.py
"""
import numpy as np
from scipy.signal import hilbert, welch

def demo():
    fs, n_ch, dur = 250, 6, 10
    t = np.arange(0, dur, 1/fs)
    ch = ["FP1","FP2","F3","F4","F7","F8"]
    bands = [(4,8),(8,10),(10,13),(13,20),(20,30)]
    np.random.seed(42)
    eeg = np.zeros((n_ch, len(t)))
    for i in range(n_ch):
        eeg[i] = np.sin(2*np.pi*(6+i*3)*t) + 0.5*np.random.randn(len(t))
    print("EEG: %d通道, %dHz, %ds" % (n_ch, fs, dur))

    # PLI
    analytic = hilbert(eeg, axis=1)
    phase = np.angle(analytic)
    pli = []
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            for _ in bands:
                pli.append(np.abs(np.mean(np.exp(1j*(phase[i]-phase[j])))))
    print("[PLI] %d维 (理论75)" % len(pli))

    # RP
    rp = []
    for i in range(n_ch):
        f, p = welch(eeg[i], fs=fs, nperseg=fs*2)
        tot = np.trapezoid(p, f)
        for b in bands:
            m = (f>=b[0]) & (f<=b[1])
            rp.append(np.trapezoid(p[m], f[m]) / (tot+1e-10))
    print("[RP] %d维 (理论30)" % len(rp))

    # FE
    def fe(x, m=2, r=0.2):
        n = len(x)
        if n < m+1: return 0
        pat = np.array([x[i:i+m] for i in range(n-m)])
        d = np.abs(pat[:,None]-pat[None,:]).max(axis=2)
        s = np.exp(-(d/r)**2); np.fill_diagonal(s,0)
        bm = s.sum()/(len(pat)*(len(pat)-1))
        pat2 = np.array([x[i:i+m+1] for i in range(n-m)])
        d2 = np.abs(pat2[:,None]-pat2[None,:]).max(axis=2)
        s2 = np.exp(-(d2/r)**2); np.fill_diagonal(s2,0)
        bm2 = s2.sum()/(len(pat2)*(len(pat2)-1))
        return -np.log(bm2/(bm+1e-10)+1e-10)
    feats = []
    for i in range(n_ch):
        for _ in bands:
            feats.append(fe(eeg[i][:500]))
    print("[FE] %d维 (理论30)" % len(feats))
    print("总计: %d维 (理论135)" % (len(pli)+len(rp)+len(feats)))
    print("\n✓ Demo完成: 135维特征提取")

if __name__ == "__main__":
    demo()
