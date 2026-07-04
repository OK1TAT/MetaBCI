# -*- coding: utf-8 -*-
"""
前额六导联特征提取 - 基于 MetaBCI 优化版
Brainda 组件:
  1. generate_filterbank - 多频段滤波器组
  2. TimeFrequencyAnalysis.fun_hilbert - Hilbert 变换求相位
特征: PLI(75) + RP(30) + FE(30) = 135 维
"""
from typing import Any

import numpy as np
from scipy.signal import sosfiltfilt, resample
from numpy.lib.stride_tricks import as_strided

# MetaBCI 组件
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis

from metabci_3class.config import (
    PASSBANDS, STOPBANDS, FILTERBANK_ORDER, FILTERBANK_RP,
    FUZZY_M, FUZZY_R, FUZZY_N
)


def extract_frontal_features(X, srate=250):
    """提取前额六导联特征 (PLI+RP+FE = 135维)"""
    n_samples, n_channels, n_times = X.shape
    n_bands = len(PASSBANDS)

    # MetaBCI: 生成滤波器组
    sos = generate_filterbank(
        passbands=PASSBANDS, stopbands=STOPBANDS,
        srate=srate, order=FILTERBANK_ORDER, rp=FILTERBANK_RP
    )

    # MetaBCI: 创建时频分析实例（需要 fs）
    tfa = TimeFrequencyAnalysis(srate)

    # 预计算所有样本所有频段的滤波数据
    print(f"  预计算滤波数据 ({n_samples} 样本 × {n_bands} 频段)...")
    filtered_all = np.zeros((n_samples, n_bands, n_channels, n_times))
    for i in range(n_samples):
        for b in range(n_bands):
            filtered_all[i, b] = sosfiltfilt(sos[b], X[i], axis=-1)
    print("  滤波完成")

    # 批量计算 PLI: 用 MetaBCI 的 hilbert
    print("  计算 PLI...")
    pli_all = _compute_pli_batch(filtered_all, tfa, n_channels)

    # 批量计算 RP: 直接 numpy
    print("  计算 RP...")
    rp_all = _compute_rp_batch(filtered_all, X, n_channels)

    # 批量计算 FE: stride_tricks + broadcasting
    print("  计算 FE...")
    fe_all = _compute_fe_batch(filtered_all, n_channels, orig_srate=srate)

    # 拼接
    features = np.concatenate([pli_all, rp_all, fe_all], axis=1)

    # 特征名
    ch_names = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8']
    band_names = ['theta', 'alpha_low', 'alpha_high', 'beta_low', 'beta_high']
    feature_names = []
    for b in range(n_bands):
        for c1 in range(n_channels):
            for c2 in range(c1 + 1, n_channels):
                feature_names.append(f"PLI_{band_names[b]}_{ch_names[c1]}_{ch_names[c2]}")
    for c in range(n_channels):
        for b in range(n_bands):
            feature_names.append(f"RP_{band_names[b]}_{ch_names[c]}")
    for c in range(n_channels):
        for b in range(n_bands):
            feature_names.append(f"FE_{band_names[b]}_{ch_names[c]}")

    return features, feature_names


# ========================= PLI (75维) =========================
def _compute_pli_batch(filtered_all, tfa, n_channels):
    """批量计算 PLI，逐通道调用 MetaBCI fun_hilbert"""
    n_samples, n_bands, _, n_times = filtered_all.shape
    n_pairs = n_channels * (n_channels - 1) // 2
    pli_all = np.zeros((n_samples, n_bands * n_pairs))

    pairs = [(c1, c2) for c1 in range(n_channels) for c2 in range(c1 + 1, n_channels)]

    for b in range(n_bands):
        for i in range(n_samples):
            sample_data = filtered_all[i, b, :, :]  # (6, n_times)

            # 逐通道调用 hilbert
            phases = np.zeros_like(sample_data)
            for c in range(n_channels):
                ch_data = sample_data[c:c + 1, :]  # (1, n_times)
                analytic = tfa.fun_hilbert(ch_data)
                phases[c] = np.angle(analytic[0])  # 取第一个通道

            offset = b * n_pairs
            for p_idx, (c1, c2) in enumerate(pairs):
                phase_diff = phases[c1] - phases[c2]
                pli_all[i, offset + p_idx] = np.abs(np.mean(np.cos(phase_diff)))

    return pli_all


# ========================= RP (30维) =========================

def _compute_rp_batch(filtered_all, raw_data, n_channels):
    """批量计算 RP，直接 numpy（FrequencyAnalysis 不适合此场景）"""
    n_samples, n_bands, _, n_times = filtered_all.shape

    # 总功率
    total_power = np.var(raw_data, axis=-1)  # (n_samples, n_channels)

    rp_all = np.zeros((n_samples, n_bands * n_channels))
    for b in range(n_bands):
        band_power = np.var(filtered_all[:, b, :, :], axis=-1)  # (n_samples, n_channels)
        rp = band_power / (total_power + 1e-10)
        offset = b * n_channels
        rp_all[:, offset:offset + n_channels] = rp

    return rp_all


# ========================= FE (30维) =========================

def _compute_fe_batch(filtered_all, n_channels, orig_srate=250, target_srate=50):
    """批量计算模糊熵，逐样本处理避免内存爆炸"""
    n_samples, n_bands, _, n_times = filtered_all.shape
    fe_all = np.zeros((n_samples, n_bands * n_channels))

    down_ratio = orig_srate / target_srate
    n_down = int(n_times / down_ratio)

    total = n_samples * n_bands * n_channels
    count = 0

    for b in range(n_bands):
        for c in range(n_channels):
            # 下采样：所有样本一起
            data = resample(filtered_all[:, b, c, :], n_down, axis=-1)  # (n_samples, n_down)

            # 逐样本计算模糊熵
            for i in range(n_samples):
                fe_all[i, b * n_channels + c] = _fuzzy_entropy_single(data[i])

            count += n_samples
            print(f"    FE 进度: {count}/{total}")

    return fe_all


def _fuzzy_entropy_single(x, m=FUZZY_M, r_ratio=FUZZY_R, n_exp=FUZZY_N):
    """
    单个样本的模糊熵，stride_tricks 向量化
    x: 1D array, shape (N,)
    """
    N = len(x)
    r = r_ratio * np.std(x)
    r = max(r, 1e-10)

    def build_templates(seq, dim):
        n_tmpl = N - dim
        # stride_tricks 构造模板矩阵
        shape = (n_tmpl, dim)
        strides = (seq.strides[0], seq.strides[0])
        templates = as_strided(seq, shape=shape, strides=strides)
        templates = templates - templates.mean(axis=-1, keepdims=True)
        return templates.copy()

    templates_m = build_templates(x, m)  # (Nm, m)
    templates_m1 = build_templates(x, m + 1)  # (Nm1, m+1)

    def compute_C(templates, r, n_exp):
        n_t, dim = templates.shape
        # 两两距离 (n_t, n_t, dim)
        A = templates[:, np.newaxis, :]  # (n_t, 1, dim)
        B = templates[np.newaxis, :, :]  # (1, n_t, dim)
        dists = np.max(np.abs(A - B), axis=-1)  # (n_t, n_t)

        # 模糊隶属
        membership = np.exp(-((dists / r) ** n_exp))

        # 排除自身
        mask = ~np.eye(n_t, dtype=bool)
        C = np.sum(membership * mask) / (n_t * (n_t - 1))
        return C

    C_m = compute_C(templates_m, r, n_exp)
    C_m1 = compute_C(templates_m1, r, n_exp)

    if C_m == 0 or C_m1 == 0:
        return 0.0

    return -np.log(C_m1 / C_m + 1e-10)


