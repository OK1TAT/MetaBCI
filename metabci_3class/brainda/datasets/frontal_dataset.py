# -*- coding: utf-8 -*-
"""
前额六导联 EEG 数据集与特征提取 - 基于 MetaBCI

Brainda 组件:
  1. BaseDataset         - 数据加载基类
  2. upper_ch_names      - 通道名标准化
  3. generate_filterbank - 多频段滤波器组
  4. TimeFrequencyAnalysis.fun_hilbert - Hilbert 变换求相位

特征: PLI(75) + RP(30) + FE(30) = 135 维
"""
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt, resample
from numpy.lib.stride_tricks import as_strided
import mne

# MetaBCI 组件
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.utils.channels import upper_ch_names
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis

from metabci_3class.config import (
    PASSBANDS, STOPBANDS, FILTERBANK_ORDER, FILTERBANK_RP,
    FUZZY_M, FUZZY_R, FUZZY_N,
    FRONTAL_CHANNELS, RESAMPLE_FREQ
)


def clean_channel_names(raw):
    """
    清洗 EDF 通道名，统一为标准导联名。

    处理规则:
        "EEG FP1-REF" → "FP1"
        "EEG F7-REF"  → "F7"
        "EEG CZ-REF"  → "CZ"
        "POL E"       → "E"        (非EEG通道，保留但后续pick会过滤)
        "POL $A1"     → "$A1"

    适用于 NeuroWorkbench / Nihon Kohden 等设备导出的 EDF。
    """
    import re

    rename_map = {}
    for ch_name in raw.info["ch_names"]:
        new_name = ch_name

        # 去掉 "EEG " 前缀
        if new_name.startswith("EEG "):
            new_name = new_name[4:]

        # 去掉 "-REF"/"-Ref" 后缀（大小写不敏感）
        if new_name.upper().endswith("-REF"):
            new_name = new_name[:-4]

        # 去掉 "POL "/"POL " 前缀（大小写不敏感）
        if new_name.upper().startswith("POL "):
            new_name = new_name[4:]

        # 去掉多余空格
        new_name = new_name.strip()

        if new_name != ch_name:
            rename_map[ch_name] = new_name

    if rename_map:
        raw = raw.rename_channels(rename_map)

    return raw


class FrontalEEGDataset(BaseDataset):
    """
    前额六导联静息态 EEG 数据集

    继承 MetaBCI BaseDataset，实现 EDF 加载与通道选择。
    通道: FP1, FP2, F3, F4, F7, F8

    Parameters
    ----------
    subject_info : dict
        由 build_subject_info() 构建的被试信息字典:
        {
            subject_id: {
                'edf_path': str,
                'label': int,     # 0=D-CI, 1=D-NCI, 2=HC
                'moca': float,
                'group': str      # 'HC' 或 'DEP'
            }
        }
    """

    def __init__(self, subject_info: dict):
        subjects = list(subject_info.keys())
        super().__init__(
            dataset_code="FrontalEEG_3Class",
            subjects=subjects,
            events={"rest": (0, (0.0, 60.0))},
            channels=FRONTAL_CHANNELS,
            srate=RESAMPLE_FREQ,
            paradigm="rest",
        )
        self.subject_info = subject_info

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        """返回被试 EDF 文件路径（嵌套列表: sessions × runs）"""
        edf_path = self.subject_info[subject]["edf_path"]
        return [[edf_path]]

    def _get_single_subject_data(
        self,
        subject: Union[str, int],
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> Dict[str, Dict[str, mne.io.Raw]]:
        """
        加载单个被试的 EDF 数据

        Returns
        -------
        dict
            {'session_0': {'run_0': mne.io.Raw}}
            Raw 对象仅包含前额六导联，通道名已大写标准化
        """
        edf_path = self.subject_info[subject]["edf_path"]

        if not os.path.exists(edf_path):
            print(f"  [警告] EDF 文件不存在: {edf_path}")
            return None

        # 读取 EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR",
                                   encoding="latin1")

        # 清洗通道名: "EEG FP1-REF" → "FP1", "EEG F7-REF" → "F7"
        raw = clean_channel_names(raw)

        # 通道名大写标准化 (MetaBCI upper_ch_names)
        raw = upper_ch_names(raw)

        # 选择前额六导联
        available = [ch for ch in FRONTAL_CHANNELS if ch in raw.info["ch_names"]]
        if len(available) < len(FRONTAL_CHANNELS):
            missing = set(FRONTAL_CHANNELS) - set(available)
            print(f"  [警告] 缺少通道: {missing}，可用: {raw.info['ch_names']}")
            if not available:
                return None

        raw.pick_channels(FRONTAL_CHANNELS, ordered=True)

        return {"session_0": {"run_0": raw}}


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

    # 批量计算 RP: 用 scipy.signal.welch 计算 PSD 后积分
    print("  计算 RP (Welch PSD)...")
    rp_all = _compute_rp_batch(filtered_all, X, n_channels, srate=srate)

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

def _compute_rp_batch(filtered_all, raw_data, n_channels, srate=250):
    """
    批量计算 RP (Relative Power)
    使用 scipy.signal.welch 计算功率谱密度(PSD)，再在频段内积分
    比 np.var() 更科学，是标准的EEG功率计算方法

    参考: MetaBCI FrequencyAnalysis 模块设计为事件相关分析，
    需要 data/meta/event 参数，不适合简单功率谱计算，
    故采用 scipy.signal.welch 实现标准PSD计算。
    """
    from scipy.signal import welch

    n_samples, n_bands, _, n_times = filtered_all.shape

    # 频段定义 (从 config 导入)
    # theta(4-8)/alpha_low(8-10)/alpha_high(10-13)/beta_low(13-20)/beta_high(20-30)
    band_ranges = [
        (4, 8),  # theta
        (8, 10),  # alpha_low
        (10, 13),  # alpha_high
        (13, 20),  # beta_low
        (20, 30),  # beta_high
    ]

    # 总功率: 用宽带(1-45Hz)计算
    # welch 返回: freqs (Hz), psd (V^2/Hz)
    _, psd_total = welch(raw_data, fs=srate, nperseg=min(256, n_times), axis=-1)
    freqs_total = np.linspace(0, srate / 2, psd_total.shape[-1])

    # 总功率 = 1-45Hz 积分
    mask_total = (freqs_total >= 1) & (freqs_total <= 45)
    total_power = np.trapezoid(psd_total[..., mask_total], freqs_total[mask_total], axis=-1)  # (n_samples, n_channels)

    rp_all = np.zeros((n_samples, n_bands * n_channels))

    for b in range(n_bands):
        # 对每个频段的滤波数据计算PSD
        band_data = filtered_all[:, b, :, :]  # (n_samples, n_channels, n_times)
        freqs_band, psd_band = welch(band_data, fs=srate, nperseg=min(256, n_times), axis=-1)

        # 在该频段内积分得到频段功率
        f_low, f_high = band_ranges[b]
        mask_band = (freqs_band >= f_low) & (freqs_band <= f_high)
        band_power = np.trapezoid(psd_band[..., mask_band], freqs_band[mask_band], axis=-1)  # (n_samples, n_channels)

        # RP = 频段功率 / 总功率
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


