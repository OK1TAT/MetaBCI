# -*- coding: utf-8 -*-
"""
静息态范式 - 离线数据处理

Brainda 功能使用:
    1. BaseDataset 的数据加载流程
    2. Hook 系统设计 (raw_hook / data_hook)
    3. generate_filterbank 预处理滤波
"""

import mne
import numpy as np
import pandas as pd

from metabci_3class.config import RESAMPLE_FREQ


class RestingStateParadigm:
    """
    静息态范式处理器

    负责:
        - 遍历被试加载数据
        - 执行预处理（带通滤波、重采样、重参考）
        - 拼接 X, y, meta
    """

    def __init__(self, dataset, srate=RESAMPLE_FREQ):
        """
        Args:
            dataset: FrontalEEGDataset 实例
            srate: 采样率
        """
        self.dataset = dataset
        self.srate = srate

    def raw_hook(self, raw):
        """
        原始数据预处理 Hook

        参考 MetaBCI 的 Hook 系统设计:
            1. 带通滤波 (1-40 Hz)
            2. 重采样
            3. 重参考 (平均参考)

        Args:
            raw: mne.io.Raw 对象

        Returns:
            mne.io.Raw: 预处理后的数据
        """
        # 带通滤波
        raw.filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')

        # 重采样
        raw.resample(self.srate)

        # 重参考（平均参考）
        raw.set_eeg_reference(ref_channels='average', verbose='ERROR')

        return raw

    def data_hook(self, X, y, meta):
        """
        数据矩阵层 Hook

        Args:
            X: ndarray, shape (n_samples, n_channels, n_times)
            y: ndarray, shape (n_samples,)
            meta: DataFrame

        Returns:
            tuple: (X, y, meta)
        """
        # 去除 NaN
        if X is not None:
            valid_mask = ~np.isnan(X).any(axis=(1, 2))
            X = X[valid_mask]
            y = y[valid_mask]
            if meta is not None:
                meta = meta.iloc[valid_mask].reset_index(drop=True)

        return X, y, meta

    def get_data(self, subjects=None):
        """
        获取所有被试数据

        Args:
            subjects: 被试列表，None 表示全部

        Returns:
            X: ndarray, shape (n_samples, n_channels, n_times)
            y: ndarray, shape (n_samples,)
            meta: DataFrame
        """
        if subjects is None:
            subjects = self.dataset.subjects

        data_list = []
        label_list = []
        meta_list = []

        for i, subject in enumerate(subjects):
            print(f"\n[{i+1}/{len(subjects)}] 处理被试: {subject}")

            # 加载数据
            data_dict = self.dataset._get_single_subject_data(subject)
            if data_dict is None:
                continue

            raw = data_dict['session_0']['run_0']

            # 预处理
            raw = self.raw_hook(raw)

            # 提取数据
            X_sub = raw.get_data()  # (n_channels, n_times)

            # 分段（取前 60 秒，不足则跳过）
            n_samples = self.srate * 60
            if X_sub.shape[1] < n_samples:
                print(f"  跳过: 数据长度不足 60 秒 ({X_sub.shape[1]/self.srate:.1f}s)")
                continue

            X_sub = X_sub[:, :n_samples]

            # 添加维度 (n_channels, n_times) → (1, n_channels, n_times)
            X_sub = X_sub[np.newaxis, :, :]

            # 获取标签
            y_sub = self.dataset.subject_info[subject]['label']

            # 元信息
            meta_sub = pd.DataFrame([{
                'subject': subject,
                'label': y_sub,
                'moca': self.dataset.subject_info[subject]['moca'],
                'group': self.dataset.subject_info[subject]['group']
            }])

            data_list.append(X_sub)
            label_list.append(y_sub)
            meta_list.append(meta_sub)

        # 拼接
        if not data_list:
            return None, None, None

        X = np.concatenate(data_list, axis=0)
        y = np.array(label_list)
        meta = pd.concat(meta_list, ignore_index=True)

        # 数据 Hook
        X, y, meta = self.data_hook(X, y, meta)

        return X, y, meta
