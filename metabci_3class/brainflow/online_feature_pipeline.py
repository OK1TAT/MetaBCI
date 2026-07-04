# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 在线滑动窗口特征更新管道
提供实时特征计算和更新的完整处理流程

功能特性:
- 滑动窗口数据管理
- 实时PLI/相对功率/模糊熵计算
- 回调函数机制
- 延迟监控

作者: 抑郁症EEG认知障碍识别项目组
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Any
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入brainflow
try:
    from brainflow.data_filter import DataFilter
    from brainflow.enums import WindowOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logger.warning("brainflow未安装，使用numpy实现")


@dataclass
class FeatureSample:
    """特征样本数据结构"""
    timestamp: float                    # 时间戳
    features: np.ndarray                # 特征向量
    feature_names: List[str]            # 特征名称列表
    band_powers: Dict[str, float] = field(default_factory=dict)  # 各频段功率
    pli_matrix: Optional[np.ndarray] = None  # PLI矩阵
    fuzzy_entropy: Optional[float] = None   # 模糊熵
    processing_time_ms: float = 0.0     # 处理耗时
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'features': self.features.tolist(),
            'feature_names': self.feature_names,
            'band_powers': self.band_powers,
            'processing_time_ms': self.processing_time_ms
        }


class OnlineFeaturePipeline:
    """
    在线滑动窗口特征更新管道
    
    实时接收EEG数据，计算认知障碍相关特征（PLI、相对功率、模糊熵等），
    支持滑动窗口更新，通过回调函数将特征推送给分类器。
    
    Features:
    - 可配置滑动窗口长度和步长
    - 实时计算多种EEG特征
    - 回调函数机制（on_feature_update）
    - 延迟监控
    - 线程安全
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        window_length: float = 2.0,
        step_size: float = 0.5,
        n_channels: int = 16,
        frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        compute_pli: bool = True,
        compute_power: bool = True,
        compute_fuzzy_entropy: bool = True,
        n_freq_bins: int = 256,
        fuzzy_entropy_params: Optional[Dict[str, float]] = None
    ):
        """
        初始化在线特征管道
        
        Parameters
        ----------
        sampling_rate : float
            采样率 (Hz)
        window_length : float
            窗口长度 (秒)
        step_size : float
            滑动步长 (秒)
        n_channels : int
            通道数
        frequency_bands : dict, optional
            频段定义，格式: {'band_name': (low_freq, high_freq)}
        compute_pli : bool
            是否计算相位滞后指数
        compute_power : bool
            是否计算频段功率
        compute_fuzzy_entropy : bool
            是否计算模糊熵
        n_freq_bins : int
            FFT频率bins数
        fuzzy_entropy_params : dict, optional
            模糊熵参数: {'m': pattern_length, 'r': tolerance, 'tau': time_delay}
        """
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.step_size = step_size
        self.n_channels = n_channels
        
        # 计算窗口和步长对应的样本数
        self.window_samples = int(window_length * sampling_rate)
        self.step_samples = int(step_size * sampling_rate)
        
        # 默认频段定义
        if frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        else:
            self.frequency_bands = frequency_bands
        
        # 计算配置
        self.compute_pli = compute_pli
        self.compute_power = compute_power
        self.compute_fuzzy_entropy = compute_fuzzy_entropy
        self.n_freq_bins = n_freq_bins
        
        # 模糊熵参数
        if fuzzy_entropy_params is None:
            self.fuzzy_entropy_params = {'m': 2, 'r': 0.2, 'tau': 1}
        else:
            self.fuzzy_entropy_params = fuzzy_entropy_params
        
        # 数据缓冲区
        self._buffer = np.zeros((n_channels, self.window_samples * 2), dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = 0
        self._lock = Lock()
        
        # 特征历史
        self._feature_history: List[FeatureSample] = []
        self._max_history = 100
        
        # 回调函数
        self._update_callbacks: List[Callable[[FeatureSample], None]] = []
        
        # 统计信息
        self._feature_count = 0
        self._total_processing_time = 0.0
        self._min_processing_time = float('inf')
        self._max_processing_time = 0.0
        
        # 构建特征名称
        self._feature_names = self._build_feature_names()
    
    @property
    def feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self._feature_names.copy()
    
    @property
    def stats(self) -> Dict[str, float]:
        """获取处理统计信息"""
        return {
            'feature_count': self._feature_count,
            'avg_processing_time_ms': self._total_processing_time / max(1, self._feature_count),
            'min_processing_time_ms': self._min_processing_time,
            'max_processing_time_ms': self._max_processing_time,
            'buffer_fill_ratio': self._buffer_filled / self.window_samples
        }
    
    def _build_feature_names(self) -> List[str]:
        """构建特征名称列表"""
        names = []
        
        # 频段功率特征
        if self.compute_power:
            for band in self.frequency_bands.keys():
                names.extend([f'{band}_mean', f'{band}_std'])
        
        # 相对功率比
        if self.compute_power:
            names.extend([
                'theta_beta_ratio',
                'theta_alpha_ratio', 
                'alpha_beta_ratio',
                'delta_theta_ratio'
            ])
        
        # PLI特征（选取部分通道对）
        if self.compute_pli:
            # 选取8个代表性的通道对
            channel_pairs = [
                (0, 4), (1, 5),   # 前-中
                (4, 8), (5, 9),   # 中-后
                (0, 8), (1, 9),   # 前-后
                (2, 6), (3, 7),   # 左-右中线
            ]
            for i, j in channel_pairs:
                if i < self.n_channels and j < self.n_channels:
                    names.append(f'pli_ch{i}_ch{j}')
        
        # 模糊熵（全局和主要频段）
        if self.compute_fuzzy_entropy:
            names.append('fuzzy_entropy_global')
            for band in ['theta', 'alpha']:
                if band in self.frequency_bands:
                    names.append(f'fuzzy_entropy_{band}')
        
        return names
    
    def push_data(self, data: np.ndarray) -> bool:
        """
        推送新数据到管道
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据，shape: (n_channels, n_samples)
            
        Returns
        -------
        feature_ready : bool
            是否有足够的窗口数据可计算特征
        """
        with self._lock:
            n_new = data.shape[1]
            buffer_len = self._buffer.shape[1]
            
            # 循环缓冲区写入
            for i in range(n_new):
                idx = self._buffer_pos % buffer_len
                self._buffer[:, idx] = data[:, i]
                self._buffer_pos += 1
            
            self._buffer_filled = min(self._buffer_filled + n_new, buffer_len)
            
            # 检查是否需要计算特征
            samples_since_last = self._buffer_pos - (len(self._feature_history) * self.step_samples)
            return samples_since_last >= self.step_samples
    
    def process_window(self) -> Optional[FeatureSample]:
        """
        处理当前窗口数据，计算特征
        
        Returns
        -------
        feature_sample : FeatureSample or None
            计算得到的特征样本
        """
        if self._buffer_filled < self.window_samples:
            return None
        
        start_time = time.time()
        
        with self._lock:
            # 提取当前窗口数据
            buffer_len = self._buffer.shape[1]
            window_data = np.zeros((self.n_channels, self.window_samples), dtype=np.float32)
            
            for i in range(self.window_samples):
                idx = (self._buffer_pos - self.window_samples + i) % buffer_len
                window_data[:, i] = self._buffer[:, idx]
        
        # 计算特征
        features = []
        band_powers = {}
        pli_matrix = None
        fuzzy_entropy = None
        
        # 1. 频段功率
        if self.compute_power:
            band_powers = self._compute_band_powers(window_data)
            for band in self.frequency_bands.keys():
                if band in band_powers:
                    features.extend([np.mean(band_powers[band]), np.std(band_powers[band])])
                else:
                    features.extend([0.0, 0.0])
            
            # 功率比
            theta = band_powers.get('theta', [0])[0] if 'theta' in band_powers else 0
            alpha = band_powers.get('alpha', [0])[0] if 'alpha' in band_powers else 0
            beta = band_powers.get('beta', [0])[0] if 'beta' in band_powers else 0
            delta = band_powers.get('delta', [0])[0] if 'delta' in band_powers else 0
            
            theta_lin = 10 ** (theta / 10) if theta > -100 else 0
            alpha_lin = 10 ** (alpha / 10) if alpha > -100 else 0
            beta_lin = 10 ** (beta / 10) if beta > -100 else 0
            delta_lin = 10 ** (delta / 10) if delta > -100 else 0
            
            features.extend([
                theta_lin / (beta_lin + 1e-10),
                theta_lin / (alpha_lin + 1e-10),
                alpha_lin / (beta_lin + 1e-10),
                delta_lin / (theta_lin + 1e-10)
            ])
        
        # 2. PLI
        if self.compute_pli:
            pli_matrix = self._compute_pli_matrix(window_data)
            # 提取通道对的PLI
            channel_pairs = [
                (0, 4), (1, 5), (4, 8), (5, 9),
                (0, 8), (1, 9), (2, 6), (3, 7)
            ]
            for i, j in channel_pairs:
                if i < self.n_channels and j < self.n_channels:
                    features.append(pli_matrix[i, j])
                else:
                    features.append(0.0)
        
        # 3. 模糊熵
        if self.compute_fuzzy_entropy:
            global_fe = self._compute_fuzzy_entropy(window_data.mean(axis=0))
            features.append(global_fe)
            fuzzy_entropy = global_fe
            
            # 主要频段模糊熵
            for band in ['theta', 'alpha']:
                if band in self.frequency_bands:
                    low, high = self.frequency_bands[band]
                    band_data = self._bandpass_filter(window_data, low, high)
                    band_fe = self._compute_fuzzy_entropy(band_data.mean(axis=0))
                    features.append(band_fe)
        
        # 确保特征数量一致
        features = np.array(features)[:len(self._feature_names)]
        if len(features) < len(self._feature_names):
            features = np.pad(features, (0, len(self._feature_names) - len(features)))
        
        processing_time = (time.time() - start_time) * 1000
        
        # 创建特征样本
        sample = FeatureSample(
            timestamp=time.time(),
            features=features,
            feature_names=self._feature_names.copy(),
            band_powers={k: np.mean(v) for k, v in band_powers.items()},
            pli_matrix=pli_matrix,
            fuzzy_entropy=fuzzy_entropy,
            processing_time_ms=processing_time
        )
        
        # 更新历史
        self._feature_history.append(sample)
        if len(self._feature_history) > self._max_history:
            self._feature_history.pop(0)
        
        # 更新统计
        self._feature_count += 1
        self._total_processing_time += processing_time
        self._min_processing_time = min(self._min_processing_time, processing_time)
        self._max_processing_time = max(self._max_processing_time, processing_time)
        
        # 触发回调
        for callback in self._update_callbacks:
            try:
                callback(sample)
            except Exception as e:
                logger.error(f"特征回调执行出错: {e}")
        
        return sample
    
    def _compute_band_powers(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算各频段功率
        
        Parameters
        ----------
        data : np.ndarray
            数据，shape: (n_channels, n_samples)
            
        Returns
        -------
        band_powers : dict
            各频段功率，格式: {'band_name': array of powers per channel}
        """
        band_powers = {}
        
        nfft = min(self.n_freq_bins, data.shape[1])
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_power_list = []
            
            for ch in range(data.shape[0]):
                try:
                    # 计算PSD
                    freqs, psd = self._welch_psd(data[ch], nfft)
                    
                    # 提取频段功率
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.mean(psd[band_mask])
                    
                    # 转换为dB
                    band_power_db = 10 * np.log10(band_power + 1e-10)
                    
                except Exception:
                    band_power_db = -100
                
                band_power_list.append(band_power_db)
            
            band_powers[band_name] = np.array(band_power_list)
        
        return band_powers
    
    def _welch_psd(self, signal_data: np.ndarray, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
        """计算Welch功率谱密度"""
        from scipy.signal import welch
        
        freqs, psd = welch(
            signal_data,
            fs=self.sampling_rate,
            nperseg=min(nfft, len(signal_data)),
            noverlap=nfft // 2
        )
        
        return freqs, psd
    
    def _bandpass_filter(self, data: np.ndarray, low: float, high: float) -> np.ndarray:
        """带通滤波"""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sampling_rate / 2
        low_norm = low / nyquist
        high_norm = min(high, nyquist - 0.1) / nyquist
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        filtered = filtfilt(b, a, data, axis=1)
        
        return filtered
    
    def _compute_pli_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        计算相位滞后指数(PLI)矩阵
        
        Parameters
        ----------
        data : np.ndarray
            数据，shape: (n_channels, n_samples)
            
        Returns
        -------
        pli_matrix : np.ndarray
            PLI矩阵，shape: (n_channels, n_channels)
        """
        from scipy.signal import hilbert
        
        n_channels = data.shape[0]
        pli_matrix = np.zeros((n_channels, n_channels))
        
        # 提取alpha频段进行PLI计算
        alpha_low, alpha_high = self.frequency_bands.get('alpha', (8, 13))
        filtered_data = self._bandpass_filter(data, alpha_low, alpha_high)
        
        # 计算每个通道的解析信号
        for i in range(n_channels):
            analytic_signal = hilbert(filtered_data[i])
            phase_i = np.angle(analytic_signal)
            
            for j in range(i + 1, n_channels):
                analytic_signal = hilbert(filtered_data[j])
                phase_j = np.angle(analytic_signal)
                
                # 计算相位差
                phase_diff = phase_j - phase_i
                
                # 计算PLI
                pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                
                pli_matrix[i, j] = pli
                pli_matrix[j, i] = pli
        
        return pli_matrix
    
    def _compute_fuzzy_entropy(self, signal_data: np.ndarray) -> float:
        """
        计算模糊熵 - 向量化加速版
        
        Parameters
        ----------
        signal_data : np.ndarray
            一维信号
            
        Returns
        -------
        fuzzy_ent : float
            模糊熵值
        """
        m = self.fuzzy_entropy_params['m']  # 模式长度
        r = self.fuzzy_entropy_params['r']  # 容忍度
        tau = self.fuzzy_entropy_params['tau']  # 时间延迟
        
        n = len(signal_data)
        
        if n < (m + 1) * tau + 1:
            return 0.0
        
        # z-score标准化，确保模糊熵与信号幅度无关
        std = np.std(signal_data)
        if std == 0:
            return 0.0
        signal_data = (signal_data - np.mean(signal_data)) / std
        r_std = r  # 标准化后 r_std = r
        
        # 降采样：信号过长时均匀抽取（最大150点）
        max_samples = 150
        if n > max_samples:
            indices = np.linspace(0, n - 1, max_samples, dtype=int)
            signal_data = signal_data[indices]
            n = max_samples
        
        def get_vectors(pattern_length):
            """构建嵌入向量并去均值"""
            if tau == 1:
                x = np.lib.stride_tricks.sliding_window_view(signal_data, pattern_length)
            else:
                vectors = []
                for i in range(n - (pattern_length - 1) * tau):
                    vector = [signal_data[i + k * tau] for k in range(pattern_length)]
                    vectors.append(vector)
                x = np.array(vectors)
            # 去均值
            x = x - np.mean(x, axis=1, keepdims=True)
            return x
        
        def _phi(m_dim):
            x = get_vectors(m_dim)
            n_vec = len(x)
            # 向量化计算切比雪夫距离
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            distances = np.max(np.abs(diff), axis=2)
            # 模糊隶属度
            similarities = np.exp(-(distances ** 2) / r_std)
            np.fill_diagonal(similarities, 0)
            phi = np.sum(similarities) / (n_vec * (n_vec - 1))
            return phi
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m1 == 0 or phi_m == 0:
            return 0.0
        
        fuzzy_ent = -np.log(phi_m1 / phi_m)
        
        return float(fuzzy_ent)
    
    def register_callback(self, callback: Callable[[FeatureSample], None]):
        """
        注册特征更新回调函数
        
        Parameters
        ----------
        callback : Callable
            回调函数，签名: callback(feature_sample: FeatureSample)
        """
        if callback not in self._update_callbacks:
            self._update_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[FeatureSample], None]):
        """注销回调函数"""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def get_feature_history(self, n: int = 10) -> List[FeatureSample]:
        """
        获取最近的特征历史
        
        Parameters
        ----------
        n : int
            获取数量
            
        Returns
        -------
        history : List[FeatureSample]
        """
        return self._feature_history[-n:]
    
    def reset(self):
        """重置管道状态"""
        with self._lock:
            self._buffer.fill(0)
            self._buffer_pos = 0
            self._buffer_filled = 0
            self._feature_history.clear()
            self._feature_count = 0
            self._total_processing_time = 0.0
            self._min_processing_time = float('inf')
            self._max_processing_time = 0.0


# ============================================================================
# 便捷函数
# ============================================================================

def create_pipeline(**kwargs) -> OnlineFeaturePipeline:
    """
    创建在线特征管道的便捷函数
    
    Parameters
    ----------
    **kwargs
        传递给OnlineFeaturePipeline的参数
        
    Returns
    -------
    pipeline : OnlineFeaturePipeline
    """
    return OnlineFeaturePipeline(**kwargs)


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MetaBCI Brainflow - 在线特征更新管道演示")
    print("=" * 60)
    
    import time
    
    # 创建管道
    pipeline = OnlineFeaturePipeline(
        sampling_rate=500.0,
        window_length=2.0,
        step_size=0.5,
        n_channels=16
    )
    
    # 注册回调
    def on_feature_update(sample):
        print(f"特征更新: 处理时间={sample.processing_time_ms:.1f}ms, "
              f"功率[alpha]={sample.band_powers.get('alpha', 0):.2f}dB, "
              f"模糊熵={sample.fuzzy_entropy:.4f}")
    
    pipeline.register_callback(on_feature_update)
    
    # 模拟数据流
    print("\n模拟10秒数据流...")
    duration = 10
    n_samples_per_step = 500  # 0.5秒数据
    
    for t in range(int(duration / 0.5)):
        # 生成模拟EEG数据
        eeg_data = np.random.randn(16, n_samples_per_step) * 10
        # 添加一些周期性成分
        eeg_data += 5 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, n_samples_per_step))
        
        # 推送到管道
        ready = pipeline.push_data(eeg_data)
        
        # 如果需要处理窗口
        if ready:
            sample = pipeline.process_window()
            if sample:
                print(f"  [{t*0.5:.1f}s] 窗口处理完成")
        
        time.sleep(0.1)
    
    # 打印统计
    stats = pipeline.stats
    print("\n处理统计:")
    print(f"  平均处理时间: {stats['avg_processing_time_ms']:.2f}ms")
    print(f"  最小处理时间: {stats['min_processing_time_ms']:.2f}ms")
    print(f"  最大处理时间: {stats['max_processing_time_ms']:.2f}ms")
    print(f"  总特征数量: {stats['feature_count']}")
    
    print("\n演示完成!")
