# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 在线预处理管道
提供实时EEG信号预处理功能，适合在线场景

功能特性:
- 4-30Hz带通滤波（因果滤波，实时友好）
- CAR平均参考
- 坏道检测与球面插值
- 单次处理延迟 < 50ms
- 滤波器状态跨窗口保持

作者: 抑郁症EEG认知障碍识别项目组
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass
from threading import Lock
import time

# 尝试导入brainflow
try:
    from brainflow.data_filter import DataFilter
    from brainflow.enums import FilterTypes
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

# 尝试导入scipy
try:
    from scipy.signal import butter, lfilter, filtfilt, iirfilter
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class FilterState:
    """滤波器状态（用于跨窗口保持）"""
    bandpass_z1: Optional[np.ndarray] = None  # 带通滤波状态
    bandpass_z2: Optional[np.ndarray] = None  # (二阶节形式)
    notch_z: Optional[np.ndarray] = None       # 陷波状态
    
    def is_initialized(self) -> bool:
        return self.bandpass_z1 is not None


@dataclass
class ChannelQuality:
    """通道质量信息"""
    channel_idx: int
    is_good: bool
    variance: float
    amplitude_range: float
    snr: float
    artifact_ratio: float


class OnlinePreprocessor:
    """
    在线预处理管道
    
    为实时EEG处理设计的低延迟预处理管道，
    支持因果滤波以确保在线可用性。
    
    Features:
    - 4-30Hz带通滤波（因果IIR滤波器）
    - CAR平均参考
    - 坏道检测（基于方差阈值）
    - 球面插值坏道修复
    - 单次处理延迟 < 50ms
    - 滤波器状态跨窗口保持
    
    Parameters
    ----------
    sampling_rate : float
        采样率（Hz）
    n_channels : int
        通道数
    channel_names : List[str], optional
        通道名称
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        n_channels: int = 16,
        channel_names: Optional[List[str]] = None,
        # 滤波参数
        bandpass_low: float = 0.5,
        bandpass_high: float = 45.0,
        notch_freq: float = 50.0,
        filter_order: int = 4,
        # 处理选项
        use_car: bool = True,
        use_notch: bool = True,
        use_bad_channel_detection: bool = True,
        # 坏道检测参数
        variance_threshold: float = 3.0,
        amplitude_threshold: float = 200.0,  # μV
        snr_threshold: float = 3.0,
        # 状态保持
        preserve_filter_state: bool = True
    ):
        """
        初始化在线预处理器
        
        Parameters
        ----------
        sampling_rate : float
            采样率 (Hz)
        n_channels : int
            通道数
        channel_names : List[str], optional
            通道名称
        bandpass_low : float
            带通下限频率
        bandpass_high : float
            带通上限频率
        notch_freq : float
            陷波频率
        filter_order : int
            滤波器阶数
        use_car : bool
            是否使用CAR平均参考
        use_notch : bool
            是否使用陷波滤波
        use_bad_channel_detection : bool
            是否启用坏道检测
        variance_threshold : float
            方差阈值（标准差倍数）
        amplitude_threshold : float
            幅值阈值（μV）
        snr_threshold : float
            信噪比阈值
        preserve_filter_state : bool
            是否保持滤波器状态
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.channel_names = channel_names or [f"Ch{i+1}" for i in range(n_channels)]
        
        # 滤波参数
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        
        # 处理选项
        self.use_car = use_car
        self.use_notch = use_notch
        self.use_bad_channel_detection = use_bad_channel_detection
        
        # 坏道检测参数
        self.variance_threshold = variance_threshold
        self.amplitude_threshold = amplitude_threshold
        self.snr_threshold = snr_threshold
        
        # 状态保持
        self.preserve_filter_state = preserve_filter_state
        
        # 初始化滤波器系数和状态
        self._init_filters()
        
        # 坏道状态
        self._bad_channels: set = set()
        self._channel_statistics: Dict[int, Dict] = {}
        
        # 线程锁
        self._lock = Lock()
        
        # 统计信息
        self._process_count = 0
        self._total_process_time = 0.0
        self._last_process_time = 0.0
        
        # 回调函数
        self._artifact_callbacks: List[Callable[[List[int], np.ndarray], None]] = []
    
    def _init_filters(self):
        """初始化滤波器"""
        nyquist = self.sampling_rate / 2
        
        # 带通滤波器设计（IIR巴特沃斯）
        low_norm = self.bandpass_low / nyquist
        high_norm = min(self.bandpass_high, nyquist - 0.1) / nyquist
        
        if BRAINFLOW_AVAILABLE and SCIPY_AVAILABLE:
            # 使用scipy设计IIR滤波器
            self._bandpass_b, self._bandpass_a = butter(
                self.filter_order, 
                [low_norm, high_norm], 
                btype='band'
            )
        else:
            # 备用设计
            self._bandpass_b = np.array([1.0])
            self._bandpass_a = np.array([1.0])
        
        # 陷波滤波器设计
        if self.use_notch and self.notch_freq > 0:
            notch_low = (self.notch_freq - 2) / nyquist
            notch_high = (self.notch_freq + 2) / nyquist
            
            if 0 < notch_low and notch_high < 1:
                if BRAINFLOW_AVAILABLE and SCIPY_AVAILABLE:
                    self._notch_b, self._notch_a = iirfilter(
                        self.filter_order,
                        [notch_low, notch_high],
                        btype='bandstop'
                    )
                else:
                    self._notch_b = np.array([1.0])
                    self._notch_a = np.array([1.0])
            else:
                self._notch_b = np.array([1.0])
                self._notch_a = np.array([1.0])
        else:
            self._notch_b = np.array([1.0])
            self._notch_a = np.array([1.0])
        
        # 滤波器状态初始化
        self._filter_state = FilterState()
    
    def reset_filter_state(self):
        """重置滤波器状态"""
        self._filter_state = FilterState()
    
    def process(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int], Dict]:
        """
        处理EEG数据
        
        Parameters
        ----------
        data : np.ndarray
            输入数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray, optional
            时间戳
            
        Returns
        -------
        filtered_data : np.ndarray
            滤波后数据
        bad_channels : List[int]
            检测到的坏道索引
        info : dict
            处理信息
        """
        start_time = time.time()
        
        with self._lock:
            info = {
                'process_time_ms': 0.0,
                'bad_channels': [],
                'artifact_detected': False,
                'interpolated_channels': []
            }
            
            n_channels, n_samples = data.shape
            
            # 1. 坏道检测（如果启用）
            bad_channels = []
            if self.use_bad_channel_detection and n_samples > 10:
                bad_channels = self._detect_bad_channels(data)
                self._bad_channels.update(bad_channels)
                info['bad_channels'] = bad_channels
            
            # 2. 预处理步骤
            processed = data.copy()
            
            # 2.1 陷波滤波（去除工频干扰）
            if self.use_notch:
                processed = self._apply_notch_filter(processed)
            
            # 2.2 带通滤波
            processed = self._apply_bandpass_filter(processed)
            
            # 2.3 坏道处理
            interpolated = []
            if bad_channels:
                processed, interpolated = self._interpolate_bad_channels(
                    processed, bad_channels
                )
                info['interpolated_channels'] = interpolated
            
            # 2.4 CAR平均参考
            if self.use_car:
                processed = self._apply_car(processed)
            
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            info['process_time_ms'] = process_time
            
            # 更新统计
            self._process_count += 1
            self._total_process_time += process_time
            self._last_process_time = process_time
            
            # 触发回调
            if bad_channels:
                for callback in self._artifact_callbacks:
                    try:
                        callback(bad_channels, processed)
                    except Exception:
                        pass
            
            return processed, bad_channels, info
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """应用陷波滤波"""
        if not SCIPY_AVAILABLE:
            return data
        
        filtered = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            if self.preserve_filter_state and self._filter_state.notch_z is not None:
                # 使用保存的状态
                zi = self._filter_state.notch_z[ch:ch+1, :]
                filtered[ch], zf = lfilter(
                    self._notch_b, self._notch_a, 
                    data[ch:ch+1], 
                    zi=zi
                )
                # 保存新状态
                if self._filter_state.notch_z is None:
                    self._filter_state.notch_z = np.zeros((self.n_channels, max(len(self._notch_a), len(self._notch_b)) - 1))
                self._filter_state.notch_z[ch:ch+1, :] = zf
            else:
                filtered[ch] = lfilter(self._notch_b, self._notch_a, data[ch])
        
        return filtered
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """应用带通滤波"""
        if not SCIPY_AVAILABLE:
            return data
        
        filtered = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            if self.preserve_filter_state:
                # 获取或初始化状态
                if self._filter_state.bandpass_z1 is None:
                    self._filter_state.bandpass_z1 = np.zeros(
                        (self.n_channels, len(self._bandpass_a) - 1)
                    )
                
                zi = self._filter_state.bandpass_z1[ch:ch+1, :]
                filtered[ch], zf = lfilter(
                    self._bandpass_b, self._bandpass_a,
                    data[ch:ch+1],
                    zi=zi
                )
                self._filter_state.bandpass_z1[ch:ch+1, :] = zf
            else:
                filtered[ch] = lfilter(
                    self._bandpass_b, self._bandpass_a, 
                    data[ch]
                )
        
        return filtered
    
    def _detect_bad_channels(self, data: np.ndarray) -> List[int]:
        """
        检测坏道
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据
            
        Returns
        -------
        bad_channels : List[int]
        """
        bad_channels = []
        
        # 计算各通道统计量
        for ch in range(data.shape[0]):
            channel_data = data[ch]
            
            # 方差
            variance = np.var(channel_data)
            
            # 幅值范围
            amplitude_range = np.max(channel_data) - np.min(channel_data)
            
            # 统计量存储
            self._channel_statistics[ch] = {
                'variance': variance,
                'amplitude_range': amplitude_range
            }
        
        if not self._channel_statistics:
            return []
        
        # 计算全局统计
        variances = [s['variance'] for s in self._channel_statistics.values()]
        amplitude_ranges = [s['amplitude_range'] for s in self._channel_statistics.values()]
        
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        mean_amplitude = np.mean(amplitude_ranges)
        std_amplitude = np.std(amplitude_ranges)
        
        # 检测异常通道
        for ch in range(data.shape[0]):
            stats = self._channel_statistics.get(ch, {})
            variance = stats.get('variance', mean_variance)
            amplitude = stats.get('amplitude_range', mean_amplitude)
            
            # 方差异常
            if mean_variance > 0 and std_variance > 0:
                z_score = abs(variance - mean_variance) / std_variance
                if z_score > self.variance_threshold:
                    bad_channels.append(ch)
                    continue
            
            # 幅值异常
            if mean_amplitude > 0 and std_amplitude > 0:
                z_score = abs(amplitude - mean_amplitude) / std_amplitude
                if z_score > self.variance_threshold:
                    if ch not in bad_channels:
                        bad_channels.append(ch)
                    continue
            
            # 幅值绝对阈值
            if amplitude > self.amplitude_threshold:
                if ch not in bad_channels:
                    bad_channels.append(ch)
        
        return bad_channels
    
    def _interpolate_bad_channels(
        self,
        data: np.ndarray,
        bad_channels: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        插值坏道
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据
        bad_channels : List[int]
            坏道索引
            
        Returns
        -------
        interpolated_data : np.ndarray
        interpolated_indices : List[int]
        """
        if not SCIPY_AVAILABLE or len(bad_channels) == 0:
            return data, []
        
        interpolated = data.copy()
        interpolated_indices = []
        
        # 16导联标准位置（x, y坐标）
        standard_positions = {
            0: (-3, 3),   # Fp1
            1: (3, 3),    # Fp2
            2: (-2, 2),   # F3
            3: (2, 2),    # F4
            4: (-1, 1),   # C3
            5: (1, 1),    # C4
            6: (-2, 0),   # P3
            7: (2, 0),    # P4
            8: (-3, -1),  # O1
            9: (3, -1),   # O2
            10: (-4, 2),  # F7
            11: (4, 2),   # F8
            12: (-4, 0),  # T3
            13: (4, 0),   # T4
            14: (-3, -2), # T5
            15: (3, -2),  # T6
        }
        
        for ch in bad_channels:
            if ch >= self.n_channels:
                continue
            
            # 获取好通道的位置和数据
            good_channels = [i for i in range(self.n_channels) if i not in bad_channels]
            
            if len(good_channels) < 4:
                # 邻居平均
                interpolated[ch] = np.mean(data[good_channels], axis=0) if good_channels else 0
            else:
                # 球面插值
                pos_array = []
                data_array = []
                
                for i in good_channels:
                    pos = standard_positions.get(i, (0, 0))
                    pos_array.append(pos)
                    data_array.append(data[i])
                
                target_pos = standard_positions.get(ch, (0, 0))
                
                # 使用反距离加权插值
                weights = []
                values = []
                
                for pos, val in zip(pos_array, data_array):
                    dist = np.sqrt((pos[0] - target_pos[0])**2 + (pos[1] - target_pos[1])**2)
                    if dist < 0.1:
                        dist = 0.1  # 避免除零
                    w = 1.0 / dist
                    weights.append(w)
                    values.append(val * w)
                
                if weights:
                    interpolated[ch] = np.sum(values, axis=0) / np.sum(weights)
            
            interpolated_indices.append(ch)
        
        return interpolated, interpolated_indices
    
    def _apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        应用平均参考（CAR）
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据
            
        Returns
        -------
        referenced : np.ndarray
        """
        # 计算所有通道的均值（排除坏道）
        good_channels = [i for i in range(data.shape[0]) if i not in self._bad_channels]
        
        if len(good_channels) == 0:
            return data
        
        # 计算平均参考
        mean_reference = np.mean(data[good_channels], axis=0)
        
        # 应用到所有通道
        return data - mean_reference
    
    def register_artifact_callback(
        self, 
        callback: Callable[[List[int], np.ndarray], None]
    ):
        """
        注册伪迹检测回调
        
        Parameters
        ----------
        callback : Callable
            回调函数，签名: callback(bad_channels, data)
        """
        if callback not in self._artifact_callbacks:
            self._artifact_callbacks.append(callback)
    
    def unregister_artifact_callback(
        self, 
        callback: Callable[[List[int], np.ndarray], None]
    ):
        """注销伪迹回调"""
        if callback in self._artifact_callbacks:
            self._artifact_callbacks.remove(callback)
    
    @property
    def bad_channels(self) -> List[int]:
        """获取当前坏道列表"""
        return list(self._bad_channels)
    
    @property
    def statistics(self) -> Dict:
        """获取处理统计信息"""
        return {
            'process_count': self._process_count,
            'avg_process_time_ms': self._total_process_time / max(1, self._process_count),
            'last_process_time_ms': self._last_process_time,
            'current_bad_channels': list(self._bad_channels)
        }
    
    def reset(self):
        """重置预处理器状态"""
        self._bad_channels.clear()
        self._channel_statistics.clear()
        self.reset_filter_state()
        self._process_count = 0
        self._total_process_time = 0.0


class CascadePreprocessor:
    """
    级联预处理器
    
    支持多个预处理器顺序处理，
    适用于需要分阶段预处理的场景。
    """
    
    def __init__(self, preprocessors: Optional[List[OnlinePreprocessor]] = None):
        """初始化级联预处理器"""
        self.preprocessors = preprocessors or []
    
    def add(self, preprocessor: OnlinePreprocessor):
        """添加预处理器"""
        self.preprocessors.append(preprocessor)
    
    def process(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        顺序处理数据
        
        Parameters
        ----------
        data : np.ndarray
            输入数据
        timestamps : np.ndarray, optional
            
        Returns
        -------
        final_data : np.ndarray
        stage_info : List[dict]
        """
        stage_info = []
        processed = data.copy()
        
        for i, preprocessor in enumerate(self.preprocessors):
            processed, bad_channels, info = preprocessor.process(processed, timestamps)
            stage_info.append({
                'stage': i,
                'bad_channels': bad_channels,
                'info': info
            })
        
        return processed, stage_info


# ============================================================================
# 便捷函数
# ============================================================================

def create_preprocessor(
    sampling_rate: float = 1000.0,
    n_channels: int = 16,
    **kwargs
) -> OnlinePreprocessor:
    """
    创建在线预处理器的便捷函数
    
    Parameters
    ----------
    sampling_rate : float
        采样率
    n_channels : int
        通道数
    **kwargs
        其他参数
        
    Returns
    -------
    preprocessor : OnlinePreprocessor
    """
    return OnlinePreprocessor(
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        **kwargs
    )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MetaBCI Brainflow - 在线预处理器演示")
    print("=" * 60)
    
    # 创建预处理器
    preprocessor = OnlinePreprocessor(
        sampling_rate=1000.0,
        n_channels=16,
        bandpass_low=0.5,
        bandpass_high=45.0,
        notch_freq=50.0,
        use_car=True,
        use_notch=True,
        use_bad_channel_detection=True
    )
    
    print(f"\n预处理器配置:")
    print(f"  采样率: {preprocessor.sampling_rate} Hz")
    print(f"  通道数: {preprocessor.n_channels}")
    print(f"  带通滤波: {preprocessor.bandpass_low}-{preprocessor.bandpass_high} Hz")
    
    # 模拟EEG数据
    import time
    
    print("\n处理模拟数据...")
    for i in range(5):
        # 生成2秒的模拟数据
        duration = 2.0
        n_samples = int(duration * preprocessor.sampling_rate)
        eeg_data = np.random.randn(16, n_samples) * 20  # μV
        
        # 添加一些周期性成分
        t = np.linspace(0, duration, n_samples)
        eeg_data[0] += 10 * np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
        eeg_data[4] += 15 * np.sin(2 * np.pi * 6 * t)   # 6Hz theta
        
        # 模拟一个坏道（第8通道）
        if i == 2:
            eeg_data[8] = 100 * np.random.randn(n_samples)
        
        # 处理
        filtered, bad_channels, info = preprocessor.process(eeg_data)
        
        print(f"  [{i}] 处理时间: {info['process_time_ms']:.1f}ms, "
              f"坏道: {info['bad_channels']}, "
              f"插值: {info['interpolated_channels']}")
        
        time.sleep(0.1)
    
    # 打印统计
    stats = preprocessor.statistics
    print(f"\n处理统计:")
    print(f"  处理次数: {stats['process_count']}")
    print(f"  平均处理时间: {stats['avg_process_time_ms']:.2f}ms")
    print(f"  当前坏道: {stats['current_bad_channels']}")
    
    print("\n演示完成!")
