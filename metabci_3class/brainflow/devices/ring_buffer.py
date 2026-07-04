# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 高效环形缓冲区

为Stroke EEG实时采集系统设计的固定内存环形缓冲区，
支持O(1)时间复杂度的写入和滑动读取操作。

功能特性:
- 固定内存占用，预分配numpy数组
- O(1)写入和滑动读取
- 多通道EEG数据支持
- 线程安全（读写锁）
- 长时间连续监测适用

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
from threading import Lock, RLock
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 缓冲区状态数据类
# ============================================================================

@dataclass
class BufferStatus:
    """
    缓冲区状态快照
    
    Attributes
    ----------
    capacity : int
        总容量（样本数）
    size : int
        当前数据量（样本数）
    write_pos : int
        当前写入位置
    read_pos : int
        最后读取位置
    fill_ratio : float
        填充比例 (0.0 ~ 1.0)
    sample_rate : float
        采样率（Hz）
    duration_seconds : float
        当前数据持续时间（秒）
    last_write_time : float
        最后一次写入时间戳
    last_read_time : float
        最后一次读取时间戳
    """
    capacity: int
    size: int
    write_pos: int
    read_pos: int
    fill_ratio: float
    sample_rate: float
    duration_seconds: float
    last_write_time: float
    last_read_time: float


# ============================================================================
# 高效环形缓冲区
# ============================================================================

class EfficientRingBuffer:
    """
    高效环形缓冲区
    
    为实时EEG数据采集设计的固定大小循环缓冲区。
    内部使用预分配的numpy数组，写入和读取操作均为O(1)时间复杂度。
    支持多通道EEG数据，线程安全。
    
    Parameters
    ----------
    n_channels : int
        通道数量
    max_duration : float
        缓冲区最大持续时间（秒）
    sampling_rate : float
        采样率（Hz）
    dtype : np.dtype
        数据类型，默认 np.float32
    
    Examples
    --------
    >>> buf = EfficientRingBuffer(n_channels=16, max_duration=10.0, sampling_rate=500)
    >>> buf.append(np.random.randn(16, 100))
    >>> data = buf.get_latest(n_samples=250)
    >>> print(data.shape)
    (16, 250)
    """
    
    def __init__(
        self,
        n_channels: int = 16,
        max_duration: float = 30.0,
        sampling_rate: float = 500.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        
        # 缓冲区容量（样本数）
        self.capacity = int(max_duration * sampling_rate)
        
        # 预分配numpy数组
        self._buffer = np.zeros((n_channels, self.capacity), dtype=dtype)
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        
        # 读写位置
        self._write_pos = 0
        self._total_written = 0
        self._last_write_time = 0.0
        self._last_read_time = 0.0
        
        # 线程安全锁
        self._lock = RLock()
        
        logger.debug(
            "环形缓冲区初始化: ch=%d capacity=%d (%.1f秒) sr=%.0fHz",
            n_channels, self.capacity, max_duration, sampling_rate,
        )
    
    @property
    def total_written(self) -> int:
        """累计写入样本数"""
        with self._lock:
            return self._total_written
    
    @property
    def available_samples(self) -> int:
        """当前可用样本数"""
        with self._lock:
            return min(self._total_written, self.capacity)
    
    def append(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> None:
        """
        向缓冲区追加数据
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray, optional
            时间戳数组，shape: (n_samples,)
        """
        if data.ndim != 2:
            raise ValueError(f"data 必须是2维数组，当前维度: {data.ndim}")
        
        n_ch, n_samples = data.shape
        if n_ch != self.n_channels:
            raise ValueError(
                f"通道数不匹配: 期望 {self.n_channels}，实际 {n_ch}"
            )
        
        with self._lock:
            for i in range(n_samples):
                pos = self._write_pos % self.capacity
                self._buffer[:, pos] = data[:, i]
                if timestamps is not None and i < len(timestamps):
                    self._timestamps[pos] = timestamps[i]
                else:
                    self._timestamps[pos] = time.time()
                self._write_pos = (self._write_pos + 1) % self.capacity
            
            self._total_written += n_samples
            self._last_write_time = time.time()
    
    def append_sample(
        self,
        sample: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        向缓冲区追加单个样本
        
        Parameters
        ----------
        sample : np.ndarray
            单样本数据，shape: (n_channels,)
        timestamp : float, optional
            时间戳
        """
        self.append(
            sample.reshape(-1, 1),
            np.array([timestamp or time.time()]),
        )
    
    def get_latest(
        self,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取最新的N个样本
        
        Parameters
        ----------
        n_samples : int, optional
            样本数量，None表示获取所有可用数据
        
        Returns
        -------
        data : np.ndarray
            EEG数据，shape: (n_channels, n_available)
        timestamps : np.ndarray
            时间戳，shape: (n_available,)
        """
        with self._lock:
            available = min(self._total_written, self.capacity)
            
            if available == 0:
                return (
                    np.zeros((self.n_channels, 0), dtype=self.dtype),
                    np.zeros(0, dtype=np.float64),
                )
            
            if n_samples is None or n_samples >= available:
                n_samples = available
            
            # 计算读取起止位置
            end_pos = self._write_pos
            start_pos = (end_pos - n_samples) % self.capacity
            
            if start_pos < end_pos:
                # 数据连续
                data = self._buffer[:, start_pos:end_pos].copy()
                timestamps = self._timestamps[start_pos:end_pos].copy()
            else:
                # 数据回绕
                part1 = self._buffer[:, start_pos:]
                part2 = self._buffer[:, :end_pos]
                data = np.concatenate([part1, part2], axis=1)
                ts1 = self._timestamps[start_pos:]
                ts2 = self._timestamps[:end_pos]
                timestamps = np.concatenate([ts1, ts2])
            
            self._last_read_time = time.time()
            return data, timestamps
    
    def get_data(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        按时间范围获取数据
        
        Parameters
        ----------
        start_time : float, optional
            起始时间戳
        end_time : float, optional
            结束时间戳
        
        Returns
        -------
        data : np.ndarray
            EEG数据
        timestamps : np.ndarray
            时间戳
        """
        all_data, all_ts = self.get_latest()
        
        if all_data.shape[1] == 0:
            return all_data, all_ts
        
        mask = np.ones(all_ts.shape[0], dtype=bool)
        if start_time is not None:
            mask &= all_ts >= start_time
        if end_time is not None:
            mask &= all_ts <= end_time
        
        return all_data[:, mask], all_ts[mask]
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._buffer.fill(0)
            self._timestamps.fill(0)
            self._write_pos = 0
            self._total_written = 0
    
    def status(self) -> BufferStatus:
        """
        获取缓冲区状态快照
        
        Returns
        -------
        status : BufferStatus
            当前缓冲区状态
        """
        with self._lock:
            available = min(self._total_written, self.capacity)
            return BufferStatus(
                capacity=self.capacity,
                size=available,
                write_pos=self._write_pos,
                read_pos=(self._write_pos - available) % self.capacity,
                fill_ratio=available / self.capacity if self.capacity > 0 else 0.0,
                sample_rate=self.sampling_rate,
                duration_seconds=available / self.sampling_rate if self.sampling_rate > 0 else 0.0,
                last_write_time=self._last_write_time,
                last_read_time=self._last_read_time,
            )
    
    def __len__(self) -> int:
        """返回当前可用样本数"""
        return self.available_samples
    
    def __repr__(self) -> str:
        s = self.status()
        return (
            f"EfficientRingBuffer(ch={self.n_channels}, "
            f"capacity={self.capacity}, "
            f"available={s.size}, "
            f"duration={s.duration_seconds:.2f}s)"
        )


# ============================================================================
# 多通道环形缓冲区（按通道独立管理）
# ============================================================================

class MultiChannelRingBuffer:
    """
    多通道独立环形缓冲区
    
    为每个通道维护独立的缓冲区，适用于需要按通道独立操作的场景。
    
    Parameters
    ----------
    n_channels : int
        通道数
    max_samples : int
        每通道最大样本数
    dtype : np.dtype
        数据类型
    """
    
    def __init__(
        self,
        n_channels: int = 16,
        max_samples: int = 5000,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.n_channels = n_channels
        self.max_samples = max_samples
        self._buffers: List[List[float]] = [[] for _ in range(n_channels)]
        self._lock = Lock()
    
    def append(self, sample: np.ndarray) -> None:
        """
        追加一个多样本
        
        Parameters
        ----------
        sample : np.ndarray
            shape: (n_channels, n_samples)
        """
        with self._lock:
            for ch in range(self.n_channels):
                for i in range(sample.shape[1] if sample.ndim > 1 else 1):
                    val = float(sample[ch, i] if sample.ndim > 1 else sample[ch])
                    self._buffers[ch].append(val)
                    if len(self._buffers[ch]) > self.max_samples:
                        self._buffers[ch].pop(0)
    
    def get_channel(self, channel: int) -> np.ndarray:
        """获取指定通道的数据"""
        with self._lock:
            return np.array(self._buffers[channel], dtype=np.float32)
    
    def get_all(self) -> np.ndarray:
        """获取所有通道数据"""
        with self._lock:
            min_len = min(len(b) for b in self._buffers)
            result = np.zeros((self.n_channels, min_len), dtype=np.float32)
            for ch in range(self.n_channels):
                result[ch] = self._buffers[ch][-min_len:]
            return result
    
    def clear(self) -> None:
        """清空所有通道"""
        with self._lock:
            for ch in range(self.n_channels):
                self._buffers[ch].clear()


# ============================================================================
# 便捷函数
# ============================================================================

def create_buffer(
    n_channels: int = 16,
    max_duration: float = 30.0,
    sampling_rate: float = 500.0,
    **kwargs,
) -> EfficientRingBuffer:
    """
    便捷函数：创建环形缓冲区
    
    Parameters
    ----------
    n_channels : int
        通道数
    max_duration : float
        最大持续时间（秒）
    sampling_rate : float
        采样率（Hz）
    **kwargs
        传递给 EfficientRingBuffer 的其他参数
    
    Returns
    -------
    buffer : EfficientRingBuffer
        创建好的缓冲区实例
    """
    return EfficientRingBuffer(
        n_channels=n_channels,
        max_duration=max_duration,
        sampling_rate=sampling_rate,
        **kwargs,
    )
