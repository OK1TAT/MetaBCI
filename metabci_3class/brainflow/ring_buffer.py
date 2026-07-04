# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 高效环形缓冲区
提供O(1)读写性能的EEG数据缓冲管理

功能特性:
- 固定内存占用，预分配numpy数组
- O(1)写入和滑动读取
- 多通道EEG数据支持
- 线程安全（读写锁）
- 长时间连续监测适用

作者: 抑郁症EEG认知障碍识别项目组
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
from threading import Lock, RLock
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import time

# 尝试导入MetaBCI模块
try:
    from metabci.brainflow.amplifiers import RingBuffer as MetaBCIRingBuffer
    METABCI_AVAILABLE = True
except ImportError:
    METABCI_AVAILABLE = False


@dataclass
class BufferStatus:
    """缓冲区状态"""
    capacity: int              # 总容量（样本数）
    size: int                  # 当前数据量
    write_pos: int             # 当前写入位置
    read_pos: int              # 最后读取位置
    fill_ratio: float          # 填充比例
    sample_rate: float         # 采样率
    duration_seconds: float    # 数据持续时间
    last_write_time: float     # 最后写入时间
    last_read_time: float      # 最后读取时间


class EfficientRingBuffer:
    """
    高效环形缓冲区
    
    为实时EEG数据采集设计的固定大小循环缓冲区，
    支持O(1)时间复杂度的写入和读取操作。
    
    Features:
    - O(1)写入和滑动读取
    - 内存占用固定，预分配numpy数组
    - 支持多通道EEG数据
    - 线程安全（读写锁）
    - 适用于长时间连续监测
    
    Parameters
    ----------
    n_channels : int
        通道数量
    max_duration : float
        最大持续时间（秒）
    sampling_rate : float
        采样率（Hz）
    dtype : np.dtype
        数据类型
    """
    
    def __init__(
        self,
        n_channels: int = 16,
        max_duration: float = 30.0,
        sampling_rate: float = 1000.0,
        dtype: np.dtype = np.float32
    ):
        """
        初始化高效环形缓冲区
        
        Parameters
        ----------
        n_channels : int
            通道数
        max_duration : float
            缓冲区最大持续时间（秒）
        sampling_rate : float
            采样率（Hz）
        dtype : np.dtype
            数据类型
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        
        # 计算缓冲区大小
        self.capacity = int(max_duration * sampling_rate)
        
        # 预分配数据缓冲区
        self._data = np.zeros((n_channels, self.capacity), dtype=dtype)
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._markers = np.zeros(self.capacity, dtype=np.int32)
        
        # 读写指针
        self._write_idx = 0
        self._read_idx = 0
        self._current_size = 0
        
        # 线程锁
        self._lock = RLock()
        
        # 时间追踪
        self._last_write_time = time.time()
        self._last_read_time = time.time()
        
        # 如果MetaBCI可用，使用其实现
        if METABCI_AVAILABLE:
            try:
                self._meta_buffer = MetaBCIRingBuffer(
                    buffer_len=self.capacity,
                    n_chans=n_channels
                )
            except Exception:
                self._meta_buffer = None
        else:
            self._meta_buffer = None
        
        # 回调函数
        self._write_callbacks: List[Callable[[np.ndarray, int], None]] = []
    
    @property
    def size(self) -> int:
        """当前缓冲区大小（样本数）"""
        return self._current_size
    
    @property
    def is_empty(self) -> bool:
        """缓冲区是否为空"""
        return self._current_size == 0
    
    @property
    def is_full(self) -> bool:
        """缓冲区是否已满"""
        return self._current_size >= self.capacity
    
    @property
    def status(self) -> BufferStatus:
        """获取缓冲区状态"""
        return BufferStatus(
            capacity=self.capacity,
            size=self._current_size,
            write_pos=self._write_idx,
            read_pos=self._read_idx,
            fill_ratio=self._current_size / max(1, self.capacity),
            sample_rate=self.sampling_rate,
            duration_seconds=self._current_size / self.sampling_rate,
            last_write_time=self._last_write_time,
            last_read_time=self._last_read_time
        )
    
    def write(
        self,
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        markers: Optional[np.ndarray] = None
    ) -> int:
        """
        写入数据到缓冲区（O(1)复杂度）
        
        Parameters
        ----------
        data : np.ndarray
            EEG数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray, optional
            时间戳，shape: (n_samples,)
        markers : np.ndarray, optional
            事件标记，shape: (n_samples,)
            
        Returns
        -------
        n_written : int
            实际写入的样本数
        """
        with self._lock:
            n_samples = data.shape[1]
            n_written = 0
            
            for i in range(n_samples):
                # 计算写入位置（循环）
                idx = (self._write_idx + i) % self.capacity
                
                # 写入数据
                self._data[:, idx] = data[:, i]
                
                # 写入时间戳
                if timestamps is not None:
                    self._timestamps[idx] = timestamps[i]
                else:
                    self._timestamps[idx] = time.time()
                
                # 写入标记
                if markers is not None:
                    self._markers[idx] = markers[i]
                
                n_written += 1
                
                # 更新写入位置和大小
                if self._current_size < self.capacity:
                    self._current_size += 1
            
            self._write_idx = (self._write_idx + n_written) % self.capacity
            self._last_write_time = time.time()
            
            # 触发写入回调
            for callback in self._write_callbacks:
                try:
                    callback(data, n_written)
                except Exception:
                    pass
            
            return n_written
    
    def write_single(self, sample: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        写入单个样本（O(1)复杂度）
        
        Parameters
        ----------
        sample : np.ndarray
            单个样本，shape: (n_channels,)
        timestamp : float, optional
            时间戳
            
        Returns
        -------
        success : bool
        """
        with self._lock:
            idx = self._write_idx
            
            # 写入数据
            self._data[:, idx] = sample
            
            # 写入时间戳
            if timestamp is not None:
                self._timestamps[idx] = timestamp
            else:
                self._timestamps[idx] = time.time()
            
            # 更新指针
            self._write_idx = (self._write_idx + 1) % self.capacity
            
            if self._current_size < self.capacity:
                self._current_size += 1
            
            self._last_write_time = time.time()
            
            return True
    
    def read(self, n_samples: int, start_pos: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取数据（O(n)复杂度）
        
        Parameters
        ----------
        n_samples : int
            要读取的样本数
        start_pos : int, optional
            起始位置（相对于最新数据），None表示从最旧数据开始
            
        Returns
        -------
        data : np.ndarray
            EEG数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray
            时间戳，shape: (n_samples,)
        """
        with self._lock:
            if self._current_size == 0:
                return np.array([]), np.array([])
            
            # 限制读取数量
            n_samples = min(n_samples, self._current_size)
            
            # 计算读取位置
            if start_pos is None:
                # 从最旧数据开始
                read_start = (self._write_idx - self._current_size + self.capacity) % self.capacity
            else:
                # 从最新数据向前n个样本
                read_start = (self._write_idx - start_pos - 1 + self.capacity) % self.capacity
            
            # 读取数据
            data = np.zeros((self.n_channels, n_samples), dtype=self.dtype)
            timestamps = np.zeros(n_samples, dtype=np.float64)
            
            for i in range(n_samples):
                idx = (read_start + i) % self.capacity
                data[:, i] = self._data[:, idx]
                timestamps[i] = self._timestamps[idx]
            
            self._last_read_time = time.time()
            
            return data, timestamps
    
    def get_latest(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取最新n个样本
        
        Parameters
        ----------
        n_samples : int
            样本数量
            
        Returns
        -------
        data : np.ndarray
            最新数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray
            时间戳
        """
        return self.read(n_samples, start_pos=n_samples)
    
    def get_window(self, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定时间窗口的数据
        
        Parameters
        ----------
        start : float
            开始时间（秒，相对于最新数据）
        end : float
            结束时间（秒，相对于最新数据）
            
        Returns
        -------
        data : np.ndarray
        timestamps : np.ndarray
        """
        with self._lock:
            if self._current_size == 0:
                return np.array([]), np.array([])
            
            # 转换为样本数
            start_samples = int(start * self.sampling_rate)
            end_samples = int(end * self.sampling_rate)
            
            # 限制范围
            start_samples = max(0, min(start_samples, self._current_size))
            end_samples = max(0, min(end_samples, self._current_size))
            
            # 确保顺序正确
            if start_samples > end_samples:
                start_samples, end_samples = end_samples, start_samples
            
            n_samples = end_samples - start_samples
            
            return self.read(n_samples, start_pos=end_samples)
    
    def get_until_marker(
        self, 
        start_marker: int, 
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取从指定标记开始的数据
        
        Parameters
        ----------
        start_marker : int
            起始标记值
        max_samples : int, optional
            最大样本数限制
            
        Returns
        -------
        data : np.ndarray
        timestamps : np.ndarray
        markers : np.ndarray
        """
        with self._lock:
            if self._current_size == 0:
                return np.array([]), np.array([]), np.array([])
            
            # 查找标记位置
            marker_pos = None
            for i in range(self._current_size):
                idx = (self._write_idx - self._current_size + i) % self.capacity
                if self._markers[idx] == start_marker:
                    marker_pos = i
                    break
            
            if marker_pos is None:
                # 未找到标记，返回空
                return np.array([]), np.array([]), np.array([])
            
            # 计算样本数
            n_samples = min(
                self._current_size - marker_pos,
                max_samples if max_samples else self._current_size - marker_pos
            )
            
            # 读取数据
            data = np.zeros((self.n_channels, n_samples), dtype=self.dtype)
            timestamps = np.zeros(n_samples, dtype=np.float64)
            markers_out = np.zeros(n_samples, dtype=np.int32)
            
            start_idx = (self._write_idx - self._current_size + marker_pos) % self.capacity
            
            for i in range(n_samples):
                idx = (start_idx + i) % self.capacity
                data[:, i] = self._data[:, idx]
                timestamps[i] = self._timestamps[idx]
                markers_out[i] = self._markers[idx]
            
            return data, timestamps, markers_out
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._data.fill(0)
            self._timestamps.fill(0)
            self._markers.fill(0)
            self._write_idx = 0
            self._read_idx = 0
            self._current_size = 0
    
    def resize(self, new_max_duration: float) -> bool:
        """
        调整缓冲区大小
        
        Parameters
        ----------
        new_max_duration : float
            新的最大持续时间（秒）
            
        Returns
        -------
        success : bool
        """
        with self._lock:
            new_capacity = int(new_max_duration * self.sampling_rate)
            
            if new_capacity == self.capacity:
                return True
            
            # 保存现有数据
            existing_data, existing_timestamps = self.read(self._current_size)
            
            # 重新分配
            self.capacity = new_capacity
            self._data = np.zeros((self.n_channels, self.capacity), dtype=self.dtype)
            self._timestamps = np.zeros(self.capacity, dtype=np.float64)
            self._markers = np.zeros(self.capacity, dtype=np.int32)
            
            # 恢复数据
            if existing_data.size > 0:
                write_samples = min(existing_data.shape[1], new_capacity)
                self._data[:, :write_samples] = existing_data[:, :write_samples]
                self._timestamps[:write_samples] = existing_timestamps[:write_samples]
                self._write_idx = write_samples
                self._current_size = write_samples
            
            return True
    
    def register_write_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        注册写入回调
        
        Parameters
        ----------
        callback : Callable
            回调函数，签名: callback(data, n_samples)
        """
        if callback not in self._write_callbacks:
            self._write_callbacks.append(callback)
    
    def unregister_write_callback(self, callback: Callable[[np.ndarray, int], None]):
        """注销写入回调"""
        if callback in self._write_callbacks:
            self._write_callbacks.remove(callback)
    
    def to_numpy(self) -> np.ndarray:
        """
        将缓冲区数据转换为numpy数组
        
        Returns
        -------
        data : np.ndarray
            所有可用数据，shape: (n_channels, n_samples)
        """
        data, _ = self.read(self._current_size)
        return data
    
    def get_data_slice(self, time_range: Tuple[float, float]) -> np.ndarray:
        """
        获取指定时间范围的数据（相对于数据开始）
        
        Parameters
        ----------
        time_range : tuple
            (start_time, end_time) 单位：秒
            
        Returns
        -------
        data : np.ndarray
        """
        start_time, end_time = time_range
        start_samples = int(start_time * self.sampling_rate)
        end_samples = int(end_time * self.sampling_rate)
        
        return self.read(end_samples - start_samples, start_pos=self._current_size - start_samples)
    
    def __len__(self) -> int:
        """返回当前缓冲区大小"""
        return self._current_size
    
    def __repr__(self) -> str:
        """返回字符串表示"""
        return (f"EfficientRingBuffer(n_channels={self.n_channels}, "
                f"capacity={self.capacity}, size={self._current_size}, "
                f"fill_ratio={self._current_size/max(1,self.capacity):.1%})")


class MultiChannelRingBuffer:
    """
    多通道分组环形缓冲区
    
    用于管理多个独立的EEG数据通道组，
    每个通道组有独立的缓冲区。
    """
    
    def __init__(
        self,
        channel_groups: Dict[str, List[int]],
        max_duration: float = 30.0,
        sampling_rate: float = 1000.0,
        dtype: np.dtype = np.float32
    ):
        """
        初始化多通道分组缓冲区
        
        Parameters
        ----------
        channel_groups : dict
            通道分组，格式: {'group_name': [channel_indices]}
        max_duration : float
            最大持续时间
        sampling_rate : float
            采样率
        dtype : np.dtype
            数据类型
        """
        self.channel_groups = {}
        self.buffers = {}
        
        for group_name, channel_indices in channel_groups.items():
            self.channel_groups[group_name] = channel_indices
            self.buffers[group_name] = EfficientRingBuffer(
                n_channels=len(channel_indices),
                max_duration=max_duration,
                sampling_rate=sampling_rate,
                dtype=dtype
            )
        
        # 全局缓冲区
        total_channels = sum(len(ch) for ch in channel_groups.values())
        self.global_buffer = EfficientRingBuffer(
            n_channels=total_channels,
            max_duration=max_duration,
            sampling_rate=sampling_rate,
            dtype=dtype
        )
    
    def write(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """写入数据到所有缓冲区"""
        # 写入全局缓冲区
        self.global_buffer.write(data, timestamps)
        
        # 写入各分组缓冲区
        for group_name, channel_indices in self.channel_groups.items():
            group_data = data[channel_indices, :]
            self.buffers[group_name].write(group_data, timestamps)
    
    def get_group_data(self, group_name: str, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定分组的数据"""
        if group_name in self.buffers:
            return self.buffers[group_name].get_latest(n_samples)
        return np.array([]), np.array([])
    
    def get_global_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取全局数据"""
        return self.global_buffer.get_latest(n_samples)


# ============================================================================
# 便捷函数
# ============================================================================

def create_buffer(
    n_channels: int = 16,
    duration: float = 30.0,
    sampling_rate: float = 1000.0
) -> EfficientRingBuffer:
    """
    创建环形缓冲区的便捷函数
    
    Parameters
    ----------
    n_channels : int
        通道数
    duration : float
        持续时间（秒）
    sampling_rate : float
        采样率
        
    Returns
    -------
    buffer : EfficientRingBuffer
    """
    return EfficientRingBuffer(
        n_channels=n_channels,
        max_duration=duration,
        sampling_rate=sampling_rate
    )


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MetaBCI Brainflow - 高效环形缓冲区演示")
    print("=" * 60)
    
    # 创建缓冲区：16通道，10秒数据，1000Hz采样
    buffer = EfficientRingBuffer(
        n_channels=16,
        max_duration=10.0,
        sampling_rate=1000.0
    )
    
    print(f"\n缓冲区信息: {buffer}")
    print(f"容量: {buffer.capacity} 样本 ({buffer.capacity/buffer.sampling_rate:.1f}秒)")
    
    # 模拟写入数据
    print("\n写入模拟数据...")
    import time
    
    for i in range(5):
        # 模拟100ms的数据
        samples = 100
        data = np.random.randn(16, samples) * 10
        buffer.write(data)
        print(f"  写入 {samples} 样本，当前大小: {buffer.size}")
        time.sleep(0.05)
    
    # 读取最新数据
    print("\n读取最新500个样本...")
    latest_data, latest_timestamps = buffer.get_latest(500)
    print(f"  读取到数据形状: {latest_data.shape}")
    
    # 获取缓冲区状态
    status = buffer.status
    print(f"\n缓冲区状态:")
    print(f"  填充比例: {status.fill_ratio:.1%}")
    print(f"  持续时间: {status.duration_seconds:.2f}秒")
    print(f"  写入位置: {status.write_pos}")
    
    print("\n演示完成!")
