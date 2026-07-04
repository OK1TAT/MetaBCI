# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 工具类与数据类

提供Stroke EEG采集系统的基础数据结构、统计工具、队列操作和日志配置。
所有数据类均使用 @dataclass 装饰器，线程安全操作通过 RuntimeStats 保证。

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class RawChunk:
    """
    原始数据块
    
    从网络设备接收到的原始字节数据，附带接收时间戳。
    
    Attributes
    ----------
    recv_timestamp : float
        数据接收时刻（time.time()）
    data : bytes
        原始字节数据
    """
    recv_timestamp: float
    data: bytes


@dataclass
class DecodedSample:
    """
    解码后的EEG样本
    
    一个完整的EEG采样点，包含样本编号、原始计数值和微伏电压值。
    在16通道模式下，一个DecodedSample由两个连续33字节包配对而成。
    
    Attributes
    ----------
    recv_timestamp : float
        接收时间戳（time.time()）
    sample_number : int
        样本编号（0-255循环，来自第一个包的sample_number）
    counts : list[int]
        各通道原始ADC计数值（24位有符号整数）
    uV : list[float]
        各通道电压值（微伏，= counts × scale_factor）
    """
    recv_timestamp: float
    sample_number: int
    counts: list[int]
    uV: list[float]


@dataclass
class EventMarker:
    """
    事件标记
    
    实验过程中打上的事件标记，用于后续ERP分析或试次切割。
    
    Attributes
    ----------
    timestamp : float
        标记时刻（time.time()）
    label : str
        标记标签（如 'stimulus', 'response', 'rest_start' 等）
    """
    timestamp: float
    label: str


# ============================================================================
# 运行时统计类
# ============================================================================

class RuntimeStats:
    """
    线程安全的运行时统计类
    
    采集系统各模块共享的运行时状态与计数器，所有操作均通过
    内部锁保证线程安全。支持快照导出和会话重置。
    
    统计项包括：
    - 连接状态
    - 数据包丢失/错误计数
    - 16ch配对失败计数
    - 队列溢出丢包计数
    - 重连次数
    - 解码/保存样本计数
    - 输出文件路径
    - 最近错误信息
    
    Examples
    --------
    >>> stats = RuntimeStats()
    >>> stats.set_connected(True)
    >>> stats.add_decoded(100)
    >>> snap = stats.snapshot()
    >>> print(snap["decoded_samples"])
    100
    """
    
    def __init__(self) -> None:
        """初始化运行时统计，所有计数器归零"""
        self._lock = threading.Lock()
        self.reset_session()
    
    def reset_session(self) -> None:
        """重置所有统计项（新会话开始时调用）"""
        with self._lock:
            self.connected: bool = False
            self.packet_drop_count: int = 0
            self.bad_packet_count: int = 0
            self.pair_mismatch_count: int = 0
            self.queue_drop_count: int = 0
            self.reconnect_count: int = 0
            self.decoded_samples: int = 0
            self.saved_samples: int = 0
            self.last_error: str = ""
            self.output_csv: Optional[str] = None
            self.output_npz: Optional[str] = None
            self.session_start_ts: float = time.time()
    
    def set_connected(self, connected: bool) -> None:
        """设置设备连接状态"""
        with self._lock:
            self.connected = connected
    
    def add_packet_gap(self, count: int) -> None:
        """增加丢包计数（sample_number不连续时调用）"""
        if count <= 0:
            return
        with self._lock:
            self.packet_drop_count += count
    
    def add_bad_packet(self, count: int = 1) -> None:
        """增加坏包计数（帧头/帧尾校验失败时调用）"""
        with self._lock:
            self.bad_packet_count += max(1, count)
    
    def add_pair_mismatch(self, count: int = 1) -> None:
        """增加16ch配对失败计数"""
        with self._lock:
            self.pair_mismatch_count += max(1, count)
    
    def add_queue_drop(self, count: int = 1) -> None:
        """增加队列溢出丢包计数"""
        with self._lock:
            self.queue_drop_count += max(1, count)
    
    def add_reconnect(self, count: int = 1) -> None:
        """增加重连次数"""
        with self._lock:
            self.reconnect_count += max(1, count)
    
    def add_decoded(self, count: int = 1) -> None:
        """增加已解码样本计数"""
        with self._lock:
            self.decoded_samples += max(1, count)
    
    def add_saved(self, count: int = 1) -> None:
        """增加已保存样本计数"""
        with self._lock:
            self.saved_samples += max(1, count)
    
    def set_error(self, msg: str) -> None:
        """记录最近一次错误信息"""
        with self._lock:
            self.last_error = msg
    
    def set_output_paths(self, csv_path: str, npz_path: Optional[str]) -> None:
        """设置输出文件路径"""
        with self._lock:
            self.output_csv = csv_path
            self.output_npz = npz_path
    
    def snapshot(self) -> dict[str, Any]:
        """
        获取统计快照
        
        Returns
        -------
        snapshot : dict
            包含所有统计项的字典，以及平均采样率 avg_rate
        """
        with self._lock:
            elapsed = max(1e-6, time.time() - self.session_start_ts)
            avg_rate = self.decoded_samples / elapsed
            return {
                "connected": self.connected,
                "packet_drop_count": self.packet_drop_count,
                "bad_packet_count": self.bad_packet_count,
                "pair_mismatch_count": self.pair_mismatch_count,
                "queue_drop_count": self.queue_drop_count,
                "reconnect_count": self.reconnect_count,
                "decoded_samples": self.decoded_samples,
                "saved_samples": self.saved_samples,
                "last_error": self.last_error,
                "output_csv": self.output_csv,
                "output_npz": self.output_npz,
                "avg_rate": avg_rate,
            }


# ============================================================================
# 工具函数
# ============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """
    配置全局日志格式
    
    仅在首次调用时生效（root logger无handler时）。
    
    Parameters
    ----------
    level : int
        日志级别，默认 logging.INFO
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_session_stem(subject: str) -> str:
    """
    生成会话文件名前缀
    
    格式：{被试名}_{HHMMSS}_{YYYYMMDD}
    被试名中的非法字符会被替换为下划线。
    
    Parameters
    ----------
    subject : str
        被试标识
    
    Returns
    -------
    stem : str
        文件名前缀（不含扩展名）
    
    Examples
    --------
    >>> format_session_stem("subject01")
    'subject01_143025_20240115'
    """
    clean_subject = re.sub(r"[^A-Za-z0-9_\-]", "_", subject.strip() or "subject")
    now = datetime.now()
    return f"{clean_subject}_{now.strftime('%H%M%S')}_{now.strftime('%Y%m%d')}"


def put_with_drop_oldest(
    q: "queue.Queue[Any]",
    item: Any,
    stats: RuntimeStats,
    logger: logging.Logger,
    queue_name: str,
) -> bool:
    """
    带丢最旧元素的队列写入
    
    当队列已满时，先丢弃队首最旧元素，再写入新元素。
    通过 RuntimeStats 记录丢包次数。
    
    Parameters
    ----------
    q : queue.Queue
        目标队列
    item : Any
        待写入元素
    stats : RuntimeStats
        运行时统计（记录丢包）
    logger : logging.Logger
        日志记录器
    queue_name : str
        队列名称（用于日志输出）
    
    Returns
    -------
    success : bool
        是否成功写入
    """
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        # 丢弃最旧元素
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        # 再次尝试写入
        try:
            q.put_nowait(item)
            stats.add_queue_drop(1)
            logger.warning("队列 %s 已满，丢弃最旧元素。", queue_name)
            return True
        except queue.Full:
            stats.add_queue_drop(1)
            logger.warning("队列 %s 仍然已满，丢弃新元素。", queue_name)
            return False
