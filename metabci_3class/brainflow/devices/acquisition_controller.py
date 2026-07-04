# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 采集控制器

整合设备适配器、数据包解码器、数据保存器和环形缓冲区，
提供统一的高层采集控制接口。管理所有后台线程的生命周期，
支持动态配置更新、事件标记注入和运行时状态查询。

数据流:
    网络设备 → DeviceAdapter → raw_queue
                                ↓
                          PacketDecoderThread
                           ↓           ↓
                      plot_queue    save_queue
                       (可视化)         ↓
                                  DataSaverThread → CSV/NPZ
                                  RawStreamSaverThread → raw.bin

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, List

from .config import DeviceConfig
from .device_adapter import DeviceAdapter
from .packet_decoder import PacketDecoderThread
from .data_saver import DataSaverThread, RawStreamSaverThread
from .ring_buffer import EfficientRingBuffer
from .utils import (
    DecodedSample,
    EventMarker,
    RawChunk,
    RuntimeStats,
    format_session_stem,
    put_with_drop_oldest,
)


class AcquisitionController:
    """
    EEG采集控制器
    
    管理设备适配器、解码器、保存器等所有后台线程的生命周期。
    提供简洁的 start/stop/add_marker 接口，支持动态配置更新。
    
    内部线程:
    1. DeviceAdapter (接收线程) - 网络设备数据接收
    2. PacketDecoderThread (解码线程) - 33字节包解析
    3. DataSaverThread (保存线程) - CSV/NPZ持久化
    4. RawStreamSaverThread (原始保存线程) - 原始字节流保存
    
    Parameters
    ----------
    config : DeviceConfig
        设备配置
    
    Examples
    --------
    >>> config = DeviceConfig(protocol="wifi_shield", host="192.168.4.1")
    >>> config.validate()
    >>> controller = AcquisitionController(config)
    >>> controller.start()
    True
    >>> controller.add_marker("stimulus")
    >>> # ... 采集进行中 ...
    >>> controller.stop()
    """
    
    def __init__(self, config: DeviceConfig) -> None:
        self.config = config
        self.stats = RuntimeStats()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 线程间队列
        self.raw_queue: "queue.Queue[RawChunk]" = queue.Queue(
            maxsize=self.config.raw_queue_size
        )
        self.raw_save_queue: "queue.Queue[RawChunk]" = queue.Queue(
            maxsize=max(self.config.raw_queue_size * 2, 10000)
        )
        self.plot_queue: "queue.Queue[DecodedSample]" = queue.Queue(
            maxsize=self.config.plot_queue_size
        )
        self.save_queue: "queue.Queue[DecodedSample]" = queue.Queue(
            maxsize=self.config.save_queue_size
        )
        self.marker_queue: "queue.Queue[EventMarker]" = queue.Queue(
            maxsize=10000
        )
        
        # 环形缓冲区（供UI实时显示）
        self._ring_buffer: Optional[EfficientRingBuffer] = None
        
        # 线程管理
        self._lock = threading.Lock()
        self._stop_event: Optional[threading.Event] = None
        self._threads: List[threading.Thread] = []
        self._adapter: Optional[DeviceAdapter] = None
        self._join_thread: Optional[threading.Thread] = None
    
    # ================================================================
    # 配置更新
    # ================================================================
    
    def update_config(
        self,
        *,
        host: str,
        port: int,
        protocol: str,
        channels: int,
        sample_rate: int,
        endian: str,
        force_input_channels: Optional[int] = None,
        display_seconds: float,
        save_dir: str,
        subject: str,
    ) -> None:
        """
        更新设备配置（停止状态下调用）
        
        Parameters
        ----------
        host : str
            设备IP
        port : int
            端口
        protocol : str
            协议类型
        channels : int
            通道数
        sample_rate : int
            采样率
        endian : str
            字节序
        force_input_channels : int, optional
            强制输入通道数
        display_seconds : float
            显示时长
        save_dir : str
            保存目录
        subject : str
            被试标识
        """
        with self._lock:
            self.config.host = host
            self.config.port = port
            self.config.protocol = protocol
            self.config.channels = channels
            self.config.sample_rate = sample_rate
            self.config.endian = endian
            if force_input_channels is not None:
                self.config.force_input_channels = force_input_channels
            self.config.display_seconds = display_seconds
            self.config.save_dir = save_dir
            self.config.subject = subject
            self.config.validate()
    
    # ================================================================
    # 状态查询
    # ================================================================
    
    def is_running(self) -> bool:
        """
        查询采集是否正在运行
        
        Returns
        -------
        running : bool
            是否有线程仍在运行
        """
        with self._lock:
            return any(t.is_alive() for t in self._threads)
    
    def get_stats(self) -> dict:
        """
        获取运行时统计快照
        
        Returns
        -------
        stats : dict
            包含所有统计项的字典
        """
        return self.stats.snapshot()
    
    def get_ring_buffer(self) -> Optional[EfficientRingBuffer]:
        """
        获取环形缓冲区（用于UI实时显示）
        
        Returns
        -------
        buffer : EfficientRingBuffer or None
            缓冲区实例（采集运行中）
        """
        return self._ring_buffer
    
    def get_plot_queue(self) -> "queue.Queue[DecodedSample]":
        """获取绘图数据队列（供UI消费）"""
        return self.plot_queue
    
    # ================================================================
    # 生命周期管理
    # ================================================================
    
    def start(self) -> bool:
        """
        启动采集
        
        创建并启动所有后台线程：
        1. DeviceAdapter - 网络设备数据接收
        2. PacketDecoderThread - 数据包解码
        3. DataSaverThread - CSV/NPZ保存
        4. RawStreamSaverThread - 原始字节流保存
        
        Returns
        -------
        success : bool
            是否成功启动
        message : str
            状态信息
        """
        with self._lock:
            if any(t.is_alive() for t in self._threads):
                return False
            if self._join_thread is not None and self._join_thread.is_alive():
                self.logger.warning("上一个会话正在停止，请稍候...")
                return False
            
            # 清空所有队列
            self._clear_queue(self.raw_queue)
            self._clear_queue(self.raw_save_queue)
            self._clear_queue(self.plot_queue)
            self._clear_queue(self.save_queue)
            self._clear_queue(self.marker_queue)
            
            # 重置统计
            self.stats.reset_session()
            
            # 创建停止事件
            self._stop_event = threading.Event()
            
            # 生成会话名
            session_stem = format_session_stem(self.config.subject)
            
            # 1. 创建设备适配器
            self._adapter = DeviceAdapter(config=self.config, stats=self.stats)
            self._adapter.set_output_queues(
                raw_queue=self.raw_queue,
                raw_save_queue=self.raw_save_queue,
            )
            
            # 2. 创建解码线程
            decoder = PacketDecoderThread(
                config=self.config,
                raw_queue=self.raw_queue,
                plot_queue=self.plot_queue,
                save_queue=self.save_queue,
                stop_event=self._stop_event,
                stats=self.stats,
            )
            
            # 3. 创建数据保存线程
            saver = DataSaverThread(
                config=self.config,
                save_queue=self.save_queue,
                marker_queue=self.marker_queue,
                stop_event=self._stop_event,
                stats=self.stats,
                session_stem=session_stem,
            )
            
            # 4. 创建原始数据保存线程
            raw_saver = RawStreamSaverThread(
                config=self.config,
                raw_save_queue=self.raw_save_queue,
                stop_event=self._stop_event,
                session_stem=session_stem,
            )
            
            # 创建环形缓冲区（用于UI实时显示）
            import numpy as np
            self._ring_buffer = EfficientRingBuffer(
                n_channels=self.config.channels,
                max_duration=self.config.display_seconds,
                sampling_rate=float(self.config.sample_rate),
                dtype=np.float32,
            )
            
            # 启动所有线程
            self._threads = [decoder, saver, raw_saver]
            
            # 启动设备适配器
            self._adapter.start()
            
            # 启动工作线程
            for thread in self._threads:
                thread.start()
            
            self.logger.info("采集已启动: session=%s", session_stem)
            return True
    
    def stop(self, wait: bool = False) -> bool:
        """
        停止采集
        
        发送停止信号，依次关闭设备适配器和所有工作线程。
        
        Parameters
        ----------
        wait : bool
            是否同步等待所有线程退出（False则在后台线程中等待）
        
        Returns
        -------
        stopped : bool
            是否成功发起停止
        """
        with self._lock:
            threads = [t for t in self._threads if t.is_alive()]
            if not threads:
                return False
            
            # 设置停止信号
            if self._stop_event is not None:
                self._stop_event.set()
            
            # 停止设备适配器
            if self._adapter is not None:
                self._adapter.stop()
            
            def _join_all() -> None:
                for thread in threads:
                    thread.join(timeout=5.0)
                with self._lock:
                    self._threads = []
                    self._ring_buffer = None
                self.logger.info("采集已停止。")
            
            if wait:
                _join_all()
            else:
                self._join_thread = threading.Thread(
                    target=_join_all, daemon=True
                )
                self._join_thread.start()
            
            return True
    
    # ================================================================
    # 事件标记
    # ================================================================
    
    def add_marker(
        self,
        label: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        添加事件标记
        
        事件标记会被写入保存队列，与EEG数据同步记录。
        
        Parameters
        ----------
        label : str
            标记标签（如 'stimulus', 'response', 'rest_start'）
        timestamp : float, optional
            标记时刻，默认使用当前时间
        """
        clean_label = str(label).strip()
        if not clean_label:
            return
        
        marker = EventMarker(
            timestamp=float(timestamp if timestamp is not None else time.time()),
            label=clean_label,
        )
        
        put_with_drop_oldest(
            self.marker_queue,
            marker,
            self.stats,
            self.logger,
            "marker_queue",
        )
    
    # ================================================================
    # 内部工具
    # ================================================================
    
    @staticmethod
    def _clear_queue(q: "queue.Queue") -> None:
        """清空队列中的所有元素"""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
