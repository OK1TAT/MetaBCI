# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - LSL实时数据采集适配器
提供局域网内LSL数据流的自动发现和实时采集功能

作者: 抑郁症EEG认知障碍识别项目组
版本: 1.0.0
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from threading import Thread, Event, Lock
from dataclasses import dataclass
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入pylsl
try:
    import pylsl
    PYLSL_AVAILABLE = True
except ImportError:
    PYLSL_AVAILABLE = False
    logger.warning("pylsl未安装，将使用模拟数据源")

# 尝试导入brainflow
try:
    from brainflow.data_filter import DataFilter
    from brainflow.board_shim import BoardShim
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logger.warning("brainflow未安装")


@dataclass
class StreamInfo:
    """LSL数据流信息"""
    name: str                      # 流名称
    stream_type: str               # 流类型 (EEG, Marker等)
    n_channels: int                # 通道数
    sampling_rate: float            # 采样率
    channel_names: List[str]       # 通道名称
    source_id: str                 # 源ID
    hostname: Optional[str] = None # 主机名

    @property
    def duration_per_sample(self) -> float:
        """每个样本的持续时间（秒）"""
        return 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0


class LSLStreamDiscovery:
    """
    LSL数据流发现器
    
    自动扫描局域网内的所有LSL数据流，支持按类型筛选
    """
    
    def __init__(self, timeout: float = 5.0):
        """
        初始化数据流发现器
        
        Parameters
        ----------
        timeout : float
            发现超时时间（秒）
        """
        self.timeout = timeout
    
    def discover(self, stream_type: Optional[str] = None) -> List[StreamInfo]:
        """
        发现所有可用的LSL数据流
        
        Parameters
        ----------
        stream_type : str, optional
            筛选流类型，如'EEG', 'Markers'等
            
        Returns
        -------
        streams : List[StreamInfo]
            发现的数据流列表
        """
        if not PYLSL_AVAILABLE:
            logger.warning("pylsl不可用，无法发现数据流")
            return []
        
        streams = []
        try:
            # 解析所有流
            stream_infos = pylsl.resolve_streams(self.timeout)
            
            for inlet in stream_infos:
                info = inlet.info()
                
                # 类型筛选
                if stream_type and info.type().lower() != stream_type.lower():
                    continue
                
                # 解析通道名称
                ch_names_xml = info.desc().child_value("channels")
                if ch_names_xml:
                    channel_names = [c.strip() for c in ch_names_xml.split(',')]
                else:
                    channel_names = [f"Ch{i+1}" for i in range(info.channel_count())]
                
                stream_info = StreamInfo(
                    name=info.name(),
                    stream_type=info.type(),
                    n_channels=info.channel_count(),
                    sampling_rate=info.nominal_srate(),
                    channel_names=channel_names,
                    source_id=info.source_id(),
                    hostname=info.hostname() if hasattr(info, 'hostname') else None
                )
                streams.append(stream_info)
                logger.info(f"发现数据流: {stream_info.name} ({stream_info.stream_type}), "
                          f"通道数: {stream_info.n_channels}, 采样率: {stream_info.sampling_rate}")
                
        except Exception as e:
            logger.error(f"发现数据流时出错: {e}")
        
        return streams
    
    def find_by_name(self, name: str) -> Optional[StreamInfo]:
        """
        按名称查找数据流
        
        Parameters
        ----------
        name : str
            流名称（支持部分匹配）
            
        Returns
        -------
        stream : StreamInfo or None
        """
        all_streams = self.discover()
        for stream in all_streams:
            if name.lower() in stream.name.lower():
                return stream
        return None


class LSLAdapter:
    """
    LSL实时数据采集适配器
    
    自动连接局域网内的LSL EEG数据流，支持多设备数据同步采集。
    支持的数据源包括：OpenBCI, g.tec, Neuroscan等通过LSL推送的设备。
    
    Features:
    - 自动发现EEG类型数据流
    - 多设备同步采集
    - 自动解析数据流元信息
    - 线程安全的数据获取
    - 优雅降级：无pylsl时提供模拟数据源
    """
    
    def __init__(
        self,
        stream_names: Optional[List[str]] = None,
        stream_type: str = 'EEG',
        buffer_size: int = 10000,
        sampling_rate: Optional[float] = None,
        n_channels: Optional[int] = None,
        channel_names: Optional[List[str]] = None
    ):
        """
        初始化LSL适配器
        
        Parameters
        ----------
        stream_names : List[str], optional
            要连接的流名称列表，None表示自动发现所有EEG流
        stream_type : str
            流类型筛选
        buffer_size : int
            内部缓冲区大小
        sampling_rate : float, optional
            采样率（从数据流自动获取，可手动指定）
        n_channels : int, optional
            通道数（从数据流自动获取，可手动指定）
        channel_names : List[str], optional
            通道名称列表（从数据流自动获取，可手动指定）
        """
        self.stream_names = stream_names or []
        self.stream_type = stream_type
        self.buffer_size = buffer_size
        
        # 流信息
        self._streams: Dict[str, StreamInfo] = {}
        self._inlets: Dict[str, any] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        
        # 运行时状态
        self._running = False
        self._acquisition_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._lock = Lock()
        
        # 回调函数
        self._data_callbacks: List[Callable[[np.ndarray, str], None]] = []
        
        # 自动发现的配置
        self._auto_discover = stream_names is None or len(stream_names) == 0
        
        # 手动配置的回退值
        self._fallback_sampling_rate = sampling_rate or 1000.0
        self._fallback_n_channels = n_channels or 16
        self._fallback_channel_names = channel_names or [f"Ch{i+1}" for i in range(16)]
        
        # 统计信息
        self._total_samples = 0
        self._last_sample_time = 0.0
    
    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running
    
    @property
    def connected_streams(self) -> List[str]:
        """获取已连接的流名称列表"""
        return list(self._streams.keys())
    
    @property
    def stream_info(self) -> Dict[str, StreamInfo]:
        """获取所有已连接流的信息"""
        return self._streams.copy()
    
    def _discover_and_connect(self) -> bool:
        """
        自动发现并连接数据流
        
        Returns
        -------
        success : bool
            是否成功连接至少一个流
        """
        if not PYLSL_AVAILABLE:
            logger.warning("pylsl不可用，将使用模拟数据源")
            return False
        
        discovery = LSLStreamDiscovery(timeout=5.0)
        discovered = discovery.discover(self.stream_type)
        
        if not discovered:
            logger.warning("未发现EEG数据流")
            return False
        
        # 连接所有发现的流或指定的流
        connected = False
        for stream in discovered:
            if self.stream_names and stream.name not in self.stream_names:
                continue
            
            try:
                # 创建 inlet
                inlet = pylsl.stream_inlet(stream, max_buffered=360)
                inlet.open_stream()
                
                self._inlets[stream.name] = inlet
                self._streams[stream.name] = stream
                self._buffers[stream.name] = np.zeros(
                    (stream.n_channels, self.buffer_size), dtype=np.float32
                )
                
                logger.info(f"成功连接到流: {stream.name}")
                connected = True
                
            except Exception as e:
                logger.error(f"连接流 {stream.name} 失败: {e}")
        
        return connected
    
    def _connect_by_name(self, name: str) -> bool:
        """
        按名称连接数据流
        
        Parameters
        ----------
        name : str
            流名称
            
        Returns
        -------
        success : bool
        """
        if not PYLSL_AVAILABLE:
            return False
        
        try:
            streams = pylsl.resolve_byename(name, timeout=5.0)
            if not streams:
                logger.warning(f"未找到名为 {name} 的流")
                return False
            
            inlet = pylsl.stream_inlet(streams[0], max_buffered=360)
            inlet.open_stream()
            
            info = streams[0]
            channel_names = []
            try:
                ch_xml = info.desc().child_value("channels")
                if ch_xml:
                    channel_names = [c.strip() for c in ch_xml.split(',')]
            except:
                channel_names = [f"Ch{i+1}" for i in range(info.channel_count())]
            
            stream_info = StreamInfo(
                name=info.name(),
                stream_type=info.type(),
                n_channels=info.channel_count(),
                sampling_rate=info.nominal_srate(),
                channel_names=channel_names,
                source_id=info.source_id()
            )
            
            self._inlets[name] = inlet
            self._streams[name] = stream_info
            self._buffers[name] = np.zeros(
                (stream_info.n_channels, self.buffer_size), dtype=np.float32
            )
            
            logger.info(f"成功连接到流: {name}")
            return True
            
        except Exception as e:
            logger.error(f"连接流 {name} 失败: {e}")
            return False
    
    def start(self) -> bool:
        """
        启动数据采集
        
        Returns
        -------
        success : bool
        """
        if self._running:
            logger.warning("适配器已在运行")
            return True
        
        self._stop_event.clear()
        
        # 连接数据流
        if self._auto_discover:
            success = self._discover_and_connect()
            if not success:
                logger.warning("自动发现失败，将使用模拟数据源")
        else:
            success = False
            for name in self.stream_names:
                if self._connect_by_name(name):
                    success = True
        
        # 如果没有连接任何流，创建模拟源
        if not self._streams:
            logger.info("使用模拟数据源")
            self._streams['synthetic'] = StreamInfo(
                name='synthetic',
                stream_type='EEG',
                n_channels=self._fallback_n_channels,
                sampling_rate=self._fallback_sampling_rate,
                channel_names=self._fallback_channel_names,
                source_id='synthetic_source'
            )
            self._inlets['synthetic'] = None
            self._buffers['synthetic'] = np.zeros(
                (self._fallback_n_channels, self.buffer_size), dtype=np.float32
            )
            success = True
        
        if success:
            self._running = True
            self._acquisition_thread = Thread(target=self._acquisition_loop, daemon=True)
            self._acquisition_thread.start()
            logger.info("数据采集已启动")
        
        return success
    
    def stop(self):
        """停止数据采集"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=2.0)
        
        # 关闭所有inlet
        for name, inlet in self._inlets.items():
            if inlet is not None:
                try:
                    inlet.close_stream()
                except:
                    pass
        
        self._inlets.clear()
        self._streams.clear()
        
        logger.info("数据采集已停止")
    
    def _acquisition_loop(self):
        """数据采集主循环"""
        buffer_idx = 0
        
        while not self._stop_event.is_set():
            for name, inlet in self._inlets.items():
                if inlet is None:
                    # 模拟数据源
                    self._generate_synthetic_data(name)
                    continue
                
                try:
                    # 尝试获取所有可用样本
                    samples, timestamps = inlet.pull_chunk(
                        timeout=0.0, max_samples=100
                    )
                    
                    if samples and timestamps:
                        samples = np.array(samples, dtype=np.float32).T  # (n_channels, n_samples)
                        self._append_to_buffer(name, samples, timestamps)
                        
                        # 触发回调
                        for callback in self._data_callbacks:
                            callback(samples, name)
                            
                except Exception as e:
                    logger.debug(f"从 {name} 获取数据时出错: {e}")
            
            time.sleep(0.001)  # 短暂休眠避免CPU占用过高
    
    def _generate_synthetic_data(self, name: str):
        """生成模拟EEG数据"""
        n_new = 10  # 每次生成10个样本
        synthetic_data = self._generate_eeg_waveform(
            n_channels=self._streams[name].n_channels,
            n_samples=n_new,
            sampling_rate=self._streams[name].sampling_rate
        )
        
        timestamps = np.array([time.time() - (n_new - i) / self._streams[name].sampling_rate 
                                for i in range(n_new)])
        
        self._append_to_buffer(name, synthetic_data, timestamps)
    
    def _generate_eeg_waveform(
        self, 
        n_channels: int, 
        n_samples: int, 
        sampling_rate: float
    ) -> np.ndarray:
        """生成模拟EEG波形"""
        t = np.arange(n_samples) / sampling_rate
        
        data = np.zeros((n_channels, n_samples))
        
        # 各频段成分
        delta = 2 * np.sin(2 * np.pi * 2 * t)       # 2Hz delta
        theta = 3 * np.sin(2 * np.pi * 6 * t)      # 6Hz theta
        alpha = 5 * np.sin(2 * np.pi * 10 * t)      # 10Hz alpha
        beta = 2 * np.sin(2 * np.pi * 20 * t)      # 20Hz beta
        
        # 混合各频段
        base_signal = delta + theta + alpha + beta
        
        for ch in range(n_channels):
            # 添加通道特异性
            phase_shift = np.random.rand() * 2 * np.pi
            noise = 0.5 * np.random.randn(n_samples)
            data[ch] = base_signal * np.sin(t * 2 + phase_shift) + noise
        
        return data
    
    def _append_to_buffer(
        self, 
        name: str, 
        data: np.ndarray, 
        timestamps: np.ndarray
    ):
        """追加数据到缓冲区"""
        with self._lock:
            if name not in self._buffers:
                return
            
            n_new = data.shape[1]
            buffer = self._buffers[name]
            
            # 循环缓冲区写入
            for i in range(n_new):
                idx = (self._total_samples + i) % self.buffer_size
                if idx < buffer.shape[1]:
                    buffer[:, idx] = data[:, i]
            
            self._total_samples += n_new
            self._last_sample_time = time.time()
    
    def get_data(
        self, 
        stream_name: Optional[str] = None,
        n_samples: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取采集的数据
        
        Parameters
        ----------
        stream_name : str, optional
            流名称，None表示获取第一个流的数据
        n_samples : int, optional
            样本数量，None表示获取所有可用数据
            
        Returns
        -------
        data : np.ndarray or None
            EEG数据，shape: (n_channels, n_samples)
        timestamps : np.ndarray or None
            时间戳
        """
        if not self._streams:
            return None, None
        
        # 选择流
        if stream_name is None:
            name = list(self._streams.keys())[0]
        else:
            name = stream_name
        
        if name not in self._buffers:
            return None, None
        
        with self._lock:
            buffer = self._buffers[name].copy()
            n_available = min(self._total_samples, self.buffer_size)
            
            if n_samples is None or n_samples >= n_available:
                # 返回所有可用数据
                return buffer[:, :n_available], None
            else:
                # 返回最新的n_samples
                start_idx = max(0, n_available - n_samples)
                return buffer[:, start_idx:n_available], None
    
    def get_latest(self, stream_name: Optional[str] = None, n_samples: int = 100) -> np.ndarray:
        """
        获取最新n个样本
        
        Parameters
        ----------
        stream_name : str, optional
            流名称
        n_samples : int
            样本数量
            
        Returns
        -------
        data : np.ndarray
            最新数据
        """
        data, _ = self.get_data(stream_name, n_samples)
        return data if data is not None else np.array([])
    
    def register_callback(self, callback: Callable[[np.ndarray, str], None]):
        """
        注册数据回调函数
        
        Parameters
        ----------
        callback : Callable
            回调函数，签名: callback(data: np.ndarray, stream_name: str)
        """
        self._data_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[np.ndarray, str], None]):
        """注销回调函数"""
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()


class SyntheticLSLSource:
    """
    模拟LSL数据源
    
    用于在没有真实设备时生成模拟EEG数据，
    并可通过pylsl推送到LSL网络供其他程序使用
    """
    
    def __init__(
        self,
        name: str = 'SyntheticEEG',
        n_channels: int = 16,
        sampling_rate: float = 1000.0,
        channel_names: Optional[List[str]] = None
    ):
        """
        初始化模拟数据源
        
        Parameters
        ----------
        name : str
            流名称
        n_channels : int
            通道数
        sampling_rate : float
            采样率
        channel_names : List[str], optional
            通道名称
        """
        self.name = name
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
        if channel_names:
            self.channel_names = channel_names
        else:
            # 默认16导联位置
            self.channel_names = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6'
            ][:n_channels]
        
        self._running = False
        self._thread: Optional[Thread] = None
        self._stream_outlet = None
        self._stop_event = Event()
        
        # EEG参数
        self._phase = np.zeros(n_channels)
        self._frequencies = np.array([
            2.0, 6.0, 10.0, 20.0  # delta, theta, alpha, beta
        ])
        self._amplitudes = np.array([2.0, 3.0, 5.0, 2.0])
    
    def _create_lsl_stream(self):
        """创建LSL输出流"""
        if not PYLSL_AVAILABLE:
            return None
        
        info = pylsl.stream_info(
            name=self.name,
            type='EEG',
            channel_count=self.n_channels,
            nominal_srate=self.sampling_rate,
            channel_format='float32',
            source_id='synthetic_amplifier'
        )
        
        # 添加通道描述
        chs = info.desc().append_child("channels")
        for ch_name in self.channel_names:
            ch = chs.append_child("channel")
            ch.append_child_value("label", ch_name)
            ch.append_child_value("type", "EEG")
            ch.append_child_value("unit", "microvolts")
        
        outlet = pylsl.stream_outlet(info)
        return outlet
    
    def start(self):
        """启动数据推送"""
        if self._running:
            return
        
        self._stream_outlet = self._create_lsl_stream()
        self._running = True
        self._stop_event.clear()
        
        self._thread = Thread(target=self._push_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"模拟LSL数据源已启动: {self.name}")
    
    def stop(self):
        """停止数据推送"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info("模拟LSL数据源已停止")
    
    def _push_loop(self):
        """数据推送循环"""
        interval = 1.0 / self.sampling_rate
        last_push = time.time()
        
        while not self._stop_event.is_set():
            # 生成一个样本
            sample = self._generate_sample()
            
            # 推送
            if self._stream_outlet:
                self._stream_outlet.push_sample(sample)
            
            # 等待到下一个采样时刻
            sleep_time = interval - (time.time() - last_push)
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_push = time.time()
    
    def _generate_sample(self) -> List[float]:
        """生成单个EEG样本"""
        sample = []
        
        for ch in range(self.n_channels):
            # 混合多频段正弦波
            value = 0
            for freq, amp in zip(self._frequencies, self._amplitudes):
                # 添加通道间相位差
                phase_offset = ch * 0.1
                value += amp * np.sin(2 * np.pi * freq * self._phase[ch] + phase_offset)
            
            # 添加噪声
            value += 0.5 * np.random.randn()
            
            sample.append(value)
        
        # 更新相位
        self._phase += 1.0 / self.sampling_rate
        
        return sample
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# ============================================================================
# 便捷函数
# ============================================================================

def discover_eeg_streams(timeout: float = 5.0) -> List[StreamInfo]:
    """
    便捷函数：发现所有EEG数据流
    
    Parameters
    ----------
    timeout : float
        发现超时时间
        
    Returns
    -------
    streams : List[StreamInfo]
    """
    discovery = LSLStreamDiscovery(timeout=timeout)
    return discovery.discover(stream_type='EEG')


def create_adapter(
    stream_name: Optional[str] = None,
    **kwargs
) -> LSLAdapter:
    """
    便捷函数：创建并启动LSL适配器
    
    Parameters
    ----------
    stream_name : str, optional
        流名称
    **kwargs
        传递给LSLAdapter的其他参数
        
    Returns
    -------
    adapter : LSLAdapter
    """
    adapter = LSLAdapter(stream_names=[stream_name] if stream_name else None, **kwargs)
    adapter.start()
    return adapter


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MetaBCI Brainflow - LSL数据采集适配器演示")
    print("=" * 60)
    
    # 1. 发现可用数据流
    print("\n[1] 正在发现局域网内的EEG数据流...")
    streams = discover_eeg_streams(timeout=3.0)
    
    if streams:
        print(f"发现 {len(streams)} 个EEG数据流:")
        for s in streams:
            print(f"  - {s.name}: {s.n_channels}通道, {s.sampling_rate}Hz")
    else:
        print("未发现EEG数据流，将使用模拟数据源")
    
    # 2. 创建适配器
    print("\n[2] 创建数据采集适配器...")
    adapter = LSLAdapter(stream_names=[s.name for s in streams] if streams else None)
    
    # 3. 启动采集
    print("[3] 启动数据采集（5秒）...")
    adapter.start()
    
    # 4. 获取数据
    import time
    time.sleep(2)
    
    data = adapter.get_latest(n_samples=500)
    if data is not None and data.size > 0:
        print(f"获取到数据: shape={data.shape}")
    else:
        print("未能获取数据")
    
    # 5. 停止采集
    adapter.stop()
    
    print("\n演示完成!")
