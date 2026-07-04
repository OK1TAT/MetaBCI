# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - Stroke EEG脑电采集系统

提供完整的Stroke脑电设备数据采集、解码、保存和分析功能。
按照MetaBCI brainflow模块的接口规范重构，支持WiFi Shield、TCP、UDP
三种连接协议，支持OpenBCI Cyton 8ch/16ch数据格式。

主要组件:
- DeviceAdapter: 设备网络适配器（对标LSLAdapter）
- PacketDecoder: 33字节数据包解码器
- AcquisitionController: 采集控制器
- DataSaverThread: CSV/NPZ数据保存
- EfficientRingBuffer: 高效环形缓冲区
- EEGSimulator: 设备模拟器
- DeviceDiscovery: 局域网设备发现

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

# 版本信息
__version__ = '1.0.0'
__author__ = 'Stroke EEG采集系统重构项目组'

# 配置模块
from .config import (
    DeviceConfig,
    CYTON_GAIN_X24_SCALE_FACTOR,
    CYTON_GAIN_X6_SCALE_FACTOR,
    DEFAULT_SCALE_FACTOR,
    GAIN_SCALE_FACTORS,
    OPENBCI_CHANNEL_SELECTORS,
    OPENBCI_GAIN_CODES,
    create_config,
)

# 工具类
from .utils import (
    RawChunk,
    DecodedSample,
    EventMarker,
    RuntimeStats,
    setup_logging,
    format_session_stem,
    put_with_drop_oldest,
)

# 数据包解码器
from .packet_decoder import (
    PacketDecoder,
    PacketDecoderThread,
    decode_int24,
    encode_int24,
)

# 设备适配器
from .device_adapter import (
    DeviceAdapter,
    DeviceDiscovery,
    DeviceInfo,
    SyntheticDeviceSource,
    create_adapter,
    discover_devices,
)

# 数据保存
from .data_saver import (
    DataSaverThread,
    RawStreamSaverThread,
)

# 环形缓冲区
from .ring_buffer import (
    EfficientRingBuffer,
    MultiChannelRingBuffer,
    BufferStatus,
    create_buffer,
)

# 采集控制器
from .acquisition_controller import (
    AcquisitionController,
)

# 设备模拟器
from .simulator import (
    EEGSimulator,
    build_packet,
)

# 模块级别配置
ENABLE_LOGGING = True

def set_logging(enabled: bool) -> None:
    """设置模块日志开关"""
    global ENABLE_LOGGING
    ENABLE_LOGGING = enabled


# 便捷别名（兼容旧接口）
Device_Adapter = DeviceAdapter
Packet_Decoder = PacketDecoder
Acquisition_Controller = AcquisitionController
Ring_Buffer = EfficientRingBuffer

# 导出的公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 配置
    'DeviceConfig',
    'CYTON_GAIN_X24_SCALE_FACTOR',
    'CYTON_GAIN_X6_SCALE_FACTOR',
    'DEFAULT_SCALE_FACTOR',
    'GAIN_SCALE_FACTORS',
    'OPENBCI_CHANNEL_SELECTORS',
    'OPENBCI_GAIN_CODES',
    'create_config',
    
    # 工具类
    'RawChunk',
    'DecodedSample',
    'EventMarker',
    'RuntimeStats',
    'setup_logging',
    'format_session_stem',
    'put_with_drop_oldest',
    
    # 数据包解码器
    'PacketDecoder',
    'PacketDecoderThread',
    'decode_int24',
    'encode_int24',
    
    # 设备适配器
    'DeviceAdapter',
    'DeviceDiscovery',
    'DeviceInfo',
    'SyntheticDeviceSource',
    'create_adapter',
    'discover_devices',
    
    # 数据保存
    'DataSaverThread',
    'RawStreamSaverThread',
    
    # 环形缓冲区
    'EfficientRingBuffer',
    'MultiChannelRingBuffer',
    'BufferStatus',
    'create_buffer',
    
    # 采集控制器
    'AcquisitionController',
    
    # 设备模拟器
    'EEGSimulator',
    'build_packet',
    
    # 便捷别名
    'Device_Adapter',
    'Packet_Decoder',
    'Acquisition_Controller',
    'Ring_Buffer',
    
    # 配置
    'set_logging',
    'ENABLE_LOGGING',
]
