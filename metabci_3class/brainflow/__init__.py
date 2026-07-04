# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块

提供实时脑电数据采集、缓冲、预处理和特征提取功能。

主要组件:
- LSLAdapter: LSL实时数据采集适配器
- OnlineFeaturePipeline: 在线滑动窗口特征更新管道
- EfficientRingBuffer: 高效环形缓冲区
- OnlinePreprocessor: 在线预处理管道
- ClassifierManager: 在线分类器管理与模型热更新

作者: 抑郁症EEG认知障碍识别项目组
版本: 1.0.0
"""

from __future__ import annotations

# 版本信息
__version__ = '1.0.0'
__author__ = '抑郁症EEG认知障碍识别项目组'

# 导入主要类和函数
from .lsl_adapter import (
    LSLAdapter,
    LSLStreamDiscovery,
    StreamInfo,
    SyntheticLSLSource,
    discover_eeg_streams,
    create_adapter,
    PYLSL_AVAILABLE,
)

from .online_feature_pipeline import (
    OnlineFeaturePipeline,
    FeatureSample,
    create_pipeline,
)

from .ring_buffer import (
    EfficientRingBuffer,
    MultiChannelRingBuffer,
    BufferStatus,
    create_buffer,
)

from .online_preprocessing import (
    OnlinePreprocessor,
    CascadePreprocessor,
    FilterState,
    ChannelQuality,
    create_preprocessor,
)

# 模块级别配置
ENABLE_LOGGING = True

def set_logging(enabled: bool):
    """设置模块日志开关"""
    global ENABLE_LOGGING
    ENABLE_LOGGING = enabled

# 便捷别名
LSL_Adapter = LSLAdapter
Feature_Pipeline = OnlineFeaturePipeline
Ring_Buffer = EfficientRingBuffer
Preprocessor = OnlinePreprocessor

# 导出的公共接口
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # LSL适配器
    'LSLAdapter',
    'LSLStreamDiscovery', 
    'StreamInfo',
    'SyntheticLSLSource',
    'discover_eeg_streams',
    'create_adapter',
    
    # 在线特征管道
    'OnlineFeaturePipeline',
    'FeatureSample',
    'create_pipeline',
    
    # 环形缓冲区
    'EfficientRingBuffer',
    'MultiChannelRingBuffer', 
    'BufferStatus',
    'create_buffer',
    
    # 在线预处理器
    'OnlinePreprocessor',
    'CascadePreprocessor',
    'FilterState',
    'ChannelQuality',
    'create_preprocessor',
    
    # 便捷别名
    'LSL_Adapter',
    'Feature_Pipeline',
    'Ring_Buffer',
    'Preprocessor',
    
    # 配置
    'set_logging',
    'ENABLE_LOGGING',
]
