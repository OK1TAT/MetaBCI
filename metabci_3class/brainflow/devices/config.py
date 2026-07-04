# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 设备配置管理

提供Stroke脑电采集系统的完整配置参数管理，支持WiFi Shield、TCP、UDP三种连接协议。
配置参数覆盖网络设备、数据格式、显示保存、线程队列、Socket重连等各个方面。

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


# ============================================================================
# 常量定义
# ============================================================================

# OpenBCI Cyton 增益比例因子 (uV/count)
CYTON_GAIN_X24_SCALE_FACTOR = 0.022351744455307063
"""Cyton增益x24时的比例因子（微伏/计数）"""

CYTON_GAIN_X6_SCALE_FACTOR = CYTON_GAIN_X24_SCALE_FACTOR * 4.0
"""Cyton增益x6时的比例因子（微伏/计数）"""

DEFAULT_SCALE_FACTOR = CYTON_GAIN_X6_SCALE_FACTOR
"""默认比例因子（对应x6增益）"""

# 各增益对应的比例因子映射表
GAIN_SCALE_FACTORS: Dict[str, float] = {
    f"x{gain}": CYTON_GAIN_X24_SCALE_FACTOR * (24.0 / gain)
    for gain in (1, 2, 4, 6, 8, 12, 24)
}
"""增益 -> 比例因子映射表"""

# OpenBCI通道选择器字符映射
OPENBCI_CHANNEL_SELECTORS = "12345678QWERTYUI"
"""OpenBCI 16通道选择器字符（1-8对应通道1-8，Q-Y对应通道9-16）"""

# OpenBCI增益编码映射
OPENBCI_GAIN_CODES: Dict[str, str] = {
    "x1": "0", "x2": "1", "x4": "2", "x6": "3",
    "x8": "4", "x12": "5", "x24": "6",
}
"""增益名称 -> 设备命令编码映射"""


@dataclass
class DeviceConfig:
    """
    脑电采集设备配置类
    
    管理Stroke EEG采集系统的所有配置参数，包括：
    - 网络连接参数（主机、端口、协议）
    - 设备数据格式（通道数、采样率、增益、字节序）
    - 显示与保存参数
    - 线程队列大小
    - Socket与重连参数
    - WiFi Shield专用参数
    
    Attributes
    ----------
    host : str
        设备IP地址（wifi_shield/tcp模式）或绑定地址（udp模式）
    port : int
        TCP/UDP数据端口
    protocol : str
        连接协议类型：'tcp' | 'udp' | 'wifi_shield'
    channels : int
        显示/保存通道数（8或16）
    device_channels : int or None
        设备实际通道数（从Shield /board接口自动检测）
    force_input_channels : int
        强制解码器输入通道数：0=自动检测，8=单包模式，16=双包配对模式
    sample_rate : int
        采样率（Hz）
    endian : str
        字节序：'auto' | 'big' | 'little'
    scale_factor : float
        电压比例因子（uV/count）
    display_seconds : float
        显示窗口时长（秒）
    ui_fps : int
        UI刷新帧率
    save_dir : str
        数据保存目录
    subject : str
        被试标识
    save_npz : bool
        是否同时保存NPZ压缩格式
    marker_sync_delay_s : float
        事件标记同步延迟（秒），用于等待标记追上数据
    raw_queue_size : int
        原始数据队列容量
    plot_queue_size : int
        绘图数据队列容量
    save_queue_size : int
        保存数据队列容量
    save_batch_size : int
        保存批处理大小
    max_plot_drain_per_tick : int
        每次UI刷新的最大绘图队列消耗量
    socket_timeout : float
        Socket超时时间（秒）
    reconnect_interval : float
        重连间隔（秒）
    udp_disconnect_timeout : float
        UDP无数据超时判定时间（秒）
    wifi_http_port : int
        WiFi Shield HTTP控制端口
    wifi_local_ip : str
        本机IP地址（用于WiFi Shield回连，空串则自动检测）
    wifi_output : str
        WiFi输出格式：'raw' | 'json'
    wifi_delimiter : bool
        WiFi数据是否使用分隔符
    wifi_latency_us : int
        WiFi延迟（微秒）
    wifi_accept_timeout : float
        WiFi Shield回连等待超时（秒）
    wifi_apply_channel_settings : bool
        是否在流开始前发送通道配置命令
    wifi_channel_gain : str
        WiFi通道增益设置
    wifi_gain_command : str
        设备特定的自定义增益命令
    """
    
    # ---- 网络参数 ----
    host: str = "192.168.4.1"
    port: int = 9000
    protocol: str = "wifi_shield"  # tcp / udp / wifi_shield
    
    # ---- 设备数据格式 ----
    channels: int = 16  # 8 or 16
    device_channels: Optional[int] = None  # 从shield /board自动检测
    force_input_channels: int = 16  # 0=自动, 8=单包, 16=双包配对
    sample_rate: int = 500
    endian: str = "big"  # auto / big / little
    scale_factor: float = DEFAULT_SCALE_FACTOR
    
    # ---- 显示与保存 ----
    display_seconds: float = 8.0
    ui_fps: int = 25
    save_dir: str = "records"
    subject: str = "subject01"
    save_npz: bool = True
    marker_sync_delay_s: float = 0.25
    
    # ---- 线程队列 ----
    raw_queue_size: int = 5000
    plot_queue_size: int = 4000
    save_queue_size: int = 12000
    save_batch_size: int = 200
    max_plot_drain_per_tick: int = 1500
    
    # ---- Socket与重连 ----
    socket_timeout: float = 1.0
    reconnect_interval: float = 2.0
    udp_disconnect_timeout: float = 3.0
    
    # ---- WiFi Shield专用 ----
    wifi_http_port: int = 80
    wifi_local_ip: str = ""
    wifi_output: str = "raw"  # raw or json
    wifi_delimiter: bool = False
    wifi_latency_us: int = 10000
    wifi_accept_timeout: float = 20.0
    wifi_apply_channel_settings: bool = True
    wifi_channel_gain: str = "x6"
    wifi_gain_command: str = "1"
    
    def validate(self) -> None:
        """
        验证并规范化所有配置参数
        
        将字符串参数统一转为小写，检查各参数的合法范围。
        不合法时抛出 ValueError。
        
        Raises
        ------
        ValueError
            当任一配置参数不在合法范围内时抛出
        """
        self.protocol = self.protocol.lower()
        self.endian = self.endian.lower()
        self.wifi_output = self.wifi_output.lower()
        self.wifi_channel_gain = self.wifi_channel_gain.lower()
        
        if self.protocol not in {"tcp", "udp", "wifi_shield"}:
            raise ValueError("protocol 必须为 tcp, udp 或 wifi_shield")
        if self.channels not in {8, 16}:
            raise ValueError("channels 必须为 8 或 16")
        if self.device_channels is not None and self.device_channels not in {8, 16}:
            raise ValueError("device_channels 必须为 8 或 16")
        if self.force_input_channels not in {0, 8, 16}:
            raise ValueError("force_input_channels 必须为 0, 8 或 16")
        if self.endian not in {"auto", "big", "little"}:
            raise ValueError("endian 必须为 auto, big 或 little")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须 > 0")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port 必须在 1..65535 范围内")
        if self.wifi_http_port <= 0 or self.wifi_http_port > 65535:
            raise ValueError("wifi_http_port 必须在 1..65535 范围内")
        if self.wifi_output not in {"raw", "json"}:
            raise ValueError("wifi_output 必须为 raw 或 json")
        if self.wifi_channel_gain not in {"x1", "x2", "x4", "x6", "x8", "x12", "x24"}:
            raise ValueError("wifi_channel_gain 必须为 x1~x24 之一")
        if self.wifi_latency_us < 50:
            raise ValueError("wifi_latency_us 必须 >= 50")
        if self.wifi_accept_timeout <= 0:
            raise ValueError("wifi_accept_timeout 必须 > 0")
        if self.display_seconds <= 0:
            raise ValueError("display_seconds 必须 > 0")
        if self.ui_fps <= 0:
            raise ValueError("ui_fps 必须 > 0")
        if self.raw_queue_size <= 0 or self.plot_queue_size <= 0 or self.save_queue_size <= 0:
            raise ValueError("队列容量必须 > 0")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor 必须 > 0")
        if self.save_batch_size <= 0:
            raise ValueError("save_batch_size 必须 > 0")
        if self.marker_sync_delay_s < 0:
            raise ValueError("marker_sync_delay_s 必须 >= 0")
        
        self.save_dir = str(Path(self.save_dir))
    
    @property
    def buffer_samples(self) -> int:
        """缓冲区可容纳的样本数（= 采样率 × 显示秒数）"""
        return max(1, int(self.sample_rate * self.display_seconds))
    
    @classmethod
    def from_cli(cls) -> "DeviceConfig":
        """
        从命令行参数解析配置
        
        解析 sys.argv 中的所有参数并构建 DeviceConfig 实例，
        解析完成后自动调用 validate() 进行参数校验。
        
        Returns
        -------
        config : DeviceConfig
            解析并校验后的配置实例
        """
        parser = argparse.ArgumentParser(
            description="Stroke EEG 脑电采集系统 (WiFi/TCP/UDP)"
        )
        # 网络参数
        parser.add_argument("--host", type=str, default="192.168.4.1",
                            help="设备IP（wifi_shield/tcp）或绑定地址（udp）")
        parser.add_argument("--port", type=int, default=9000, help="TCP/UDP数据端口")
        parser.add_argument("--protocol", type=str, default="wifi_shield",
                            choices=["tcp", "udp", "wifi_shield"])
        
        # 设备数据格式
        parser.add_argument("--channels", type=int, default=16, choices=[8, 16])
        parser.add_argument("--sample-rate", type=int, default=500)
        parser.add_argument("--endian", type=str, default="big",
                            choices=["auto", "big", "little"])
        parser.add_argument("--force-input-channels", type=int, default=16,
                            choices=[0, 8, 16],
                            help="0=自动检测, 8=单包模式, 16=双包配对模式")
        parser.add_argument("--scale-factor", type=float, default=None,
                            help="uV/count比例因子，默认根据WiFi通道增益自动选择")
        
        # 显示与保存
        parser.add_argument("--display-seconds", type=float, default=8.0)
        parser.add_argument("--save-dir", type=str, default="records")
        parser.add_argument("--subject", type=str, default="subject01")
        parser.add_argument("--ui-fps", type=int, default=25)
        parser.add_argument("--save-batch-size", type=int, default=200)
        
        # Socket与重连
        parser.add_argument("--reconnect-interval", type=float, default=2.0)
        parser.add_argument("--socket-timeout", type=float, default=1.0)
        parser.add_argument("--udp-disconnect-timeout", type=float, default=3.0)
        
        # WiFi Shield专用
        parser.add_argument("--wifi-http-port", type=int, default=80)
        parser.add_argument("--wifi-local-ip", type=str, default="")
        parser.add_argument("--wifi-output", type=str, default="raw",
                            choices=["raw", "json"])
        parser.add_argument("--wifi-delimiter", action="store_true")
        parser.add_argument("--wifi-latency-us", type=int, default=10000)
        parser.add_argument("--wifi-accept-timeout", type=float, default=20.0)
        parser.add_argument("--wifi-channel-gain", type=str, default="x6",
                            choices=["x1", "x2", "x4", "x6", "x8", "x12", "x24"],
                            help="WiFi通道增益")
        parser.add_argument("--wifi-gain-command", type=str, default="1",
                            help="设备特定自定义增益命令，空串表示跳过")
        parser.add_argument("--no-wifi-channel-settings",
                            dest="wifi_apply_channel_settings",
                            action="store_false",
                            help="不发送OpenBCI通道设置命令")
        
        # 线程队列
        parser.add_argument("--raw-queue-size", type=int, default=5000)
        parser.add_argument("--plot-queue-size", type=int, default=4000)
        parser.add_argument("--save-queue-size", type=int, default=12000)
        parser.add_argument("--max-plot-drain-per-tick", type=int, default=1500)
        
        # NPZ保存
        parser.add_argument("--save-npz", action="store_true")
        parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")
        parser.set_defaults(save_npz=True, wifi_apply_channel_settings=True)
        
        args = parser.parse_args()
        
        config = cls(
            host=args.host,
            port=args.port,
            protocol=args.protocol,
            channels=args.channels,
            sample_rate=args.sample_rate,
            endian=args.endian,
            force_input_channels=args.force_input_channels,
            scale_factor=(
                args.scale_factor
                if args.scale_factor is not None
                else GAIN_SCALE_FACTORS[args.wifi_channel_gain]
            ),
            display_seconds=args.display_seconds,
            ui_fps=args.ui_fps,
            save_dir=args.save_dir,
            subject=args.subject,
            save_npz=args.save_npz,
            save_batch_size=args.save_batch_size,
            reconnect_interval=args.reconnect_interval,
            socket_timeout=args.socket_timeout,
            udp_disconnect_timeout=args.udp_disconnect_timeout,
            wifi_http_port=args.wifi_http_port,
            wifi_local_ip=args.wifi_local_ip,
            wifi_output=args.wifi_output,
            wifi_delimiter=args.wifi_delimiter,
            wifi_latency_us=args.wifi_latency_us,
            wifi_accept_timeout=args.wifi_accept_timeout,
            wifi_apply_channel_settings=args.wifi_apply_channel_settings,
            wifi_channel_gain=args.wifi_channel_gain,
            wifi_gain_command=args.wifi_gain_command,
            raw_queue_size=args.raw_queue_size,
            plot_queue_size=args.plot_queue_size,
            save_queue_size=args.save_queue_size,
            max_plot_drain_per_tick=args.max_plot_drain_per_tick,
        )
        config.validate()
        return config


# ============================================================================
# 便捷函数
# ============================================================================

def create_config(protocol: str = "wifi_shield", **kwargs) -> DeviceConfig:
    """
    便捷函数：创建设备配置
    
    Parameters
    ----------
    protocol : str
        连接协议
    **kwargs
        传递给 DeviceConfig 的其他参数
    
    Returns
    -------
    config : DeviceConfig
        校验后的配置实例
    """
    config = DeviceConfig(protocol=protocol, **kwargs)
    config.validate()
    return config
