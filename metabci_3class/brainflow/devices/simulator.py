# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 设备模拟器

提供独立运行的EEG设备模拟器，可模拟OpenBCI WiFi Shield/Cyton板卡的数据输出。
支持TCP Server和UDP Sender两种模式，生成符合33字节协议格式的数据包。

信号特征:
- 各通道正弦波（频率6.0Hz起步，间隔0.8Hz）
- 叠加高斯白噪声（±3uV）
- 支持8ch/16ch模式
- 支持大端/小端字节序

独立运行方式:
    python -m metabci_brainflow.simulator --host 127.0.0.1 --port 9000 --protocol tcp

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import time
from typing import Callable, List, Optional

from .config import DEFAULT_SCALE_FACTOR
from .packet_decoder import encode_int24


# ============================================================================
# 数据包构建
# ============================================================================

def build_packet(sample_number: int, ch_counts: List[int], endian: str) -> bytes:
    """
    构建单个33字节EEG数据包
    
    Parameters
    ----------
    sample_number : int
        样本编号（0-255）
    ch_counts : list[int]
        8通道的ADC计数值
    endian : str
        字节序（'big' 或 'little'）
    
    Returns
    -------
    packet : bytes
        33字节数据包
    
    Raises
    ------
    ValueError
        通道数不是8时
    """
    if len(ch_counts) != 8:
        raise ValueError("build_packet 需要恰好8个通道计数值")
    
    payload = bytearray()
    payload.append(0xA0)  # 帧头
    payload.append(sample_number & 0xFF)
    
    for c in ch_counts:
        payload.extend(encode_int24(c, endian=endian))
    
    payload.extend(b"\x00" * 6)  # Aux Data（6字节保留位）
    payload.append(0xC0)  # 帧尾
    return bytes(payload)


# ============================================================================
# EEG模拟器
# ============================================================================

class EEGSimulator:
    """
    OpenBCI EEG设备模拟器
    
    模拟OpenBCI Cyton/WiFi Shield的数据输出行为，生成符合
    33字节协议格式的EEG数据包。可用于测试和解码器验证。
    
    Parameters
    ----------
    host : str
        TCP绑定地址或UDP目标地址
    port : int
        TCP端口或UDP目标端口
    protocol : str
        输出协议：'tcp'（Server模式）或 'udp'（Sender模式）
    channels : int
        通道数（8或16）
    sample_rate : int
        采样率（Hz）
    endian : str
        字节序（'big' 或 'little'）
    scale_factor : float
        比例因子（uV/count）
    
    Examples
    --------
    >>> sim = EEGSimulator(host="127.0.0.1", port=9000, protocol="tcp")
    >>> sim.run()  # 阻塞运行，等待客户端连接
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        protocol: str = "tcp",
        channels: int = 8,
        sample_rate: int = 500,
        endian: str = "big",
        scale_factor: float = DEFAULT_SCALE_FACTOR,
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.channels = channels
        self.sample_rate = sample_rate
        self.endian = endian
        self.scale_factor = scale_factor
        
        # 模拟信号状态
        self.sample_number = 0
        self.phase: List[float] = [0.0] * channels
        self.freq: List[float] = [6.0 + i * 0.8 for i in range(channels)]
        self.amp_uV: List[float] = [30.0 + i * 2.0 for i in range(channels)]
    
    def _next_counts(self) -> List[int]:
        """
        生成下一时刻各通道的ADC计数值
        
        各通道产生不同频率的正弦波叠加高斯噪声。
        
        Returns
        -------
        counts : list[int]
            各通道的ADC计数值
        """
        counts: List[int] = []
        for i in range(self.channels):
            self.phase[i] += 2.0 * math.pi * self.freq[i] / self.sample_rate
            uv = self.amp_uV[i] * math.sin(self.phase[i]) + random.uniform(-3.0, 3.0)
            counts.append(int(round(uv / self.scale_factor)))
        return counts
    
    def _next_payload(self) -> bytes:
        """
        生成下一个时刻的完整数据包
        
        8ch模式：直接生成单个33字节包。
        16ch模式：生成两个33字节包（相同sample_number配对）。
        
        Returns
        -------
        payload : bytes
            数据包字节流（33或66字节）
        """
        counts = self._next_counts()
        
        if self.channels == 8:
            pkt = build_packet(self.sample_number, counts, self.endian)
            self.sample_number = (self.sample_number + 1) % 256
            return pkt
        
        # 16ch模式：拆分为两个33字节包
        # 注意：两个包使用不同的sample_number（模拟实际设备行为）
        pkt1 = build_packet(self.sample_number, counts[:8], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        pkt2 = build_packet(self.sample_number, counts[8:16], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        return pkt1 + pkt2
    
    def run(self) -> None:
        """
        启动模拟器（阻塞运行）
        
        根据协议类型启动TCP Server或UDP Sender。
        按 Ctrl+C 退出。
        """
        if self.protocol == "tcp":
            self._run_tcp_server()
        else:
            self._run_udp_sender()
    
    def _run_tcp_server(self) -> None:
        """TCP Server模式：等待客户端连接后推送"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"[模拟器] TCP服务器监听 {self.host}:{self.port}")
        print(f"[模拟器] 通道={self.channels} 采样率={self.sample_rate}Hz 字节序={self.endian}")
        
        try:
            while True:
                conn, addr = server.accept()
                print(f"[模拟器] 客户端已连接: {addr}")
                with conn:
                    self._stream_loop(send_func=conn.sendall)
                print("[模拟器] 客户端断开，等待下一个连接...")
        except KeyboardInterrupt:
            print("\n[模拟器] 已停止。")
        finally:
            server.close()
    
    def _run_udp_sender(self) -> None:
        """UDP Sender模式：向目标地址持续推送"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[模拟器] UDP发送目标 {self.host}:{self.port}")
        print(f"[模拟器] 通道={self.channels} 采样率={self.sample_rate}Hz 字节序={self.endian}")
        
        try:
            self._stream_loop(
                send_func=lambda b: sock.sendto(b, (self.host, self.port))
            )
        except KeyboardInterrupt:
            print("\n[模拟器] 已停止。")
        finally:
            sock.close()
    
    def _stream_loop(self, send_func: Callable[[bytes], None]) -> None:
        """
        定时推送循环
        
        按照采样率精确控制发送间隔。
        
        Parameters
        ----------
        send_func : Callable
            发送函数，签名为 send_func(data: bytes)
        """
        next_send = time.perf_counter()
        dt = 1.0 / self.sample_rate
        packet_count = 0
        
        while True:
            payload = self._next_payload()
            try:
                send_func(payload)
            except Exception as exc:
                print(f"[模拟器] 发送失败: {exc}")
                break
            
            packet_count += 1
            if packet_count % 500 == 0:
                print(f"[模拟器] 已发送 {packet_count} 个数据包")
            
            next_send += dt
            sleep_s = next_send - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_send = time.perf_counter()


# ============================================================================
# 命令行入口
# ============================================================================

def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Stroke EEG 设备模拟器"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TCP绑定地址或UDP目标地址")
    parser.add_argument("--port", type=int, default=9000,
                        help="TCP端口或UDP目标端口")
    parser.add_argument("--protocol", choices=["tcp", "udp"], default="tcp",
                        help="输出协议")
    parser.add_argument("--channels", choices=[8, 16], type=int, default=8,
                        help="通道数")
    parser.add_argument("--sample-rate", type=int, default=500,
                        help="采样率（Hz）")
    parser.add_argument("--endian", choices=["big", "little"], default="big",
                        help="字节序")
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR,
                        help="比例因子（uV/count）")
    args = parser.parse_args()
    
    sim = EEGSimulator(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        channels=args.channels,
        sample_rate=args.sample_rate,
        endian=args.endian,
        scale_factor=args.scale_factor,
    )
    sim.run()


if __name__ == "__main__":
    main()
