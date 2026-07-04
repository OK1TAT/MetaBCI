# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 数据包解码器

将来自Stroke EEG设备的33字节原始数据包解析为多通道EEG样本。
支持8通道单包模式和16通道双包配对模式，包含24位有符号整数解码、
scale_factor电压转换、包连续性追踪和重复包去重逻辑。

数据帧格式（33字节）:
    Byte 0:   0xA0 帧头
    Byte 1:   Sample Number（0-255循环）
    Byte 2-25: 8通道 × 3字节 = 24位有符号整数（大端序）
    Byte 26-31: Aux Data（6字节保留位）
    Byte 32:  0xC0 帧尾

16通道配对规则:
    两个连续的33字节包，具有相同的sample_number，
    第一个包携带通道1-8数据，第二个包携带通道9-16数据。

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List, Callable

from .config import DeviceConfig
from .utils import DecodedSample, RawChunk, RuntimeStats, put_with_drop_oldest


# ============================================================================
# 24位整数解码
# ============================================================================

def decode_int24(data3: bytes, endian: str = "big") -> int:
    """
    将3个字节解码为24位有符号整数
    
    协议规定EEG载荷为MSB优先（大端序）的有符号24位整数。
    负数通过二进制补码表示：当最高字节 >= 0x80 时，
    实际值 = 无符号值 - 2^24。
    
    Parameters
    ----------
    data3 : bytes
        恰好3字节的原始数据
    endian : str
        字节序，'big'（默认，协议标准）或 'little'
    
    Returns
    -------
    value : int
        24位有符号整数值
    
    Raises
    ------
    ValueError
        当输入不是恰好3字节时
    """
    if len(data3) != 3:
        raise ValueError("decode_int24 需要恰好3字节输入")
    if endian not in {"big", "little"}:
        raise ValueError("endian 必须为 'big' 或 'little'")
    
    if endian == "big":
        b0, b1, b2 = data3
    else:
        b2, b1, b0 = data3
    
    unsigned = (b0 << 16) | (b1 << 8) | b2
    # 24位补码：b0 < 0x80 为正数（或零），b0 >= 0x80 为负数
    if b0 < 0x80:
        return unsigned
    return unsigned - (1 << 24)


def encode_int24(value: int, endian: str = "big") -> bytes:
    """
    将整数编码为3字节24位表示（模拟器使用）
    
    Parameters
    ----------
    value : int
        待编码的整数值（需在 -8388608 ~ 8388607 范围内）
    endian : str
        字节序，'big' 或 'little'
    
    Returns
    -------
    data : bytes
        3字节编码结果
    """
    if value < 0:
        value = (1 << 24) + value
    value &= 0xFFFFFF
    if endian == "big":
        return bytes([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])
    if endian == "little":
        return bytes([value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF])
    raise ValueError("endian 必须为 'big' 或 'little'")


# ============================================================================
# 内部辅助数据类
# ============================================================================

@dataclass
class _HalfPacket:
    """16ch配对模式的半包缓存"""
    recv_timestamp: float
    sample_number: int
    counts: list[int]


@dataclass
class _SinglePacketCandidate:
    """8ch模式下的单包候选（用于去重质量选择）"""
    recv_timestamp: float
    sample_number: int
    counts: list[int]
    quality_score: float


# ============================================================================
# 数据包解码器
# ============================================================================

class PacketDecoder:
    """
    33字节EEG数据包解码器（非线程，纯逻辑层）
    
    负责：
    - 字节流重组与帧同步（0xA0帧头 + 0xC0帧尾）
    - 24位有符号整数解码（大端序）
    - 16通道双包配对（相同sample_number的两个连续包）
    - 8通道去重（字节级签名比对）
    - scale_factor电压转换（uV）
    - sample_number连续性追踪（丢包检测）
    
    本类为纯解码逻辑，不含线程管理。可在线程中通过
    feed_chunk() 方法喂入原始数据，通过 get_samples() 获取结果。
    
    Attributes
    ----------
    PACKET_SIZE : int
        数据包固定长度（33字节）
    HEADER : int
        帧头标志（0xA0）
    TAIL : int
        帧尾标志（0xC0）
    
    Examples
    --------
    >>> decoder = PacketDecoder(config)
    >>> decoder.feed_chunk(raw_chunk)
    >>> samples = decoder.get_samples()
    """
    
    PACKET_SIZE: int = 33
    HEADER: int = 0xA0
    TAIL: int = 0xC0
    
    def __init__(
        self,
        config: DeviceConfig,
        stats: Optional[RuntimeStats] = None,
    ) -> None:
        """
        初始化解码器
        
        Parameters
        ----------
        config : DeviceConfig
            设备配置（通道数、字节序、比例因子等）
        stats : RuntimeStats, optional
            运行时统计对象，不传则不统计
        """
        self.config = config
        self.stats = stats or RuntimeStats()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 字节流缓冲区
        self._buffer = bytearray()
        
        # 包连续性追踪
        self._last_packet_sample: Optional[int] = None
        
        # 16ch配对状态
        self._pending_half: Optional[_HalfPacket] = None
        
        # 8ch去重状态
        self._pending_single: Optional[_SinglePacketCandidate] = None
        self._last_packet_signature: Optional[bytes] = None
        self._duplicate_packet_count: int = 0
        
        # 调试计数器
        self._debug_logged_packets: int = 0
        self._aux_nonzero_count: int = 0
        
        # 输出样本队列（线程安全）
        self._output_samples: List[DecodedSample] = []
        self._output_lock = threading.Lock()
        
        # 确定有效字节序
        # 协议规定EEG载荷为MSB优先，auto和big统一使用大端序解码
        if config.endian == "little":
            self._effective_endian = "little"
        else:
            self._effective_endian = "big"
            if config.endian == "auto":
                self.logger.info(
                    "字节序为auto模式，协议规定为MSB优先，使用大端序解码。"
                )
    
    def _input_channels(self) -> int:
        """
        确定解码器输入通道数
        
        优先级：
        1. force_input_channels（强制指定 8 或 16）
        2. channels == 16 时返回 16（显示配置为16通道）
        3. device_channels（从Shield自动检测）
        4. channels（默认值）
        
        Returns
        -------
        n_channels : int
            解码器输入通道数（8或16）
        """
        if self.config.force_input_channels in (8, 16):
            return int(self.config.force_input_channels)
        if self.config.channels == 16:
            return 16
        if self.config.device_channels in (8, 16):
            return int(self.config.device_channels)
        return self.config.channels
    
    def feed_chunk(self, chunk: RawChunk) -> List[DecodedSample]:
        """
        喂入一个原始数据块并返回解码得到的样本
        
        将原始字节追加到内部缓冲区，然后尽可能多地解析完整帧。
        每次调用返回本次解析产生的所有 DecodedSample。
        
        Parameters
        ----------
        chunk : RawChunk
            从网络设备接收的原始数据块
        
        Returns
        -------
        samples : list[DecodedSample]
            本次解析产生的解码样本列表（可能为空）
        """
        self._buffer.extend(chunk.data)
        self._output_samples.clear()
        self._parse_buffer(chunk.recv_timestamp)
        return list(self._output_samples)
    
    def flush(self) -> List[DecodedSample]:
        """
        刷新解码器内部状态，返回残留样本
        
        在停止采集时调用，将缓冲区中残留的半包数据强制输出。
        
        Returns
        -------
        samples : list[DecodedSample]
            残留样本列表
        """
        self._output_samples.clear()
        
        if self._pending_half is not None:
            self.logger.warning("16ch模式停止时有未配对的半包数据。")
            self._pending_half = None
        
        if self._pending_single is not None:
            self._emit_sample(
                recv_timestamp=self._pending_single.recv_timestamp,
                sample_number=self._pending_single.sample_number,
                counts=self._pending_single.counts,
            )
            self._pending_single = None
        
        return list(self._output_samples)
    
    def _parse_buffer(self, default_ts: float) -> None:
        """
        从缓冲区中尽可能多地解析完整帧
        
        帧同步逻辑：
        1. 寻找 0xA0 帧头
        2. 检查第33字节是否为 0xC0 帧尾
        3. 校验通过则提取33字节送入 _handle_packet
        
        Parameters
        ----------
        default_ts : float
            默认接收时间戳
        """
        while True:
            # 缓冲区至少1字节才能检查帧头
            if len(self._buffer) < 1:
                return
            
            # 帧头同步
            if self._buffer[0] != self.HEADER:
                idx = self._buffer.find(bytes([self.HEADER]))
                if idx == -1:
                    # 没找到帧头，清空缓冲区
                    dropped = len(self._buffer)
                    self._buffer.clear()
                    self.stats.add_bad_packet(1)
                    self.logger.warning(
                        "未找到帧头，丢弃 %d 字节进行重同步。", dropped
                    )
                    return
                # 丢弃帧头之前的数据
                dropped = idx
                del self._buffer[:idx]
                self.stats.add_bad_packet(1)
                self.logger.warning(
                    "流未对齐，丢弃 %d 字节到下一个帧头。", dropped
                )
            
            # 检查是否有完整帧
            if len(self._buffer) < self.PACKET_SIZE:
                return
            
            # 提取候选帧并校验帧尾
            packet = bytes(self._buffer[:self.PACKET_SIZE])
            if packet[-1] != self.TAIL:
                # 帧尾不匹配，跳过1字节重新同步
                del self._buffer[0]
                self.stats.add_bad_packet(1)
                self.logger.warning("帧尾不匹配（期望0xC0），偏移1字节重同步。")
                continue
            
            # 帧校验通过，从缓冲区移除
            del self._buffer[:self.PACKET_SIZE]
            
            try:
                self._handle_packet(packet, default_ts)
            except Exception as exc:
                self.stats.add_bad_packet(1)
                self.stats.set_error(str(exc))
                self.logger.exception("数据包解码异常: %s", exc)
    
    def _handle_packet(self, packet: bytes, recv_timestamp: float) -> None:
        """
        处理一个完整的33字节数据包
        
        解析帧内容并根据通道模式分发给不同的处理逻辑：
        - 8ch模式：直接输出，带字节级去重
        - 16ch模式：缓存半包，配对成功后合并输出
        
        Parameters
        ----------
        packet : bytes
            完整的33字节数据包
        recv_timestamp : float
            接收时间戳
        """
        sample_number = packet[1]
        eeg_bytes = packet[2:26]
        aux_bytes = packet[26:32]
        packet_signature = packet[1:33]
        
        # Aux字节非零告警
        if aux_bytes != b"\x00\x00\x00\x00\x00\x00":
            self._aux_nonzero_count += 1
            if self._aux_nonzero_count % 200 == 1:
                self.logger.warning(
                    "Aux字节非零（第 %d 次）: %s",
                    self._aux_nonzero_count,
                    aux_bytes.hex(" "),
                )
        
        # 8ch模式下的重复包去重
        # 注意：16ch配对模式下两个半包可能字节相同（如空闲状态），不去重
        if self._input_channels() == 8:
            if self._last_packet_signature == packet_signature:
                self._duplicate_packet_count += 1
                if self._duplicate_packet_count % 2000 == 1:
                    self.logger.info(
                        "检测到重复数据包，已丢弃 %d 个。",
                        self._duplicate_packet_count,
                    )
                return
            self._last_packet_signature = packet_signature
        
        # 解码8通道ADC计数值
        active_endian = self._effective_endian
        counts = self._decode_counts(eeg_bytes, active_endian)
        
        # 调试日志（前4个包）
        if self._debug_logged_packets < 4:
            self.logger.info(
                "包调试 #%d sample=%d tail=0x%02X endian=%s input_ch=%d "
                "counts=%s hex=%s",
                self._debug_logged_packets + 1,
                sample_number,
                packet[-1],
                active_endian,
                self._input_channels(),
                counts,
                packet.hex(" "),
            )
            self._debug_logged_packets += 1
        
        # 包连续性追踪
        self._track_sample_continuity(sample_number)
        
        # 根据通道模式分发处理
        if self._input_channels() == 8:
            self._handle_single_channel_packet(
                recv_timestamp=recv_timestamp,
                sample_number=sample_number,
                counts=counts,
            )
            return
        
        # 16ch双包配对模式
        half = _HalfPacket(
            recv_timestamp=recv_timestamp,
            sample_number=sample_number,
            counts=counts,
        )
        
        if self._pending_half is None:
            # 第一个半包，缓存等待配对
            self._pending_half = half
            return
        
        first = self._pending_half
        
        if half.sample_number != first.sample_number:
            # sample_number不匹配，丢弃旧半包，用新的替代
            self.stats.add_pair_mismatch(1)
            self.logger.warning(
                "16ch配对失败：第一个包sample=%d，第二个包sample=%d。"
                "本设备使用相同sample_number的两个33字节包编码一个16ch样本。",
                first.sample_number,
                half.sample_number,
            )
            self._pending_half = half
            return
        
        # 配对成功：合并两个半包的8通道数据为16通道
        combined_counts = first.counts + half.counts
        self._emit_sample(
            recv_timestamp=half.recv_timestamp,
            sample_number=first.sample_number,
            counts=combined_counts,
        )
        self._pending_half = None
    
    def _handle_single_channel_packet(
        self,
        recv_timestamp: float,
        sample_number: int,
        counts: list[int],
    ) -> None:
        """
        处理8ch模式的单包数据
        
        部分Shield固件可能对相同sample_number发送不同载荷质量的数据包，
        因此采用"质量选择"策略：保留同一sample_number中能量最低的候选
        （能量越低表示越接近真实信号，越远离rail噪声）。
        
        Parameters
        ----------
        recv_timestamp : float
            接收时间戳
        sample_number : int
            样本编号
        counts : list[int]
            8通道ADC计数值
        """
        quality = float(sum(abs(c) for c in counts))
        
        if self._pending_single is None:
            self._pending_single = _SinglePacketCandidate(
                recv_timestamp=recv_timestamp,
                sample_number=sample_number,
                counts=counts,
                quality_score=quality,
            )
            return
        
        if sample_number == self._pending_single.sample_number:
            # 同一样本编号，选择质量更好的（能量更低的）
            if quality < self._pending_single.quality_score:
                self._pending_single = _SinglePacketCandidate(
                    recv_timestamp=recv_timestamp,
                    sample_number=sample_number,
                    counts=counts,
                    quality_score=quality,
                )
            return
        
        # 不同样本编号：先输出旧候选，再缓存新候选
        self._emit_sample(
            recv_timestamp=self._pending_single.recv_timestamp,
            sample_number=self._pending_single.sample_number,
            counts=self._pending_single.counts,
        )
        self._pending_single = _SinglePacketCandidate(
            recv_timestamp=recv_timestamp,
            sample_number=sample_number,
            counts=counts,
            quality_score=quality,
        )
    
    def _decode_counts(self, eeg_bytes: bytes, endian: str) -> list[int]:
        """
        解码8通道的24位ADC计数值
        
        Parameters
        ----------
        eeg_bytes : bytes
            24字节的EEG载荷数据（8通道 × 3字节）
        endian : str
            字节序
        
        Returns
        -------
        counts : list[int]
            8个通道的有符号整数值
        """
        counts: list[int] = []
        for i in range(8):
            start = i * 3
            data3 = eeg_bytes[start:start + 3]
            counts.append(decode_int24(data3, endian=endian))
        return counts
    
    def _track_sample_continuity(self, sample_number: int) -> None:
        """
        追踪sample_number的连续性，检测丢包
        
        sample_number在0-255之间循环。正常情况每个包递增1。
        当delta不为0且不为1时，表示有数据包丢失。
        
        Parameters
        ----------
        sample_number : int
            当前包的样本编号
        """
        if self._last_packet_sample is None:
            self._last_packet_sample = sample_number
            return
        
        delta = (sample_number - self._last_packet_sample) % 256
        
        # delta == 0：重复包或配对场景，不算丢包
        if delta == 0:
            return
        
        if delta != 1:
            missing = delta - 1
            self.stats.add_packet_gap(missing)
            self.logger.warning(
                "sample_number不连续: prev=%d curr=%d missing=%d",
                self._last_packet_sample,
                sample_number,
                missing,
            )
        
        self._last_packet_sample = sample_number
    
    def _emit_sample(
        self,
        recv_timestamp: float,
        sample_number: int,
        counts: list[int],
    ) -> None:
        """
        将解码后的样本加入输出列表
        
        将原始计数值乘以scale_factor转换为微伏电压值。
        
        Parameters
        ----------
        recv_timestamp : float
            接收时间戳
        sample_number : int
            样本编号
        counts : list[int]
            各通道ADC计数值（8或16个）
        """
        uV = [value * self.config.scale_factor for value in counts]
        sample = DecodedSample(
            recv_timestamp=recv_timestamp,
            sample_number=sample_number,
            counts=counts,
            uV=uV,
        )
        with self._output_lock:
            self._output_samples.append(sample)
        self.stats.add_decoded(1)


class PacketDecoderThread(threading.Thread):
    """
    数据包解码线程
    
    在独立线程中持续从raw_queue读取原始数据块，
    通过PacketDecoder进行解码，将结果写入plot_queue和save_queue。
    
    继承自threading.Thread，以守护线程方式运行。
    
    Parameters
    ----------
    config : DeviceConfig
        设备配置
    raw_queue : queue.Queue[RawChunk]
        原始数据输入队列
    plot_queue : queue.Queue[DecodedSample]
        绘图数据输出队列
    save_queue : queue.Queue[DecodedSample]
        保存数据输出队列
    stop_event : threading.Event
        停止信号
    stats : RuntimeStats
        运行时统计
    """
    
    def __init__(
        self,
        config: DeviceConfig,
        raw_queue: "queue.Queue[RawChunk]",
        plot_queue: "queue.Queue[DecodedSample]",
        save_queue: "queue.Queue[DecodedSample]",
        stop_event: threading.Event,
        stats: RuntimeStats,
    ) -> None:
        super().__init__(name="DecoderThread", daemon=True)
        self.config = config
        self.raw_queue = raw_queue
        self.plot_queue = plot_queue
        self.save_queue = save_queue
        self.stop_event = stop_event
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 内部解码器实例
        self._decoder = PacketDecoder(config=config, stats=stats)
    
    def run(self) -> None:
        """
        解码线程主循环
        
        持续从raw_queue读取RawChunk，喂入PacketDecoder，
        将解码结果分发到plot_queue和save_queue。
        """
        while not self.stop_event.is_set() or not self.raw_queue.empty():
            try:
                chunk = self.raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            # 解码
            samples = self._decoder.feed_chunk(chunk)
            
            # 分发到下游队列
            for sample in samples:
                put_with_drop_oldest(
                    self.plot_queue, sample, self.stats, self.logger, "plot_queue"
                )
                put_with_drop_oldest(
                    self.save_queue, sample, self.stats, self.logger, "save_queue"
                )
        
        # 线程退出前刷新残留
        remaining = self._decoder.flush()
        for sample in remaining:
            put_with_drop_oldest(
                self.plot_queue, sample, self.stats, self.logger, "plot_queue"
            )
            put_with_drop_oldest(
                self.save_queue, sample, self.stats, self.logger, "save_queue"
            )
        
        self.logger.info("解码线程已停止。")
