# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 数据保存管理

提供实验数据的持久化保存功能，支持三种保存格式：
- CSV：可读性强的采样数据文本格式
- NPZ：numpy压缩二进制格式，适合后续Python分析
- Raw Bin：原始字节流+索引文件，用于数据溯源和重放

所有格式均支持事件标记同步，CSV中每行可附带标签，
同时生成独立的.events.csv文件记录完整事件时间线。

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

from .config import DeviceConfig
from .utils import DecodedSample, EventMarker, RawChunk, RuntimeStats


# ============================================================================
# 数据保存线程
# ============================================================================

class DataSaverThread(threading.Thread):
    """
    数据保存线程（CSV + NPZ格式）
    
    从save_queue读取解码后的EEG样本，以批量方式写入CSV文件。
    同时支持生成NPZ压缩文件。事件标记通过marker_queue同步，
    延迟 marker_sync_delay_s 秒再写入，确保标记追上数据。
    
    CSV文件列结构:
        recv_timestamp, sample_number, ch1_uV, ch2_uV, ..., chN_uV, label, event_marker
    
    Parameters
    ----------
    config : DeviceConfig
        设备配置
    save_queue : queue.Queue[DecodedSample]
        待保存的解码样本队列
    marker_queue : queue.Queue[EventMarker]
        事件标记队列
    stop_event : threading.Event
        停止信号
    stats : RuntimeStats
        运行时统计
    session_stem : str
        会话文件名前缀
    """
    
    def __init__(
        self,
        config: DeviceConfig,
        save_queue: "queue.Queue[DecodedSample]",
        marker_queue: "queue.Queue[EventMarker]",
        stop_event: threading.Event,
        stats: RuntimeStats,
        session_stem: str,
    ) -> None:
        super().__init__(name="SaverThread", daemon=True)
        self.config = config
        self.save_queue = save_queue
        self.marker_queue = marker_queue
        self.stop_event = stop_event
        self.stats = stats
        self.session_stem = session_stem
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 事件标记缓存
        self._pending_markers: List[EventMarker] = []
        self._current_label: str = ""
    
    def run(self) -> None:
        """
        保存线程主循环
        
        1. 创建输出目录
        2. 打开CSV和events文件
        3. 循环读取样本，批量写入
        4. 线程退出前刷新残留数据和标记
        5. 生成NPZ文件（如果启用）
        """
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        stem = self.session_stem
        csv_path = save_path / f"{stem}.csv"
        events_path = save_path / f"{stem}.events.csv"
        npz_path = save_path / f"{stem}.npz"
        
        # 构建CSV表头
        numeric_header = ["recv_timestamp", "sample_number"]
        numeric_header.extend(
            [f"ch{i + 1}_uV" for i in range(self.config.channels)]
        )
        header = [*numeric_header, "label", "event_marker"]
        
        npz_cache: List[List[float]] = []
        batch: List[DecodedSample] = []
        
        self.logger.info("保存CSV到 %s", csv_path)
        
        with (
            csv_path.open("w", newline="", encoding="utf-8") as f,
            events_path.open("w", newline="", encoding="utf-8") as events_f,
        ):
            writer = csv.writer(f)
            event_writer = csv.writer(events_f)
            writer.writerow(header)
            event_writer.writerow(["timestamp", "label"])
            
            self.stats.set_output_paths(
                str(csv_path),
                str(npz_path) if self.config.save_npz else None,
            )
            
            last_flush_ts = time.time()
            
            while not self.stop_event.is_set() or not self.save_queue.empty():
                try:
                    sample = self.save_queue.get(timeout=0.2)
                    batch.append(sample)
                except queue.Empty:
                    pass
                
                now = time.time()
                should_flush = len(batch) >= self.config.save_batch_size
                timed_flush = batch and (now - last_flush_ts > 1.0)
                
                if should_flush or timed_flush:
                    flush_batch = self._take_ready_batch(batch)
                    if flush_batch:
                        self._flush_batch(
                            writer, event_writer, flush_batch, npz_cache
                        )
                        last_flush_ts = now
            
            # 线程退出前刷新残留
            if batch:
                self._flush_batch(writer, event_writer, batch, npz_cache)
            
            self._drain_marker_queue()
            self._write_remaining_marker_events(event_writer)
        
        # 生成NPZ文件
        if self.config.save_npz and npz_cache:
            arr = np.asarray(npz_cache, dtype=np.float64)
            np.savez_compressed(
                npz_path,
                data=arr,
                columns=np.asarray(numeric_header, dtype="U64"),
            )
            self.logger.info("已保存NPZ到 %s", npz_path)
        
        self.logger.info("保存线程已停止。")
    
    def _flush_batch(
        self,
        writer: csv.writer,
        event_writer: csv.writer,
        batch: List[DecodedSample],
        npz_cache: List[List[float]],
    ) -> None:
        """
        将一批样本写入CSV
        
        Parameters
        ----------
        writer : csv.writer
            CSV写入器
        event_writer : csv.writer
            事件CSV写入器
        batch : list[DecodedSample]
            待写入的样本批次
        npz_cache : list[list[float]]
            NPZ数据缓存（追加写入）
        """
        if not batch:
            return
        
        self._drain_marker_queue()
        
        rows: List[List[object]] = []
        for sample in batch:
            row: List[object] = [
                f"{sample.recv_timestamp:.6f}",
                sample.sample_number,
            ]
            
            # 通道电压值（补NaN到config.channels）
            padded = list(sample.uV[:self.config.channels])
            if len(padded) < self.config.channels:
                padded.extend(
                    [float("nan")] * (self.config.channels - len(padded))
                )
            row.extend(
                "nan" if np.isnan(v) else f"{v:.6f}" for v in padded
            )
            
            # 消费到当前样本时间为止的事件标记
            event_labels = self._consume_markers_until(
                sample.recv_timestamp, event_writer
            )
            row.extend([self._current_label, ";".join(event_labels)])
            rows.append(row)
            
            # NPZ缓存
            if self.config.save_npz:
                npz_row = [sample.recv_timestamp, float(sample.sample_number)]
                npz_row.extend(float(v) for v in padded)
                npz_cache.append(npz_row)
        
        writer.writerows(rows)
        self.stats.add_saved(len(batch))
        self.logger.debug("已写入 %d 行。", len(batch))
        batch.clear()
    
    def _take_ready_batch(
        self, batch: List[DecodedSample]
    ) -> List[DecodedSample]:
        """
        取出已就绪的样本批次（考虑标记同步延迟）
        
        当 marker_sync_delay_s > 0 时，最后若干秒的数据暂不输出，
        等待事件标记追上。
        
        Parameters
        ----------
        batch : list[DecodedSample]
            当前累积的样本列表
        
        Returns
        -------
        ready : list[DecodedSample]
            已就绪可写入的样本
        """
        if self.stop_event.is_set() or self.config.marker_sync_delay_s <= 0:
            ready = list(batch)
            batch.clear()
            return ready
        
        cutoff = time.time() - self.config.marker_sync_delay_s
        ready_count = 0
        while (
            ready_count < len(batch)
            and batch[ready_count].recv_timestamp <= cutoff
        ):
            ready_count += 1
        
        if ready_count <= 0:
            return []
        
        ready = batch[:ready_count]
        del batch[:ready_count]
        return ready
    
    def _drain_marker_queue(self) -> None:
        """清空标记队列到待处理列表"""
        while True:
            try:
                self._pending_markers.append(self.marker_queue.get_nowait())
            except queue.Empty:
                break
        if len(self._pending_markers) > 1:
            self._pending_markers.sort(key=lambda m: m.timestamp)
    
    def _consume_markers_until(
        self,
        sample_timestamp: float,
        event_writer: csv.writer,
    ) -> List[str]:
        """
        消费时间戳不晚于样本的事件标记
        
        Parameters
        ----------
        sample_timestamp : float
            样本时间戳
        event_writer : csv.writer
            事件文件写入器
        
        Returns
        -------
        labels : list[str]
            本样本附带的事件标签列表
        """
        event_labels: List[str] = []
        while (
            self._pending_markers
            and self._pending_markers[0].timestamp <= sample_timestamp
        ):
            marker = self._pending_markers.pop(0)
            self._current_label = marker.label
            event_labels.append(marker.label)
            event_writer.writerow(
                [f"{marker.timestamp:.6f}", marker.label]
            )
        return event_labels
    
    def _write_remaining_marker_events(
        self, event_writer: csv.writer
    ) -> None:
        """写入线程退出前残留的事件标记"""
        for marker in self._pending_markers:
            event_writer.writerow(
                [f"{marker.timestamp:.6f}", marker.label]
            )
        self._pending_markers.clear()


# ============================================================================
# 原始字节流保存线程
# ============================================================================

class RawStreamSaverThread(threading.Thread):
    """
    原始字节流保存线程
    
    将网络设备接收到的原始字节流保存为二进制文件，
    同时生成索引文件（CSV格式）记录每个chunk的时间戳和偏移量。
    可用于数据溯源、重放和故障排查。
    
    输出文件:
    - {session_stem}.raw.bin: 原始字节流
    - {session_stem}.raw_index.csv: 索引文件
    
    Parameters
    ----------
    config : DeviceConfig
        设备配置
    raw_save_queue : queue.Queue[RawChunk]
        原始数据保存队列
    stop_event : threading.Event
        停止信号
    session_stem : str
        会话文件名前缀
    """
    
    def __init__(
        self,
        config: DeviceConfig,
        raw_save_queue: "queue.Queue[RawChunk]",
        stop_event: threading.Event,
        session_stem: str,
    ) -> None:
        super().__init__(name="RawSaverThread", daemon=True)
        self.config = config
        self.raw_save_queue = raw_save_queue
        self.stop_event = stop_event
        self.session_stem = session_stem
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self) -> None:
        """
        原始数据保存主循环
        
        持续从队列读取RawChunk，追加写入二进制文件，
        同时更新索引文件。
        """
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        raw_bin_path = save_path / f"{self.session_stem}.raw.bin"
        raw_idx_path = save_path / f"{self.session_stem}.raw_index.csv"
        
        self.logger.info("保存原始字节流到 %s", raw_bin_path)
        offset = 0
        
        with raw_bin_path.open("wb") as fb,              raw_idx_path.open("w", newline="", encoding="utf-8") as fi:
            
            idx_writer = csv.writer(fi)
            idx_writer.writerow([
                "recv_timestamp", "offset_start",
                "chunk_size", "offset_end_exclusive"
            ])
            
            while not self.stop_event.is_set() or not self.raw_save_queue.empty():
                try:
                    chunk = self.raw_save_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                data = chunk.data
                size = len(data)
                if size <= 0:
                    continue
                
                start = offset
                end = start + size
                fb.write(data)
                idx_writer.writerow([
                    f"{chunk.recv_timestamp:.6f}",
                    start, size, end,
                ])
                offset = end
        
        self.logger.info(
            "原始数据保存线程已停止。总字节=%d 索引=%s",
            offset, raw_idx_path,
        )
