# -*- coding: utf-8 -*-
"""
MetaBCI Brainflow模块 - 设备网络适配器

提供Stroke EEG设备的数据采集适配器，支持WiFi Shield、TCP、UDP三种连接协议。
对标LSLAdapter的接口风格（start/stop/is_running/data_callbacks），
同时包含设备发现、WiFi Shield HTTP配置流程和模拟数据源功能。

WiFi Shield工作流程:
    1. 获取设备信息 (GET /board)
    2. 停止已有流 (GET /stream/stop)
    3. 断开已有TCP连接 (DELETE /tcp)
    4. 发送通道增益配置 (POST /command × N)
    5. 启动本地TCP监听服务器
    6. 配置Shield回连 (POST /tcp {ip, port})
    7. 启动数据流 (GET /stream/start)
    8. 接受Shield回连并持续接收数据

作者: Stroke EEG采集系统重构项目组
版本: 1.0.0
"""

from __future__ import annotations

import json
import logging
import math
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, List, Callable, Dict

from .config import (
    DeviceConfig,
    OPENBCI_CHANNEL_SELECTORS,
    OPENBCI_GAIN_CODES,
    DEFAULT_SCALE_FACTOR,
)
from .utils import RawChunk, RuntimeStats, put_with_drop_oldest


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class DeviceInfo:
    """
    EEG设备信息
    
    描述一个局域网内发现的EEG采集设备的基本属性。
    
    Attributes
    ----------
    name : str
        设备名称
    host : str
        设备IP地址
    port : int
        数据端口
    protocol : str
        连接协议
    n_channels : int
        通道数
    sampling_rate : float
        采样率
    firmware : str
        固件版本（可选）
    board_type : str
        板卡类型（可选）
    """
    name: str
    host: str
    port: int = 9000
    protocol: str = "wifi_shield"
    n_channels: int = 16
    sampling_rate: float = 500.0
    firmware: str = ""
    board_type: str = ""


# ============================================================================
# 设备发现
# ============================================================================

class DeviceDiscovery:
    """
    EEG设备发现器
    
    在局域网内扫描可用的EEG采集设备。
    WiFi Shield设备通过HTTP /board接口获取设备信息。
    也支持手动添加已知IP的设备。
    
    Examples
    --------
    >>> discovery = DeviceDiscovery()
    >>> devices = discovery.scan_wifi_shield(subnet="192.168.4", ports=[80])
    >>> for d in devices:
    ...     print(f"{d.name} @ {d.host}:{d.port}")
    """
    
    def __init__(self, timeout: float = 2.0) -> None:
        """
        初始化设备发现器
        
        Parameters
        ----------
        timeout : float
            HTTP请求超时时间（秒）
        """
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def scan_wifi_shield(
        self,
        subnet: str = "192.168.4",
        ports: Optional[List[int]] = None,
    ) -> List[DeviceInfo]:
        """
        扫描指定子网内的WiFi Shield设备
        
        对子网内的每个IP发送 GET /board HTTP请求，
        成功返回JSON的设备即为WiFi Shield。
        
        Parameters
        ----------
        subnet : str
            子网前缀（如 "192.168.4"）
        ports : list[int], optional
            扫描的HTTP端口列表，默认 [80]
        
        Returns
        -------
        devices : list[DeviceInfo]
            发现的设备列表
        """
        if ports is None:
            ports = [80]
        
        devices: List[DeviceInfo] = []
        
        for last_octet in range(1, 255):
            host = f"{subnet}.{last_octet}"
            for port in ports:
                info = self._probe_wifi_shield(host, port)
                if info is not None:
                    devices.append(info)
                    self.logger.info(
                        "发现WiFi Shield设备: %s @ %s:%d (ch=%d, fw=%s)",
                        info.name, info.host, info.port,
                        info.n_channels, info.firmware,
                    )
        
        return devices
    
    def probe_single(self, host: str, http_port: int = 80) -> Optional[DeviceInfo]:
        """
        探测单个设备是否为WiFi Shield
        
        Parameters
        ----------
        host : str
            设备IP地址
        http_port : int
            HTTP控制端口
        
        Returns
        -------
        info : DeviceInfo or None
            设备信息，非WiFi Shield设备返回None
        """
        return self._probe_wifi_shield(host, http_port)
    
    def _probe_wifi_shield(self, host: str, http_port: int) -> Optional[DeviceInfo]:
        """向设备发送 GET /board 请求"""
        try:
            status, body = self._http_request(host, http_port, "GET", "/board")
            if status != 200:
                return None
            
            parsed = json.loads(body) if body else {}
            if not isinstance(parsed, dict):
                return None
            
            return DeviceInfo(
                name=parsed.get("board_type", "OpenBCI WiFi Shield"),
                host=host,
                port=9000,
                protocol="wifi_shield",
                n_channels=parsed.get("num_channels", 16),
                sampling_rate=parsed.get("sample_rate", 500.0),
                firmware=parsed.get("firmware_version", ""),
                board_type=parsed.get("board_type", ""),
            )
        except Exception:
            return None
    
    @staticmethod
    def _http_request(
        host: str, port: int, method: str, path: str,
        payload: Optional[dict] = None, timeout: float = 2.0,
    ) -> tuple[int, str]:
        """发送原始HTTP请求"""
        body_bytes = b""
        if payload is not None:
            body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            sock.connect((host, port))
            
            req_lines = [
                f"{method} {path} HTTP/1.1",
                f"Host: {host}",
                "Connection: close",
            ]
            if payload is not None:
                req_lines.append("Content-Type: application/json")
                req_lines.append(f"Content-Length: {len(body_bytes)}")
            
            request = ("\r\n".join(req_lines) + "\r\n\r\n").encode("utf-8") + body_bytes
            sock.sendall(request)
            
            response = bytearray()
            while True:
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    break
                if not data:
                    break
                response.extend(data)
        finally:
            try:
                sock.close()
            except OSError:
                pass
        
        text = response.decode("utf-8", errors="ignore")
        if "\r\n\r\n" in text:
            _head, body = text.split("\r\n\r\n", 1)
        else:
            _head, body = text, ""
        
        status = 0
        first_line = text.splitlines()[0] if text.splitlines() else ""
        parts = first_line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
        
        return status, body


# ============================================================================
# 设备适配器（核心类，对标LSLAdapter）
# ============================================================================

class DeviceAdapter:
    """
    EEG设备网络适配器
    
    对标LSLAdapter的接口风格，支持WiFi Shield、TCP、UDP三种协议。
    自动重连，数据通过回调函数或内部队列传递给下游。
    
    接口对齐LSLAdapter:
    - start() / stop() 启停采集
    - is_running 运行状态属性
    - data_callbacks 回调函数列表
    - register_callback() 注册回调
    
    Features:
    - 三种协议支持（WiFi Shield / TCP / UDP）
    - WiFi Shield完整HTTP配置流程
    - 自动重连机制
    - 线程安全回调
    - 运行时统计
    
    Examples
    --------
    >>> adapter = DeviceAdapter(config)
    >>> adapter.register_callback(on_data)
    >>> adapter.start()
    >>> # ... 采集进行中 ...
    >>> adapter.stop()
    """
    
    def __init__(
        self,
        config: DeviceConfig,
        stats: Optional[RuntimeStats] = None,
    ) -> None:
        """
        初始化设备适配器
        
        Parameters
        ----------
        config : DeviceConfig
            设备配置
        stats : RuntimeStats, optional
            运行时统计对象
        """
        self.config = config
        self.stats = stats or RuntimeStats()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 运行时状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # 数据回调
        self._data_callbacks: List[Callable[[bytes, float], None]] = []
        
        # 原始数据输出队列（供下游decoder消费）
        self._raw_queue: Optional["queue.Queue[RawChunk]"] = None
        self._raw_save_queue: Optional["queue.Queue[RawChunk]"] = None
    
    # ---- 属性 ----
    
    @property
    def is_running(self) -> bool:
        """检查适配器是否正在运行"""
        return self._running
    
    @property
    def data_callbacks(self) -> List[Callable]:
        """获取回调函数列表（只读副本）"""
        return list(self._data_callbacks)
    
    # ---- 队列绑定 ----
    
    def set_output_queues(
        self,
        raw_queue: "queue.Queue[RawChunk]",
        raw_save_queue: Optional["queue.Queue[RawChunk]"] = None,
    ) -> None:
        """
        绑定输出队列
        
        绑定后，接收到的数据会自动封装为 RawChunk 放入队列，
        供下游 PacketDecoder 消费。
        
        Parameters
        ----------
        raw_queue : queue.Queue[RawChunk]
            原始数据队列（必须）
        raw_save_queue : queue.Queue[RawChunk], optional
            原始数据保存队列（可选）
        """
        self._raw_queue = raw_queue
        self._raw_save_queue = raw_save_queue
    
    # ---- 回调管理 ----
    
    def register_callback(self, callback: Callable[[bytes, float], None]) -> None:
        """
        注册数据回调函数
        
        每当收到新的原始数据块时，所有已注册的回调会被调用。
        
        Parameters
        ----------
        callback : Callable[[bytes, float], None]
            回调签名: callback(data: bytes, recv_timestamp: float)
        """
        self._data_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[bytes, float], None]) -> None:
        """注销回调函数"""
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)
    
    # ---- 生命周期管理 ----
    
    def start(self) -> bool:
        """
        启动数据采集
        
        根据配置的协议启动相应的接收线程。
        
        Returns
        -------
        success : bool
            是否成功启动
        """
        if self._running:
            self.logger.warning("适配器已在运行中。")
            return True
        
        self._stop_event.clear()
        self._running = True
        
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="DeviceAdapter"
        )
        self._thread.start()
        
        self.logger.info(
            "设备适配器已启动 (protocol=%s, host=%s, port=%d)",
            self.config.protocol, self.config.host, self.config.port,
        )
        return True
    
    def stop(self) -> None:
        """
        停止数据采集
        
        发送停止信号并等待接收线程退出。
        """
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        
        self.stats.set_connected(False)
        self.logger.info("设备适配器已停止。")
    
    def get_raw_queue(self) -> Optional["queue.Queue[RawChunk]"]:
        """获取原始数据输出队列"""
        return self._raw_queue
    
    # ---- 内部接收循环 ----
    
    def _receive_loop(self) -> None:
        """根据协议分发到不同的接收逻辑"""
        if self.config.protocol == "tcp":
            self._run_tcp_client()
        elif self.config.protocol == "udp":
            self._run_udp()
        else:
            self._run_wifi_shield()
    
    def _forward_bytes(self, data: bytes) -> None:
        """
        将接收到的原始字节转发到下游
        
        封装为 RawChunk，放入绑定的队列，同时触发回调。
        
        Parameters
        ----------
        data : bytes
            原始数据
        """
        recv_ts = time.time()
        chunk = RawChunk(recv_timestamp=recv_ts, data=data)
        
        # 放入队列
        if self._raw_queue is not None:
            put_with_drop_oldest(
                self._raw_queue, chunk, self.stats, self.logger, "raw_queue"
            )
        if self._raw_save_queue is not None:
            put_with_drop_oldest(
                self._raw_save_queue, chunk, self.stats, self.logger, "raw_save_queue"
            )
        
        # 触发回调
        for callback in self._data_callbacks:
            try:
                callback(data, recv_ts)
            except Exception as exc:
                self.logger.warning("回调执行失败: %s", exc)
    
    # ================================================================
    # TCP客户端模式
    # ================================================================
    
    def _run_tcp_client(self) -> None:
        """
        TCP客户端模式接收循环
        
        连接到设备IP:Port，持续接收数据，断线自动重连。
        """
        while not self._stop_event.is_set():
            sock: Optional[socket.socket] = None
            connected_once = False
            
            try:
                self.logger.info(
                    "正在TCP连接设备 %s:%d ...",
                    self.config.host, self.config.port,
                )
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.socket_timeout)
                
                try:
                    sock.connect((self.config.host, self.config.port))
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"TCP连接超时: {self.config.host}:{self.config.port}"
                    ) from exc
                
                self.stats.set_connected(True)
                connected_once = True
                self.logger.info("TCP连接成功。")
                
                while not self._stop_event.is_set():
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("对端关闭了TCP连接。")
                    self._forward_bytes(data)
                    
            except Exception as exc:
                self.stats.set_connected(False)
                self.stats.set_error(str(exc))
                if connected_once:
                    self.logger.warning("TCP连接断开: %s", exc)
                else:
                    self.logger.warning("TCP连接失败: %s", exc)
            finally:
                self.stats.set_connected(False)
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            
            if self._stop_event.is_set():
                break
            
            self.stats.add_reconnect(1)
            self.logger.info(
                "将在 %.1f 秒后重试TCP连接 ...", self.config.reconnect_interval
            )
            time.sleep(self.config.reconnect_interval)
    
    # ================================================================
    # UDP模式
    # ================================================================
    
    def _run_udp(self) -> None:
        """
        UDP接收模式
        
        绑定本地地址，监听设备发来的UDP数据流。
        超过 udp_disconnect_timeout 无数据则判定断连。
        """
        sock: Optional[socket.socket] = None
        last_recv_ts = 0.0
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.config.host, self.config.port))
            sock.settimeout(self.config.socket_timeout)
            self.logger.info(
                "UDP监听 %s:%d", self.config.host, self.config.port
            )
            
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    if (
                        self.stats.snapshot()["connected"]
                        and time.time() - last_recv_ts > self.config.udp_disconnect_timeout
                    ):
                        self.stats.set_connected(False)
                        self.logger.warning("UDP数据流超时。")
                    continue
                except Exception as exc:
                    self.stats.set_error(str(exc))
                    self.logger.warning("UDP接收错误: %s", exc)
                    continue
                
                last_recv_ts = time.time()
                self.stats.set_connected(True)
                self._forward_bytes(data)
                
        finally:
            self.stats.set_connected(False)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
    
    # ================================================================
    # WiFi Shield模式
    # ================================================================
    
    def _run_wifi_shield(self) -> None:
        """
        WiFi Shield模式接收循环
        
        完整的WiFi Shield工作流程：
        1. 获取板卡信息
        2. 停止已有流和TCP连接
        3. 配置通道增益
        4. 启动本地TCP监听
        5. 配置Shield TCP回连
        6. 启动数据流
        7. 接受Shield回连
        8. 持续接收数据
        """
        while not self._stop_event.is_set():
            server_sock: Optional[socket.socket] = None
            data_sock: Optional[socket.socket] = None
            connected_once = False
            
            try:
                # 确定本机IP
                local_ip = self.config.wifi_local_ip.strip() or self._guess_local_ip(
                    self.config.host
                )
                self.logger.info(
                    "WiFi Shield模式: 设备=%s:%d 本机流=%s:%d",
                    self.config.host, self.config.wifi_http_port,
                    local_ip, self.config.port,
                )
                
                # 1. 获取板卡信息
                board_info = self._wifi_get_board_info()
                if board_info:
                    self.logger.info("Shield板卡信息: %s", board_info)
                    board_channels = board_info.get("num_channels")
                    if self.config.force_input_channels in (8, 16):
                        self.config.device_channels = int(self.config.force_input_channels)
                        self.logger.info(
                            "输入通道强制为 %d；忽略 /board 的 num_channels=%s。",
                            self.config.force_input_channels, board_channels,
                        )
                    elif board_channels in (8, 16):
                        self.config.device_channels = int(board_channels)
                        self.logger.info(
                            "检测到Shield板卡通道=%d（显示保持 %d）。",
                            self.config.device_channels, self.config.channels,
                        )
                
                # 2. 停止已有流和TCP连接
                try:
                    self._wifi_stream_stop()
                    self._wifi_tcp_disconnect()
                except Exception:
                    pass
                
                # 3. 配置通道增益
                if self.config.wifi_apply_channel_settings:
                    self._wifi_apply_channel_settings()
                
                # 4. 启动本地TCP监听
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind((local_ip, self.config.port))
                server_sock.listen(1)
                server_sock.settimeout(self.config.socket_timeout)
                
                # 5. 配置Shield TCP回连
                self._wifi_configure_tcp(local_ip=local_ip, local_port=self.config.port)
                
                # 6. 启动数据流
                self._wifi_stream_start()
                self.logger.info("已发送 /stream/start，等待Shield回连...")
                
                # 7. 接受Shield回连
                accept_deadline = time.time() + self.config.wifi_accept_timeout
                while not self._stop_event.is_set():
                    if time.time() > accept_deadline:
                        raise TimeoutError(
                            f"Shield在 {self.config.wifi_accept_timeout:.0f}s 内未回连到 "
                            f"{local_ip}:{self.config.port}。"
                            "请检查Windows防火墙、本机IP和Shield状态。"
                        )
                    try:
                        data_sock, peer = server_sock.accept()
                        self.logger.info(
                            "Shield流连接来自 %s:%d", peer[0], peer[1]
                        )
                        break
                    except socket.timeout:
                        continue
                
                if data_sock is None:
                    raise RuntimeError("流Socket未建立。")
                
                data_sock.settimeout(self.config.socket_timeout)
                self.stats.set_connected(True)
                connected_once = True
                
                # 8. 持续接收数据
                while not self._stop_event.is_set():
                    try:
                        data = data_sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("Shield流Socket已关闭。")
                    self._forward_bytes(data)
                    
            except Exception as exc:
                self.stats.set_connected(False)
                self.stats.set_error(str(exc))
                if connected_once:
                    self.logger.warning("Shield流断开: %s", exc)
                else:
                    self.logger.warning("Shield流建立失败: %s", exc)
            finally:
                self.stats.set_connected(False)
                try:
                    self._wifi_stream_stop()
                    self._wifi_tcp_disconnect()
                except Exception:
                    pass
                if data_sock is not None:
                    try:
                        data_sock.close()
                    except OSError:
                        pass
                if server_sock is not None:
                    try:
                        server_sock.close()
                    except OSError:
                        pass
            
            if self._stop_event.is_set():
                break
            
            self.stats.add_reconnect(1)
            self.logger.info(
                "将在 %.1f 秒后重试WiFi Shield流建立 ...",
                self.config.reconnect_interval,
            )
            time.sleep(self.config.reconnect_interval)
    
    # ================================================================
    # WiFi Shield HTTP命令
    # ================================================================
    
    def _wifi_get_board_info(self) -> Optional[dict[str, Any]]:
        """GET /board - 获取板卡信息"""
        status, body = self._wifi_http_request("GET", "/board")
        if status != 200:
            return None
        try:
            parsed = json.loads(body) if body else {}
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    
    def _wifi_configure_tcp(self, local_ip: str, local_port: int) -> None:
        """POST /tcp - 配置Shield TCP回连参数"""
        payload = {
            "ip": local_ip,
            "port": local_port,
            "output": self.config.wifi_output,
            "delimiter": bool(self.config.wifi_delimiter),
            "latency": int(self.config.wifi_latency_us),
        }
        status, body = self._wifi_http_request("POST", "/tcp", payload=payload)
        if status != 200:
            raise ConnectionError(f"/tcp 失败 status={status} body={body}")
        
        parsed: dict[str, Any] = {}
        if body.strip():
            try:
                maybe = json.loads(body)
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = {}
        
        if parsed.get("success") is False:
            raise ConnectionError(f"/tcp 返回 success=false body={body}")
        if "connected" in parsed and not bool(parsed.get("connected")):
            raise ConnectionError(
                "/tcp 已配置但Shield报告 connected=false"
                "（请检查本机IP/端口或防火墙）。"
            )
        self.logger.info("POST /tcp 成功: %s", body.strip() or "{}")
    
    def _wifi_apply_channel_settings(self) -> None:
        """配置OpenBCI通道增益"""
        gain = self.config.wifi_channel_gain.lower()
        custom_command = self.config.wifi_gain_command.strip()
        
        # 先发送设备特定的自定义命令
        if custom_command:
            self._wifi_send_board_command(custom_command)
            time.sleep(0.2)
            self.logger.info(
                "已发送设备特定增益命令: %s", custom_command,
            )
        
        # 构建通道配置命令
        gain_code = OPENBCI_GAIN_CODES[gain]
        commands = [
            self._build_channel_setting_command(selector, gain_code)
            for selector in OPENBCI_CHANNEL_SELECTORS[:self.config.channels]
        ]
        
        for command in commands:
            if self._stop_event.is_set():
                return
            self._wifi_send_board_command(command)
            time.sleep(0.02)
        
        self.logger.info(
            "已配置OpenBCI通道: channels=%d gain=%s normal bias=include "
            "srb2=connect srb1=disconnect。",
            self.config.channels, gain,
        )
    
    @staticmethod
    def _build_channel_setting_command(channel_selector: str, gain_code: str) -> str:
        """
        构建OpenBCI通道设置命令
        
        格式: x CHANNEL POWER_ON GAIN NORMAL_INPUT BIAS_INCLUDE SRB2_CONNECT SRB1_DISCONNECT X
        
        Parameters
        ----------
        channel_selector : str
            通道选择器字符（'1'-'8' 或 'Q'-'Y'）
        gain_code : str
            增益编码（'0'-'6'）
        
        Returns
        -------
        command : str
            完整的通道设置命令
        """
        return f"x{channel_selector}0{gain_code}0110X"
    
    def _wifi_send_board_command(self, command: str) -> None:
        """POST /command - 发送板卡命令"""
        payload = {"command": command}
        status, body = self._wifi_http_request("POST", "/command", payload=payload)
        if status != 200:
            raise ConnectionError(f"/command 失败 status={status} body={body}")
        
        parsed: dict[str, Any] = {}
        if body.strip():
            try:
                maybe = json.loads(body)
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = {}
        if parsed.get("success") is False:
            raise ConnectionError(f"/command 返回 success=false body={body}")
        self.logger.info("POST /command 成功: %s", body.strip() or "{}")
    
    def _wifi_stream_start(self) -> None:
        """GET /stream/start - 启动数据流"""
        status, body = self._wifi_http_request("GET", "/stream/start")
        if status != 200:
            raise ConnectionError(f"/stream/start 失败 status={status} body={body}")
        self.logger.info("GET /stream/start 成功: %s", body.strip() or "{}")
    
    def _wifi_stream_stop(self) -> None:
        """GET /stream/stop - 停止数据流"""
        try:
            status, body = self._wifi_http_request("GET", "/stream/stop")
            if status == 200:
                self.logger.info("GET /stream/stop 成功: %s", body.strip() or "{}")
        except Exception:
            pass
    
    def _wifi_tcp_disconnect(self) -> None:
        """DELETE /tcp - 断开Shield已有TCP连接"""
        try:
            status, body = self._wifi_http_request("DELETE", "/tcp")
            if status == 200:
                self.logger.info("DELETE /tcp 成功: %s", body.strip() or "{}")
        except Exception:
            pass
    
    def _wifi_http_request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> tuple[int, str]:
        """
        向WiFi Shield发送HTTP请求
        
        Parameters
        ----------
        method : str
            HTTP方法（GET/POST/DELETE）
        path : str
            URL路径
        payload : dict, optional
            JSON请求体
        
        Returns
        -------
        status : int
            HTTP状态码
        body : str
            响应体
        """
        body_bytes = b""
        if payload is not None:
            body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(max(1.0, self.config.socket_timeout + 1.0))
            sock.connect((self.config.host, self.config.wifi_http_port))
            
            req_lines = [
                f"{method} {path} HTTP/1.1",
                f"Host: {self.config.host}",
                "Connection: close",
            ]
            if payload is not None:
                req_lines.append("Content-Type: application/json")
                req_lines.append(f"Content-Length: {len(body_bytes)}")
            
            request = ("\r\n".join(req_lines) + "\r\n\r\n").encode("utf-8") + body_bytes
            sock.sendall(request)
            
            response = bytearray()
            while True:
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    break
                if not data:
                    break
                response.extend(data)
        finally:
            try:
                sock.close()
            except OSError:
                pass
        
        text = response.decode("utf-8", errors="ignore")
        if "\r\n\r\n" in text:
            _head, body = text.split("\r\n\r\n", 1)
        else:
            _head, body = text, ""
        
        status = 0
        first_line = text.splitlines()[0] if text.splitlines() else ""
        parts = first_line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
        
        return status, body
    
    @staticmethod
    def _guess_local_ip(device_ip: str) -> str:
        """
        自动检测本机IP地址
        
        通过UDP连接（不实际发送）获取本机到设备的出口IP。
        
        Parameters
        ----------
        device_ip : str
            设备IP地址
        
        Returns
        -------
        local_ip : str
            本机IP地址
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((device_ip, 80))
            local = sock.getsockname()[0]
            return local
        finally:
            sock.close()
    
    # ---- 上下文管理器 ----
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# 模拟设备源（对标SyntheticLSLSource）
# ============================================================================

class SyntheticDeviceSource:
    """
    模拟EEG设备数据源
    
    在没有真实设备时生成符合OpenBCI协议格式的模拟数据包，
    支持TCP Server和UDP Sender两种模式向外推送数据。
    
    信号特征：
    - 各通道为正弦波 + 高斯噪声
    - 各通道频率不同（6.0Hz起步，间隔0.8Hz）
    - 各通道振幅不同（30uV起步，间隔2uV）
    
    Parameters
    ----------
    host : str
        TCP绑定地址或UDP目标地址
    port : int
        TCP端口或UDP目标端口
    protocol : str
        输出协议：'tcp' 或 'udp'
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
    >>> source = SyntheticDeviceSource(host="127.0.0.1", port=9000)
    >>> source.start()
    >>> # ... 数据持续推送 ...
    >>> source.stop()
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
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 模拟信号参数
        self.sample_number = 0
        self.phase: List[float] = [0.0] * channels
        self.freq: List[float] = [6.0 + i * 0.8 for i in range(channels)]
        self.amp_uV: List[float] = [30.0 + i * 2.0 for i in range(channels)]
    
    def start(self) -> None:
        """启动模拟数据推送"""
        if self._running:
            return
        
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._push_loop, daemon=True, name="SyntheticDevice")
        self._thread.start()
        self.logger.info(
            "模拟设备已启动: %s %s:%d ch=%d sr=%d",
            self.protocol, self.host, self.port,
            self.channels, self.sample_rate,
        )
    
    def stop(self) -> None:
        """停止模拟数据推送"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        
        self.logger.info("模拟设备已停止。")
    
    def _push_loop(self) -> None:
        """数据推送主循环"""
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
        self.logger.info("[模拟] TCP服务器监听 %s:%d", self.host, self.port)
        
        try:
            while not self._stop_event.is_set():
                server.settimeout(1.0)
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                self.logger.info("[模拟] 客户端已连接: %s", addr)
                with conn:
                    self._stream_loop(send_func=conn.sendall)
                self.logger.info("[模拟] 客户端断开，等待下一个连接...")
        finally:
            server.close()
    
    def _run_udp_sender(self) -> None:
        """UDP Sender模式：向目标地址持续推送"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logger.info("[模拟] UDP发送目标 %s:%d", self.host, self.port)
        try:
            self._stream_loop(
                send_func=lambda b: sock.sendto(b, (self.host, self.port))
            )
        finally:
            sock.close()
    
    def _stream_loop(self, send_func: Callable[[bytes], None]) -> None:
        """定时推送循环"""
        next_send = time.perf_counter()
        dt = 1.0 / self.sample_rate
        
        while not self._stop_event.is_set():
            payload = self._next_payload()
            try:
                send_func(payload)
            except Exception as exc:
                self.logger.warning("[模拟] 发送失败: %s", exc)
                break
            
            next_send += dt
            sleep_s = next_send - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_send = time.perf_counter()
    
    def _next_payload(self) -> bytes:
        """生成下一个数据包的字节流"""
        counts = self._next_counts()
        
        if self.channels == 8:
            pkt = self._build_packet(self.sample_number, counts, self.endian)
            self.sample_number = (self.sample_number + 1) % 256
            return pkt
        
        # 16ch模式：拆分为两个33字节包
        pkt1 = self._build_packet(self.sample_number, counts[:8], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        pkt2 = self._build_packet(self.sample_number, counts[8:16], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        return pkt1 + pkt2
    
    def _next_counts(self) -> List[int]:
        """生成各通道的ADC计数值"""
        counts: List[int] = []
        for i in range(self.channels):
            self.phase[i] += 2.0 * math.pi * self.freq[i] / self.sample_rate
            uv = self.amp_uV[i] * math.sin(self.phase[i]) + random.uniform(-3.0, 3.0)
            counts.append(int(round(uv / self.scale_factor)))
        return counts
    
    @staticmethod
    def _build_packet(sample_number: int, ch_counts: List[int], endian: str) -> bytes:
        """
        构建单个33字节数据包
        
        Parameters
        ----------
        sample_number : int
            样本编号（0-255）
        ch_counts : list[int]
            8通道的ADC计数值
        endian : str
            字节序
        
        Returns
        -------
        packet : bytes
            33字节数据包
        """
        from .packet_decoder import encode_int24
        
        if len(ch_counts) != 8:
            raise ValueError("_build_packet 需要恰好8个通道计数值")
        
        payload = bytearray()
        payload.append(0xA0)  # 帧头
        payload.append(sample_number & 0xFF)
        
        for c in ch_counts:
            payload.extend(encode_int24(c, endian=endian))
        
        payload.extend(b"\x00" * 6)  # Aux Data
        payload.append(0xC0)  # 帧尾
        return bytes(payload)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# ============================================================================
# 便捷函数
# ============================================================================

def create_adapter(
    protocol: str = "wifi_shield",
    host: str = "192.168.4.1",
    port: int = 9000,
    **kwargs,
) -> DeviceAdapter:
    """
    便捷函数：创建设备适配器
    
    Parameters
    ----------
    protocol : str
        连接协议
    host : str
        设备地址
    port : int
        端口号
    **kwargs
        传递给 DeviceConfig 的其他参数
    
    Returns
    -------
    adapter : DeviceAdapter
        配置好的设备适配器（未启动）
    """
    config = DeviceConfig(
        protocol=protocol, host=host, port=port, **kwargs
    )
    config.validate()
    return DeviceAdapter(config)


def discover_devices(
    subnet: str = "192.168.4",
    timeout: float = 2.0,
) -> List[DeviceInfo]:
    """
    便捷函数：发现局域网内的EEG设备
    
    Parameters
    ----------
    subnet : str
        子网前缀
    timeout : float
        探测超时
    
    Returns
    -------
    devices : list[DeviceInfo]
        发现的设备列表
    """
    discovery = DeviceDiscovery(timeout=timeout)
    return discovery.scan_wifi_shield(subnet=subnet)
