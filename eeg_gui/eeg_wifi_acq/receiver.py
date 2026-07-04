from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from typing import Any

from .config import AppConfig
from .utils import RawChunk, RuntimeStats, put_with_drop_oldest


class NetworkReceiverThread(threading.Thread):
    OPENBCI_CHANNEL_SELECTORS = "12345678QWERTYUI"
    OPENBCI_GAIN_CODES = {
        "x1": "0",
        "x2": "1",
        "x4": "2",
        "x6": "3",
        "x8": "4",
        "x12": "5",
        "x24": "6",
    }

    def __init__(
        self,
        config: AppConfig,
        raw_queue: "queue.Queue[RawChunk]",
        raw_save_queue: "queue.Queue[RawChunk] | None",
        save_enabled_event: threading.Event | None,
        stop_event: threading.Event,
        stats: RuntimeStats,
    ) -> None:
        super().__init__(name="ReceiverThread", daemon=True)
        self.config = config
        self.raw_queue = raw_queue
        self.raw_save_queue = raw_save_queue
        self.save_enabled_event = save_enabled_event
        self.stop_event = stop_event
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        if self.config.protocol == "tcp":
            self._run_tcp_client()
        elif self.config.protocol == "udp":
            self._run_udp()
        else:
            self._run_wifi_shield()

    def _forward_bytes(self, data: bytes) -> None:
        chunk = RawChunk(recv_timestamp=time.time(), data=data)
        put_with_drop_oldest(
            self.raw_queue, chunk, self.stats, self.logger, "raw_queue"
        )
        if (
            self.raw_save_queue is not None
            and self.save_enabled_event is not None
            and self.save_enabled_event.is_set()
        ):
            put_with_drop_oldest(
                self.raw_save_queue, chunk, self.stats, self.logger, "raw_save_queue"
            )

    def _run_tcp_client(self) -> None:
        while not self.stop_event.is_set():
            sock: socket.socket | None = None
            connected_once = False
            try:
                self.logger.info(
                    "Connecting TCP device at %s:%d ...",
                    self.config.host,
                    self.config.port,
                )
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.socket_timeout)
                try:
                    sock.connect((self.config.host, self.config.port))
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"TCP connect timeout to {self.config.host}:{self.config.port}"
                    ) from exc
                self.stats.set_connected(True)
                connected_once = True
                self.logger.info("TCP connected.")

                while not self.stop_event.is_set():
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("Peer closed TCP connection.")
                    self._forward_bytes(data)
            except Exception as exc:
                self.stats.set_connected(False)
                self.stats.set_error(str(exc))
                if connected_once:
                    self.logger.warning("TCP disconnected: %s", exc)
                else:
                    self.logger.warning("TCP connect failed: %s", exc)
            finally:
                self.stats.set_connected(False)
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

            if self.stop_event.is_set():
                break
            self.stats.add_reconnect(1)
            self.logger.info(
                "Retrying TCP connection in %.1f s ...", self.config.reconnect_interval
            )
            time.sleep(self.config.reconnect_interval)

    def _run_udp(self) -> None:
        sock: socket.socket | None = None
        last_recv_ts = 0.0
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.config.host, self.config.port))
            sock.settimeout(self.config.socket_timeout)
            self.logger.info(
                "UDP listening on %s:%d", self.config.host, self.config.port
            )

            while not self.stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    if (
                        self.stats.snapshot()["connected"]
                        and time.time() - last_recv_ts > self.config.udp_disconnect_timeout
                    ):
                        self.stats.set_connected(False)
                        self.logger.warning("UDP stream timeout.")
                    continue
                except Exception as exc:
                    self.stats.set_error(str(exc))
                    self.logger.warning("UDP receive error: %s", exc)
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

    def _run_wifi_shield(self) -> None:
        """
        OpenBCI WiFi Shield flow:
        1) open local TCP server on host PC
        2) POST /tcp with local ip/port
        3) GET /stream/start
        4) accept incoming TCP stream and forward bytes
        """
        while not self.stop_event.is_set():
            server_sock: socket.socket | None = None
            data_sock: socket.socket | None = None
            connected_once = False
            try:
                local_ip = self.config.wifi_local_ip.strip() or self._guess_local_ip(
                    self.config.host
                )
                self.logger.info(
                    "WiFi shield mode: device=%s:%d local_stream=%s:%d",
                    self.config.host,
                    self.config.wifi_http_port,
                    local_ip,
                    self.config.port,
                )

                board_info = self._wifi_get_board_info()
                if board_info:
                    self.logger.info("Shield board info: %s", board_info)
                    board_channels = board_info.get("num_channels")
                    if self.config.force_input_channels in (8, 16):
                        self.config.device_channels = int(self.config.force_input_channels)
                        self.logger.info(
                            "Input channels are forced to %d; ignoring /board num_channels=%s.",
                            self.config.force_input_channels,
                            board_channels,
                        )
                    elif board_channels in (8, 16):
                        self.config.device_channels = int(board_channels)
                        self.logger.info(
                            "Detected shield board channels=%d (display remains %d).",
                            self.config.device_channels,
                            self.config.channels,
                        )

                try:
                    self._wifi_stream_stop()
                    self._wifi_tcp_disconnect()
                except Exception:
                    pass

                if self.config.wifi_apply_channel_settings:
                    self._wifi_apply_channel_settings()

                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind((local_ip, self.config.port))
                server_sock.listen(1)
                server_sock.settimeout(self.config.socket_timeout)

                self._wifi_configure_tcp(local_ip=local_ip, local_port=self.config.port)
                self._wifi_stream_start()
                self.logger.info(
                    "Configured /tcp and sent /stream/start, waiting for incoming stream..."
                )

                accept_deadline = time.time() + self.config.wifi_accept_timeout
                while not self.stop_event.is_set():
                    if time.time() > accept_deadline:
                        raise TimeoutError(
                            f"No incoming stream from shield within {self.config.wifi_accept_timeout:.0f}s. "
                            f"The shield accepted /tcp, but did not connect back to "
                            f"{local_ip}:{self.config.port}. Check Windows inbound firewall, "
                            "local IP, and whether another application is using the shield."
                        )
                    try:
                        data_sock, peer = server_sock.accept()
                        self.logger.info("Shield stream connected from %s:%d", peer[0], peer[1])
                        break
                    except socket.timeout:
                        continue

                if data_sock is None:
                    raise RuntimeError("Stream socket is not established.")
                data_sock.settimeout(self.config.socket_timeout)
                self.stats.set_connected(True)
                connected_once = True

                while not self.stop_event.is_set():
                    try:
                        data = data_sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("Shield stream socket closed.")
                    self._forward_bytes(data)
            except Exception as exc:
                self.stats.set_connected(False)
                self.stats.set_error(str(exc))
                if connected_once:
                    self.logger.warning("Shield stream disconnected: %s", exc)
                else:
                    self.logger.warning("Shield stream setup failed: %s", exc)
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

            if self.stop_event.is_set():
                break
            self.stats.add_reconnect(1)
            self.logger.info(
                "Retrying WiFi shield stream setup in %.1f s ...",
                self.config.reconnect_interval,
            )
            time.sleep(self.config.reconnect_interval)

    def _wifi_get_board_info(self) -> dict[str, Any] | None:
        status, body = self._http_request("GET", "/board")
        if status != 200:
            return None
        try:
            parsed = json.loads(body) if body else {}
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _wifi_configure_tcp(self, local_ip: str, local_port: int) -> None:
        payload = {
            "ip": local_ip,
            "port": local_port,
            "output": self.config.wifi_output,
            "delimiter": bool(self.config.wifi_delimiter),
            "latency": int(self.config.wifi_latency_us),
        }
        status, body = self._http_request("POST", "/tcp", payload=payload)
        if status != 200:
            raise ConnectionError(f"/tcp failed status={status} body={body}")
        parsed: dict[str, Any] = {}
        if body.strip():
            try:
                maybe = json.loads(body)
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = {}
        if parsed.get("success") is False:
            raise ConnectionError(f"/tcp returned success=false body={body}")
        if "connected" in parsed and not bool(parsed.get("connected")):
            raise ConnectionError(
                "/tcp configured but shield reports connected=false "
                "(check local IP/port or firewall)."
            )
        self.logger.info("POST /tcp success: %s", body.strip() or "{}")

    def _wifi_apply_channel_settings(self) -> None:
        gain = self.config.wifi_channel_gain.lower()
        custom_command = self.config.wifi_gain_command.strip()
        if custom_command:
            self._wifi_send_board_command(custom_command)
            time.sleep(0.2)
            self.logger.info(
                "Applied device-specific gain command before channel settings: %s",
                custom_command,
            )

        gain_code = self.OPENBCI_GAIN_CODES[gain]
        commands = [
            self._build_channel_setting_command(selector, gain_code)
            for selector in self.OPENBCI_CHANNEL_SELECTORS[: self.config.channels]
        ]
        for command in commands:
            if self.stop_event.is_set():
                return
            self._wifi_send_board_command(command)
            time.sleep(0.02)
        self.logger.info(
            "Applied OpenBCI channel settings: channels=%d gain=%s normal bias=include srb2=connect srb1=disconnect.",
            self.config.channels,
            gain,
        )

    @staticmethod
    def _build_channel_setting_command(channel_selector: str, gain_code: str) -> str:
        # x CHANNEL POWER_ON GAIN NORMAL_INPUT BIAS_INCLUDE SRB2_CONNECT SRB1_DISCONNECT X
        return f"x{channel_selector}0{gain_code}0110X"

    def _wifi_send_board_command(self, command: str) -> None:
        payload = {"command": command}
        status, body = self._http_request("POST", "/command", payload=payload)
        if status != 200:
            raise ConnectionError(f"/command failed status={status} body={body}")

        parsed: dict[str, Any] = {}
        if body.strip():
            try:
                maybe = json.loads(body)
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = {}
        if parsed.get("success") is False:
            raise ConnectionError(f"/command returned success=false body={body}")
        self.logger.info("POST /command success: %s", body.strip() or "{}")

    def _wifi_stream_start(self) -> None:
        status, body = self._http_request("GET", "/stream/start")
        if status != 200:
            raise ConnectionError(f"/stream/start failed status={status} body={body}")
        self.logger.info("GET /stream/start success: %s", body.strip() or "{}")

    def _wifi_stream_stop(self) -> None:
        try:
            status, body = self._http_request("GET", "/stream/stop")
            if status == 200:
                self.logger.info("GET /stream/stop success: %s", body.strip() or "{}")
        except Exception:
            pass

    def _wifi_tcp_disconnect(self) -> None:
        try:
            status, body = self._http_request("DELETE", "/tcp")
            if status == 200:
                self.logger.info("DELETE /tcp success: %s", body.strip() or "{}")
        except Exception:
            pass

    def _http_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, str]:
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
            head, body = text.split("\r\n\r\n", 1)
        else:
            head, body = text, ""
        status = 0
        first = head.splitlines()[0] if head.splitlines() else ""
        parts = first.split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
        return status, body

    @staticmethod
    def _guess_local_ip(device_ip: str) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((device_ip, 80))
            local = sock.getsockname()[0]
            return local
        finally:
            sock.close()
