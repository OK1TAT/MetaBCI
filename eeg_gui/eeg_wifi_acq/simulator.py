from __future__ import annotations

import argparse
import math
import random
import socket
import time

from .config import DEFAULT_SCALE_FACTOR


def encode_int24(value: int, endian: str = "big") -> bytes:
    if value < 0:
        value = (1 << 24) + value
    value &= 0xFFFFFF
    if endian == "big":
        return bytes([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])
    if endian == "little":
        return bytes([value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF])
    raise ValueError("endian must be big or little")


def build_packet(sample_number: int, ch_counts: list[int], endian: str) -> bytes:
    if len(ch_counts) != 8:
        raise ValueError("build_packet needs 8 channel counts")
    payload = bytearray()
    payload.append(0xA0)
    payload.append(sample_number & 0xFF)
    for c in ch_counts:
        payload.extend(encode_int24(c, endian=endian))
    payload.extend(b"\x00" * 6)
    payload.append(0xC0)
    return bytes(payload)


class EEGSimulator:
    def __init__(
        self,
        host: str,
        port: int,
        protocol: str,
        channels: int,
        sample_rate: int,
        endian: str,
        scale_factor: float,
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.channels = channels
        self.sample_rate = sample_rate
        self.endian = endian
        self.scale_factor = scale_factor
        self.sample_number = 0
        self.phase = [0.0 for _ in range(channels)]
        self.freq = [6.0 + i * 0.8 for i in range(channels)]
        self.amp_uV = [30.0 + i * 2.0 for i in range(channels)]

    def _next_counts(self) -> list[int]:
        counts: list[int] = []
        for i in range(self.channels):
            self.phase[i] += 2.0 * math.pi * self.freq[i] / self.sample_rate
            uv = self.amp_uV[i] * math.sin(self.phase[i]) + random.uniform(-3.0, 3.0)
            counts.append(int(round(uv / self.scale_factor)))
        return counts

    def _next_payload(self) -> bytes:
        counts = self._next_counts()
        if self.channels == 8:
            pkt = build_packet(self.sample_number, counts, self.endian)
            self.sample_number = (self.sample_number + 1) % 256
            return pkt

        # 16ch mode: pair two consecutive 33-byte packets
        pkt1 = build_packet(self.sample_number, counts[:8], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        pkt2 = build_packet(self.sample_number, counts[8:16], self.endian)
        self.sample_number = (self.sample_number + 1) % 256
        return pkt1 + pkt2

    def run(self) -> None:
        if self.protocol == "tcp":
            self._run_tcp_server()
        else:
            self._run_udp_sender()

    def _run_tcp_server(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"[SIM] TCP server listening on {self.host}:{self.port}")
        try:
            while True:
                conn, addr = server.accept()
                print(f"[SIM] Client connected: {addr}")
                with conn:
                    self._stream_loop(send_func=conn.sendall)
                print("[SIM] Client disconnected, waiting next ...")
        finally:
            server.close()

    def _run_udp_sender(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[SIM] UDP sending to {self.host}:{self.port}")
        try:
            self._stream_loop(send_func=lambda b: sock.sendto(b, (self.host, self.port)))
        finally:
            sock.close()

    def _stream_loop(self, send_func) -> None:
        next_send = time.perf_counter()
        dt = 1.0 / self.sample_rate
        while True:
            payload = self._next_payload()
            send_func(payload)
            next_send += dt
            sleep_s = next_send - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_send = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG WiFi device simulator")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--protocol", choices=["tcp", "udp"], default="tcp")
    parser.add_argument("--channels", choices=[8, 16], type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--endian", choices=["big", "little"], default="big")
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
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
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n[SIM] Stopped.")


if __name__ == "__main__":
    main()
