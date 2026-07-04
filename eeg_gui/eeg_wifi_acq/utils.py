from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class RawChunk:
    recv_timestamp: float
    data: bytes


@dataclass
class DecodedSample:
    recv_timestamp: float
    sample_number: int
    counts: list[int]
    uV: list[float]


@dataclass
class EventMarker:
    timestamp: float
    label: str


class RuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset_session()

    def reset_session(self) -> None:
        with self._lock:
            self.connected = False
            self.packet_drop_count = 0
            self.bad_packet_count = 0
            self.pair_mismatch_count = 0
            self.queue_drop_count = 0
            self.reconnect_count = 0
            self.decoded_samples = 0
            self.saved_samples = 0
            self.last_error = ""
            self.output_csv: Optional[str] = None
            self.output_npz: Optional[str] = None
            self.session_start_ts = time.time()

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.connected = connected

    def add_packet_gap(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self.packet_drop_count += count

    def add_bad_packet(self, count: int = 1) -> None:
        with self._lock:
            self.bad_packet_count += max(1, count)

    def add_pair_mismatch(self, count: int = 1) -> None:
        with self._lock:
            self.pair_mismatch_count += max(1, count)

    def add_queue_drop(self, count: int = 1) -> None:
        with self._lock:
            self.queue_drop_count += max(1, count)

    def add_reconnect(self, count: int = 1) -> None:
        with self._lock:
            self.reconnect_count += max(1, count)

    def add_decoded(self, count: int = 1) -> None:
        with self._lock:
            self.decoded_samples += max(1, count)

    def add_saved(self, count: int = 1) -> None:
        with self._lock:
            self.saved_samples += max(1, count)

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.last_error = msg

    def set_output_paths(self, csv_path: str, npz_path: Optional[str]) -> None:
        with self._lock:
            self.output_csv = csv_path
            self.output_npz = npz_path

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1e-6, time.time() - self.session_start_ts)
            avg_rate = self.decoded_samples / elapsed
            return {
                "connected": self.connected,
                "packet_drop_count": self.packet_drop_count,
                "bad_packet_count": self.bad_packet_count,
                "pair_mismatch_count": self.pair_mismatch_count,
                "queue_drop_count": self.queue_drop_count,
                "reconnect_count": self.reconnect_count,
                "decoded_samples": self.decoded_samples,
                "saved_samples": self.saved_samples,
                "last_error": self.last_error,
                "output_csv": self.output_csv,
                "output_npz": self.output_npz,
                "avg_rate": avg_rate,
            }


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_session_stem(subject: str) -> str:
    clean_subject = re.sub(r"[^A-Za-z0-9_\-]", "_", subject.strip() or "subject")
    now = datetime.now()
    return f"{clean_subject}_{now.strftime('%H%M%S')}_{now.strftime('%Y%m%d')}"


def put_with_drop_oldest(
    q: "queue.Queue[Any]",
    item: Any,
    stats: RuntimeStats,
    logger: logging.Logger,
    queue_name: str,
) -> bool:
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            stats.add_queue_drop(1)
            logger.warning("Queue %s full, dropped oldest item.", queue_name)
            return True
        except queue.Full:
            stats.add_queue_drop(1)
            logger.warning("Queue %s still full, dropped new item.", queue_name)
            return False
