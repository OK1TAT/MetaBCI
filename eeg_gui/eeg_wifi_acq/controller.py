from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .config import AppConfig
from .decoder import PacketDecoderThread
from .raw_saver import RawStreamSaverThread
from .receiver import NetworkReceiverThread
from .saver import DataSaverThread
from .utils import (
    DecodedSample,
    EventMarker,
    RawChunk,
    RuntimeStats,
    format_session_stem,
    put_with_drop_oldest,
)


class EEGAcquisitionController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.stats = RuntimeStats()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.raw_queue: "queue.Queue[RawChunk]" = queue.Queue(maxsize=self.config.raw_queue_size)
        self.raw_save_queue: "queue.Queue[RawChunk]" = queue.Queue(
            maxsize=max(self.config.raw_queue_size * 2, 10000)
        )
        self.plot_queue: "queue.Queue[DecodedSample]" = queue.Queue(maxsize=self.config.plot_queue_size)
        self.save_queue: "queue.Queue[DecodedSample]" = queue.Queue(maxsize=self.config.save_queue_size)
        self.marker_queue: "queue.Queue[EventMarker]" = queue.Queue(maxsize=10000)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._save_enabled_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._save_threads: list[threading.Thread] = []
        self._join_thread: Optional[threading.Thread] = None

    def update_config(
        self,
        *,
        host: str,
        port: int,
        protocol: str,
        channels: int,
        sample_rate: int,
        endian: str,
        force_input_channels: int | None = None,
        display_seconds: float,
        save_dir: str,
        subject: str,
    ) -> None:
        with self._lock:
            self.config.host = host
            self.config.port = port
            self.config.protocol = protocol
            self.config.channels = channels
            self.config.sample_rate = sample_rate
            self.config.endian = endian
            if force_input_channels is not None:
                self.config.force_input_channels = force_input_channels
            self.config.display_seconds = display_seconds
            self.config.save_dir = save_dir
            self.config.subject = subject
            self.config.validate()

    def is_running(self) -> bool:
        with self._lock:
            return any(t.is_alive() for t in self._threads)

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if any(t.is_alive() for t in self._threads):
                return False, "Acquisition is already running."
            if self._join_thread is not None and self._join_thread.is_alive():
                return False, "Stopping previous session. Please wait."

            self._clear_queue(self.raw_queue)
            self._clear_queue(self.raw_save_queue)
            self._clear_queue(self.plot_queue)
            self._clear_queue(self.save_queue)
            self._clear_queue(self.marker_queue)
            self.stats.reset_session()
            self._stop_event = threading.Event()
            self._save_enabled_event.clear()
            self._save_threads = []

            receiver = NetworkReceiverThread(
                config=self.config,
                raw_queue=self.raw_queue,
                raw_save_queue=self.raw_save_queue,
                save_enabled_event=self._save_enabled_event,
                stop_event=self._stop_event,
                stats=self.stats,
            )
            decoder = PacketDecoderThread(
                config=self.config,
                raw_queue=self.raw_queue,
                plot_queue=self.plot_queue,
                save_queue=self.save_queue,
                save_enabled_event=self._save_enabled_event,
                stop_event=self._stop_event,
                stats=self.stats,
            )

            self._threads = [receiver, decoder]
            if self.config.save_on_start:
                self._start_saving_locked()
            for thread in self._threads:
                thread.start()
            self.logger.info("Acquisition started.")
            return True, "Started."

    def is_saving(self) -> bool:
        with self._lock:
            return self._save_enabled_event.is_set() and any(
                t.is_alive() for t in self._save_threads
            )

    def start_saving(self) -> tuple[bool, str]:
        with self._lock:
            if not any(t.is_alive() for t in self._threads):
                return False, "Acquisition is not running."
            if self._save_enabled_event.is_set() and any(
                t.is_alive() for t in self._save_threads
            ):
                return False, "Saving is already running."

            self._start_saving_locked()
            self.logger.info("Saving started.")
            return True, "Saving started."

    def _start_saving_locked(self) -> None:
        self._clear_queue(self.raw_save_queue)
        self._clear_queue(self.save_queue)
        self._clear_queue(self.marker_queue)
        session_stem = format_session_stem(self.config.subject)

        saver = DataSaverThread(
            config=self.config,
            save_queue=self.save_queue,
            marker_queue=self.marker_queue,
            stop_event=self._stop_event,
            stats=self.stats,
            session_stem=session_stem,
        )
        raw_saver = RawStreamSaverThread(
            config=self.config,
            raw_save_queue=self.raw_save_queue,
            stop_event=self._stop_event,
            session_stem=session_stem,
        )

        self._save_threads = [saver, raw_saver]
        self._save_enabled_event.set()
        for thread in self._save_threads:
            thread.start()

    def stop(self, wait: bool = False) -> tuple[bool, str]:
        with self._lock:
            threads = [
                t
                for t in [*self._threads, *self._save_threads]
                if t.is_alive()
            ]
            if not threads:
                return False, "Not running."
            self._stop_event.set()

        def _join_all() -> None:
            for thread in threads:
                thread.join(timeout=5.0)
            with self._lock:
                self._threads = []
                self._save_threads = []
                self._save_enabled_event.clear()
            self.logger.info("Acquisition stopped.")

        if wait:
            _join_all()
        else:
            self._join_thread = threading.Thread(target=_join_all, daemon=True)
            self._join_thread.start()
        return True, "Stopping."

    def add_marker(self, label: str, timestamp: float | None = None) -> None:
        if not self._save_enabled_event.is_set():
            return
        clean_label = str(label).strip()
        if not clean_label:
            return
        marker = EventMarker(
            timestamp=float(timestamp if timestamp is not None else time.time()),
            label=clean_label,
        )
        put_with_drop_oldest(
            self.marker_queue,
            marker,
            self.stats,
            self.logger,
            "marker_queue",
        )

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
