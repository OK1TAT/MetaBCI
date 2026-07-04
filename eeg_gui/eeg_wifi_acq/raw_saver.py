from __future__ import annotations

import csv
import logging
import queue
import threading
from pathlib import Path

from .config import AppConfig
from .utils import RawChunk


class RawStreamSaverThread(threading.Thread):
    def __init__(
        self,
        config: AppConfig,
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
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        raw_bin_path = save_path / f"{self.session_stem}.raw.bin"
        raw_idx_path = save_path / f"{self.session_stem}.raw_index.csv"

        self.logger.info("Saving raw stream to %s", raw_bin_path)
        offset = 0
        with raw_bin_path.open("wb") as fb, raw_idx_path.open(
            "w", newline="", encoding="utf-8"
        ) as fi:
            idx_writer = csv.writer(fi)
            idx_writer.writerow(
                ["recv_timestamp", "offset_start", "chunk_size", "offset_end_exclusive"]
            )

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
                idx_writer.writerow(
                    [f"{chunk.recv_timestamp:.6f}", start, size, end]
                )
                offset = end

        self.logger.info(
            "Raw saver stopped. bytes=%d index=%s", offset, raw_idx_path
        )

