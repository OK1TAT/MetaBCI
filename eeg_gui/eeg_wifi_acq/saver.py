from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .config import AppConfig
from .utils import DecodedSample, EventMarker, RuntimeStats


class DataSaverThread(threading.Thread):
    def __init__(
        self,
        config: AppConfig,
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
        self._pending_markers: list[EventMarker] = []
        self._current_label = ""

    def run(self) -> None:
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        stem = self.session_stem
        csv_path = save_path / f"{stem}.csv"
        events_path = save_path / f"{stem}.events.csv"
        npz_path = save_path / f"{stem}.npz"

        numeric_header = ["recv_timestamp", "sample_number"]
        numeric_header.extend([f"ch{i + 1}_uV" for i in range(self.config.channels)])
        header = [*numeric_header, "label", "event_marker"]

        npz_cache: list[list[float]] = []
        batch: list[DecodedSample] = []

        self.logger.info("Saving CSV to %s", csv_path)
        with (
            csv_path.open("w", newline="", encoding="utf-8") as f,
            events_path.open("w", newline="", encoding="utf-8") as events_f,
        ):
            writer = csv.writer(f)
            event_writer = csv.writer(events_f)
            writer.writerow(header)
            event_writer.writerow(["timestamp", "label"])
            self.stats.set_output_paths(str(csv_path), str(npz_path) if self.config.save_npz else None)

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
                        self._flush_batch(writer, event_writer, flush_batch, npz_cache)
                        last_flush_ts = now

            if batch:
                self._flush_batch(writer, event_writer, batch, npz_cache)
            self._drain_marker_queue()
            self._write_remaining_marker_events(event_writer)

        if self.config.save_npz and npz_cache:
            arr = np.asarray(npz_cache, dtype=np.float64)
            np.savez_compressed(
                npz_path,
                data=arr,
                columns=np.asarray(numeric_header, dtype="U64"),
            )
            self.logger.info("Saved NPZ to %s", npz_path)

        self.logger.info("Saver thread stopped.")

    def _flush_batch(
        self,
        writer: csv.writer,
        event_writer: csv.writer,
        batch: list[DecodedSample],
        npz_cache: list[list[float]],
    ) -> None:
        if not batch:
            return
        self._drain_marker_queue()
        rows: list[list[object]] = []
        for sample in batch:
            row: list[object] = [f"{sample.recv_timestamp:.6f}", sample.sample_number]
            padded = list(sample.uV[: self.config.channels])
            if len(padded) < self.config.channels:
                padded.extend([float("nan")] * (self.config.channels - len(padded)))
            row.extend("nan" if np.isnan(v) else f"{v:.6f}" for v in padded)
            event_labels = self._consume_markers_until(sample.recv_timestamp, event_writer)
            row.extend([self._current_label, ";".join(event_labels)])
            rows.append(row)
            if self.config.save_npz:
                npz_row = [sample.recv_timestamp, float(sample.sample_number)]
                npz_row.extend(float(v) for v in padded)
                npz_cache.append(npz_row)
        writer.writerows(rows)
        self.stats.add_saved(len(batch))
        self.logger.debug("Flushed %d rows.", len(batch))
        batch.clear()

    def _take_ready_batch(self, batch: list[DecodedSample]) -> list[DecodedSample]:
        if self.stop_event.is_set() or self.config.marker_sync_delay_s <= 0:
            ready = list(batch)
            batch.clear()
            return ready

        cutoff = time.time() - self.config.marker_sync_delay_s
        ready_count = 0
        while ready_count < len(batch) and batch[ready_count].recv_timestamp <= cutoff:
            ready_count += 1
        if ready_count <= 0:
            return []

        ready = batch[:ready_count]
        del batch[:ready_count]
        return ready

    def _drain_marker_queue(self) -> None:
        while True:
            try:
                self._pending_markers.append(self.marker_queue.get_nowait())
            except queue.Empty:
                break
        if len(self._pending_markers) > 1:
            self._pending_markers.sort(key=lambda marker: marker.timestamp)

    def _consume_markers_until(
        self,
        sample_timestamp: float,
        event_writer: csv.writer,
    ) -> list[str]:
        event_labels: list[str] = []
        while self._pending_markers and self._pending_markers[0].timestamp <= sample_timestamp:
            marker = self._pending_markers.pop(0)
            self._current_label = marker.label
            event_labels.append(marker.label)
            event_writer.writerow([f"{marker.timestamp:.6f}", marker.label])
        return event_labels

    def _write_remaining_marker_events(self, event_writer: csv.writer) -> None:
        for marker in self._pending_markers:
            event_writer.writerow([f"{marker.timestamp:.6f}", marker.label])
        self._pending_markers.clear()
