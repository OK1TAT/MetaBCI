from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from .config import AppConfig
from .utils import DecodedSample, RawChunk, RuntimeStats, put_with_drop_oldest


def decode_int24(data3: bytes, endian: str = "big") -> int:
    if len(data3) != 3:
        raise ValueError("decode_int24 expects exactly 3 bytes.")
    if endian not in {"big", "little"}:
        raise ValueError("endian must be 'big' or 'little'.")
    if endian == "big":
        b0, b1, b2 = data3
    else:
        b2, b1, b0 = data3

    unsigned = (b0 << 16) | (b1 << 8) | b2
    # Int24 two's-complement:
    # b0 < 0x80 => positive (or zero)
    # b0 >= 0x80 => negative, equals unsigned - 0x1000000
    if b0 < 0x80:
        return unsigned
    return unsigned - (1 << 24)


@dataclass
class _HalfPacket:
    recv_timestamp: float
    sample_number: int
    counts: list[int]


@dataclass
class _SinglePacketCandidate:
    recv_timestamp: float
    sample_number: int
    counts: list[int]
    quality_score: float


class PacketDecoderThread(threading.Thread):
    PACKET_SIZE = 33
    HEADER = 0xA0
    TAIL = 0xC0

    def __init__(
        self,
        config: AppConfig,
        raw_queue: "queue.Queue[RawChunk]",
        plot_queue: "queue.Queue[DecodedSample]",
        save_queue: "queue.Queue[DecodedSample]",
        save_enabled_event: threading.Event | None,
        stop_event: threading.Event,
        stats: RuntimeStats,
    ) -> None:
        super().__init__(name="DecoderThread", daemon=True)
        self.config = config
        self.raw_queue = raw_queue
        self.plot_queue = plot_queue
        self.save_queue = save_queue
        self.save_enabled_event = save_enabled_event
        self.stop_event = stop_event
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)
        self._buffer = bytearray()
        self._last_packet_sample: int | None = None
        self._pending_half: _HalfPacket | None = None
        self._pending_single: _SinglePacketCandidate | None = None
        self._last_packet_signature: bytes | None = None
        self._duplicate_packet_count = 0
        self._debug_logged_packets = 0
        self._aux_nonzero_count = 0
        if self.config.endian == "little":
            self._effective_endian = "little"
        else:
            # Protocol document defines EEG payload as MSB-first signed int24.
            # To avoid auto-misclassification on railed/noisy startup frames,
            # treat both explicit "big" and "auto" as big-endian decode.
            self._effective_endian = "big"
            if self.config.endian == "auto":
                self.logger.info(
                    "Endian mode is auto, but protocol is fixed MSB-first. Using big-endian decode."
                )

    def _input_channels(self) -> int:
        if self.config.force_input_channels in (8, 16):
            return int(self.config.force_input_channels)
        # Protocol pairing for this device is 16ch: same-sample two-packet frame.
        # Prefer 16ch pairing when display/config is set to 16 even if /board metadata
        # is temporarily inconsistent.
        if self.config.channels == 16:
            return 16
        if self.config.device_channels in (8, 16):
            return int(self.config.device_channels)
        return self.config.channels

    def _decode_counts(self, eeg_bytes: bytes, endian: str) -> list[int]:
        counts: list[int] = []
        for i in range(8):
            start = i * 3
            data3 = eeg_bytes[start : start + 3]
            counts.append(decode_int24(data3, endian=endian))
        return counts

    def _resolve_endian(self, eeg_bytes: bytes) -> str:
        _ = eeg_bytes
        return self._effective_endian

    def run(self) -> None:
        while not self.stop_event.is_set() or not self.raw_queue.empty():
            try:
                chunk = self.raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._buffer.extend(chunk.data)
            self._parse_buffer(chunk.recv_timestamp)

        if self._pending_half is not None:
            self.logger.warning("16ch mode stopped with one unpaired half-packet.")
            self._pending_half = None
        if self._pending_single is not None:
            self._emit_sample(
                recv_timestamp=self._pending_single.recv_timestamp,
                sample_number=self._pending_single.sample_number,
                counts=self._pending_single.counts,
            )
            self._pending_single = None

    def _parse_buffer(self, default_ts: float) -> None:
        while True:
            if len(self._buffer) < 1:
                return
            if self._buffer[0] != self.HEADER:
                idx = self._buffer.find(bytes([self.HEADER]))
                if idx == -1:
                    dropped = len(self._buffer)
                    self._buffer.clear()
                    self.stats.add_bad_packet(1)
                    self.logger.warning(
                        "Header not found. Discarded %d bytes while resyncing.",
                        dropped,
                    )
                    return
                dropped = idx
                del self._buffer[:idx]
                self.stats.add_bad_packet(1)
                self.logger.warning(
                    "Misaligned stream. Dropped %d bytes before next header.", dropped
                )

            if len(self._buffer) < self.PACKET_SIZE:
                return

            packet = bytes(self._buffer[: self.PACKET_SIZE])
            if packet[-1] != self.TAIL:
                del self._buffer[0]
                self.stats.add_bad_packet(1)
                self.logger.warning("Tail mismatch (expected 0xC0). Resync by 1 byte.")
                continue

            del self._buffer[: self.PACKET_SIZE]
            try:
                self._handle_packet(packet, default_ts)
            except Exception as exc:
                self.stats.add_bad_packet(1)
                self.stats.set_error(str(exc))
                self.logger.exception("Packet decode error: %s", exc)

    def _handle_packet(self, packet: bytes, recv_timestamp: float) -> None:
        sample_number = packet[1]
        eeg_bytes = packet[2:26]
        aux_bytes = packet[26:32]
        packet_signature = packet[1:33]

        if aux_bytes != b"\x00\x00\x00\x00\x00\x00":
            self._aux_nonzero_count += 1
            if self._aux_nonzero_count % 200 == 1:
                self.logger.warning(
                    "Aux bytes non-zero (count=%d): %s",
                    self._aux_nonzero_count,
                    aux_bytes.hex(" "),
                )

        # In 8ch mode we can safely drop exact duplicate packets.
        # In forced 16ch pairing mode, two halves may be byte-identical when channels are idle
        # (for example both halves are 0x80 00 00...), so do not deduplicate there.
        if self._input_channels() == 8:
            if self._last_packet_signature == packet_signature:
                self._duplicate_packet_count += 1
                if self._duplicate_packet_count % 2000 == 1:
                    self.logger.info(
                        "Detected duplicated packet(s), dropped=%d.",
                        self._duplicate_packet_count,
                    )
                return
            self._last_packet_signature = packet_signature

        active_endian = self._resolve_endian(eeg_bytes)
        counts = self._decode_counts(eeg_bytes, active_endian)
        if self._debug_logged_packets < 4:
            self.logger.info(
                "Packet debug #%d sample=%d tail=0x%02X endian=%s input_ch=%d counts=%s hex=%s",
                self._debug_logged_packets + 1,
                sample_number,
                packet[-1],
                active_endian,
                self._input_channels(),
                counts,
                packet.hex(" "),
            )
            self._debug_logged_packets += 1
        self._track_sample_continuity(sample_number)

        if self._input_channels() == 8:
            self._handle_single_channel_packet(
                recv_timestamp=recv_timestamp,
                sample_number=sample_number,
                counts=counts,
            )
            return

        half = _HalfPacket(
            recv_timestamp=recv_timestamp,
            sample_number=sample_number,
            counts=counts,
        )
        if self._pending_half is None:
            self._pending_half = half
            return

        first = self._pending_half
        if half.sample_number != first.sample_number:
            self.stats.add_pair_mismatch(1)
            self.logger.warning(
                "16ch pair mismatch. first=%d second=%d. "
                "This device encodes one 16ch sample as two 33-byte packets "
                "with the same sample number.",
                first.sample_number,
                half.sample_number,
            )
            self._pending_half = half
            return

        combined_counts = first.counts + half.counts
        # Strategy (per protocol):
        # - two consecutive packets with the same sample_number encode halves 1-8 and 9-16
        # - sample_number uses the first packet id to avoid channel mixing
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
        # Some shield firmware duplicates a sample number with different payload quality.
        # We keep one candidate per sample_number and choose the one with lower rail/noise energy.
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
            if quality < self._pending_single.quality_score:
                self._pending_single = _SinglePacketCandidate(
                    recv_timestamp=recv_timestamp,
                    sample_number=sample_number,
                    counts=counts,
                    quality_score=quality,
                )
            return

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

    def _track_sample_continuity(self, sample_number: int) -> None:
        if self._last_packet_sample is None:
            self._last_packet_sample = sample_number
            return
        delta = (sample_number - self._last_packet_sample) % 256
        # Same sample number can occur in duplicated-packet cases and
        # in some 16ch pairing strategies. Do not count as packet loss.
        if delta == 0:
            return
        if delta != 1:
            missing = delta - 1
            self.stats.add_packet_gap(missing)
            self.logger.warning(
                "Packet sample number discontinuity: prev=%d curr=%d missing=%d",
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
        uV = [value * self.config.scale_factor for value in counts]
        sample = DecodedSample(
            recv_timestamp=recv_timestamp,
            sample_number=sample_number,
            counts=counts,
            uV=uV,
        )
        put_with_drop_oldest(
            self.plot_queue, sample, self.stats, self.logger, "plot_queue"
        )
        if self.save_enabled_event is not None and self.save_enabled_event.is_set():
            put_with_drop_oldest(
                self.save_queue, sample, self.stats, self.logger, "save_queue"
            )
        self.stats.add_decoded(1)
