from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean


PACKET_SIZE = 33
HEADER = 0xA0
TAIL = 0xC0


def iter_packets(data: bytes):
    i = 0
    n = len(data)
    while i + PACKET_SIZE <= n:
        if data[i] != HEADER:
            i += 1
            continue
        pkt = data[i : i + PACKET_SIZE]
        if pkt[-1] == TAIL:
            yield i, pkt
            i += PACKET_SIZE
        else:
            i += 1


def decode_int24(data3: bytes, endian: str) -> int:
    if endian == "big":
        b0, b1, b2 = data3
    else:
        b2, b1, b0 = data3
    unsigned = (b0 << 16) | (b1 << 8) | b2
    if b0 < 0x80:
        return unsigned
    return unsigned - (1 << 24)


def paired_samples(packets: list[bytes], endian: str) -> list[list[int]]:
    out: list[list[int]] = []
    pending: tuple[int, list[int]] | None = None
    for pkt in packets:
        sn = pkt[1]
        eeg = pkt[2:26]
        counts = [decode_int24(eeg[i * 3 : i * 3 + 3], endian=endian) for i in range(8)]
        if pending is None:
            pending = (sn, counts)
            continue
        if sn != pending[0]:
            pending = (sn, counts)
            continue
        out.append(pending[1] + counts)
        pending = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw EEG stream (*.raw.bin)")
    parser.add_argument("raw_bin", type=str, help="Path to raw bin file")
    parser.add_argument("--max-lines", type=int, default=30)
    parser.add_argument("--endian", type=str, default="big", choices=["big", "little"])
    parser.add_argument("--stats", action="store_true", help="Print 16ch paired decode stats")
    args = parser.parse_args()

    p = Path(args.raw_bin)
    if not p.exists():
        raise FileNotFoundError(p)

    data = p.read_bytes()
    print(f"file={p} bytes={len(data)}")

    packet_list: list[bytes] = []
    total = 0
    for offset, pkt in iter_packets(data):
        packet_list.append(pkt)
        total += 1
        if total <= args.max_lines:
            sample = pkt[1]
            print(
                f"#{total:06d} off={offset:08d} sample=0x{sample:02X} "
                f"hex={pkt.hex(' ')}"
            )
    print(f"valid_packets={total}")

    if args.stats and packet_list:
        samples = paired_samples(packet_list, endian=args.endian)
        if not samples:
            print("paired_samples=0 (no same-sample adjacent packet pairs found)")
            return
        n = len(samples)
        rail_threshold = 8_380_000
        rail_ratio = []
        std = []
        for ch in range(16):
            arr = [row[ch] for row in samples]
            rail = sum(1 for v in arr if abs(v) >= rail_threshold)
            rail_ratio.append(rail / n)
            mu = mean(arr)
            var = mean((v - mu) ** 2 for v in arr)
            std.append(var ** 0.5)
        print(f"paired_samples={n} endian={args.endian}")
        print(
            "rail_ratio_per_ch="
            + ", ".join(f"CH{i+1}:{rail_ratio[i]:.3f}" for i in range(16))
        )
        print("std_per_ch=" + ", ".join(f"CH{i+1}:{std[i]:.1f}" for i in range(16)))


if __name__ == "__main__":
    main()
