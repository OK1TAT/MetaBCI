"""
完全照搬 eeg_realtime_window.py 的管线: AppConfig + EEGAcquisitionController + RealtimeEEGProcessor
只把最后 pyqtgraph 画图改成 stdout CSV
"""
import sys, os, queue, time
import numpy as np
from scipy import signal as scipy_signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eeg_wifi_acq.config import AppConfig
from eeg_wifi_acq.controller import EEGAcquisitionController
from eeg_wifi_acq.utils import setup_logging


# ====== RealtimeEEGProcessor (逐字复制 eeg_realtime_window.py 第42-144行) ======

class RealtimeEEGProcessor:
    """实时去基线、50 Hz 去工频，并做 4-30 Hz 带通滤波。"""

    def __init__(
        self,
        channels: int,
        sample_rate: float,
        low_hz: float = 4.0,
        high_hz: float = 30.0,
        notch_hz: float = 50.0,
        notch_quality: float = 30.0,
        baseline_seconds: float = 1.0,
    ) -> None:
        self.channels = channels
        self.sample_rate = float(sample_rate)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.notch_hz = float(notch_hz)
        self.notch_quality = float(notch_quality)
        self.baseline_seconds = float(baseline_seconds)

        nyquist = self.sample_rate / 2.0
        high = min(self.high_hz, nyquist * 0.90)
        if self.low_hz <= 0 or high <= self.low_hz:
            raise ValueError("滤波范围无效：采样率必须支持 4-30 Hz 带通。")

        self.notch_sos = None
        self.notch_zi = None
        if 0.0 < self.notch_hz < nyquist:
            notch_b, notch_a = scipy_signal.iirnotch(
                w0=self.notch_hz,
                Q=self.notch_quality,
                fs=self.sample_rate,
            )
            self.notch_sos = scipy_signal.tf2sos(notch_b, notch_a)
            self.notch_zi = np.zeros(
                (self.notch_sos.shape[0], 2, channels), dtype=np.float64
            )

        self.bandpass_sos = scipy_signal.butter(
            4,
            [self.low_hz, high],
            btype="bandpass",
            fs=self.sample_rate,
            output="sos",
        )
        self.bandpass_zi = np.zeros(
            (self.bandpass_sos.shape[0], 2, channels), dtype=np.float64
        )
        self.baseline = np.zeros((channels,), dtype=np.float64)
        self.baseline_ready = np.zeros((channels,), dtype=bool)
        self.baseline_alpha = 1.0 / max(1.0, self.sample_rate * self.baseline_seconds)

    def process(self, values_uv: list[float]) -> np.ndarray:
        return self.process_batch([values_uv])[0]

    def process_batch(self, rows_uv: list[list[float]] | np.ndarray) -> np.ndarray:
        src = np.asarray(rows_uv, dtype=np.float64)
        if src.ndim == 1:
            src = src.reshape(1, -1)

        sample_count = src.shape[0]
        values = np.full((sample_count, self.channels), np.nan, dtype=np.float64)
        n = min(src.shape[1], self.channels)
        if n:
            values[:, :n] = src[:, :n]

        detrended = np.zeros((sample_count, self.channels), dtype=np.float64)
        valid_mask = np.isfinite(values)
        for row_idx in range(sample_count):
            row_valid = valid_mask[row_idx]
            if not np.any(row_valid):
                continue

            first_valid = row_valid & ~self.baseline_ready
            if np.any(first_valid):
                self.baseline[first_valid] = values[row_idx, first_valid]
                self.baseline_ready[first_valid] = True

            ready_valid = row_valid & self.baseline_ready
            if np.any(ready_valid):
                current = values[row_idx, ready_valid]
                self.baseline[ready_valid] += self.baseline_alpha * (
                    current - self.baseline[ready_valid]
                )
                detrended[row_idx, ready_valid] = current - self.baseline[ready_valid]

        notched = detrended
        if self.notch_sos is not None and self.notch_zi is not None:
            notched, self.notch_zi = scipy_signal.sosfilt(
                self.notch_sos,
                detrended,
                axis=0,
                zi=self.notch_zi,
            )
        filtered, self.bandpass_zi = scipy_signal.sosfilt(
            self.bandpass_sos,
            notched,
            axis=0,
            zi=self.bandpass_zi,
        )
        filtered[~valid_mask] = np.nan
        return filtered.astype(np.float32, copy=False)


# ====== 主流程 (与 eeg_realtime_window.py LiveEEGWindow.__init__ 一致) ======

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
DISPLAY_CHANNELS = 16

setup_logging()

config = AppConfig(
    host=HOST,
    port=PORT,
    protocol="wifi_shield",
    channels=DISPLAY_CHANNELS,               # 16 通道配置 (选通器 1-8 + QWERTYUI)
    force_input_channels=DISPLAY_CHANNELS,   # 16 通道解码 (双包配对)
    sample_rate=500,
    endian="big",
    display_seconds=8.0,
    ui_fps=15,
    max_plot_drain_per_tick=250,
    save_dir="records",
    subject="subject01",
    save_npz=False,
    save_on_start=False,  # 不启动存盘线程，只接收
    wifi_accept_timeout=20.0,
    # 与 stroke 参考实现 (eeg_realtime_window.py / test_live_eeg_config.py) 保持一致
    # wifi_gain_command="1" → 设备专用初始化命令, 在通道配置前发送, 激活第二片ADS1299
    # wifi_channel_gain="x6" → x6 增益 (非默认 x24), 匹配 scale_factor
    wifi_apply_channel_settings=True,
    wifi_channel_gain="x6",
    wifi_gain_command="1",
    # scale_factor 不显式传, 使用 AppConfig 默认值 CYTON_GAIN_X6_SCALE_FACTOR = 0.08941
    # 公式: Scale (V/count) = 4.5V / gain / (2^23-1)
    # x6: 4.5/6/8388607 = 0.08941 µV/count
)
config.validate()

controller = EEGAcquisitionController(config=config)
processor = RealtimeEEGProcessor(
    channels=DISPLAY_CHANNELS,
    sample_rate=controller.config.sample_rate,
    low_hz=4.0,
    high_hz=30.0,
)

ok, msg = controller.start()
if not ok:
    sys.stderr.write(f"START FAILED: {msg}\n"); sys.stderr.flush(); sys.exit(1)

sys.stderr.write("CONNECTED\n"); sys.stderr.flush()

# 同时写入文件，方便检查脑电信号是否正常
from datetime import datetime
_record_dir = os.path.join(SCRIPT_DIR, "records")
os.makedirs(_record_dir, exist_ok=True)
_record_file = os.path.join(
    _record_dir,
    f"bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
_record_fh = open(_record_file, "w", encoding="utf-8")
_record_fh.write("timestamp,sample_number," + ",".join(f"ch{i+1}_uV" for i in range(16)) + "\n")
sys.stderr.write(f"RECORDING to {_record_file}\n"); sys.stderr.flush()

count = 0
last_heartbeat = time.time()
try:
    while controller.is_running():
        drained = 0
        limit = controller.config.max_plot_drain_per_tick
        samples_uv = []
        sample_numbers = []
        while drained < limit:
            try:
                sample = controller.plot_queue.get_nowait()
            except queue.Empty:
                break
            samples_uv.append(sample.uV)
            sample_numbers.append(sample.sample_number)
            drained += 1

        if samples_uv:
            filtered = processor.process_batch(samples_uv)
            now_ts = time.time()
            for i, row in enumerate(filtered):
                line = ",".join(
                    "0.0000" if np.isnan(float(v)) else f"{float(v):.4f}"
                    for v in row
                )
                sys.stdout.write(line + "\n")
                # 文件写入：时间戳 + 采样序号 + 16通道uV
                _record_fh.write(f"{now_ts:.6f},{sample_numbers[i]}," + line + "\n")
                count += 1
            sys.stdout.flush()
            _record_fh.flush()
        else:
            time.sleep(0.01)

        # 每 2 秒一次心跳，报告解码/接收状态 + 队列深度
        now = time.time()
        if now - last_heartbeat >= 2.0:
            snap = controller.stats.snapshot()
            sys.stderr.write(
                f"HEARTBEAT samples={count} decoded={snap['decoded_samples']} "
                f"drop={snap['packet_drop_count']} bad={snap['bad_packet_count']} "
                f"mismatch={snap['pair_mismatch_count']} rate={snap['avg_rate']:.1f}/s "
                f"raw_q={controller.raw_queue.qsize()} plot_q={controller.plot_queue.qsize()}\n"
            )
            sys.stderr.flush()
            last_heartbeat = now

except KeyboardInterrupt:
    pass
finally:
    controller.stop(wait=True)
    sys.stdout.flush()
    _record_fh.close()
    sys.stderr.write(f"DONE: {count} samples → {_record_file}\n"); sys.stderr.flush()
