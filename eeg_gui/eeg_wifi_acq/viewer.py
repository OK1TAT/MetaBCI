from __future__ import annotations

import queue
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .controller import EEGAcquisitionController


class EEGAcquisitionWindow(QMainWindow):
    DISPLAY_CHANNELS = 16
    TRACE_SPACING_UV = 220.0
    DISPLAY_CLIP_UV = 300.0
    DISPLAY_TARGET_UV = 80.0
    DISPLAY_MIN_GAIN = 0.001
    DISPLAY_MAX_GAIN = 5.0
    TRACE_COLORS = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#5f9ea0",
        "#ff6347",
        "#6a5acd",
        "#20b2aa",
        "#4682b4",
        "#cd5c5c",
    ]

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EEG WiFi Acquisition")
        self.setMinimumSize(1300, 850)

        # Force fixed 16-channel acquisition/display as requested.
        config.channels = self.DISPLAY_CHANNELS
        config.force_input_channels = self.DISPLAY_CHANNELS
        config.endian = "big"
        config.validate()

        self.controller = EEGAcquisitionController(config=config)
        self.config = self.controller.config

        self._channel_offsets = np.asarray(
            [
                (self.DISPLAY_CHANNELS - 1 - i) * self.TRACE_SPACING_UV
                for i in range(self.DISPLAY_CHANNELS)
            ],
            dtype=np.float32,
        )
        self._buffer = np.zeros(
            (self.DISPLAY_CHANNELS, self.config.buffer_samples), dtype=np.float32
        )
        self._write_index = 0
        self._filled = 0
        self._x_axis = np.zeros((self.config.buffer_samples,), dtype=np.float32)
        self._rate_times: deque[float] = deque()
        self._curves: list[pg.PlotDataItem] = []
        self._plot_item: pg.PlotItem | None = None

        self._build_ui()
        self._reset_plot_layout()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(max(10, int(1000 / self.config.ui_fps)))

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        control_card = QFrame()
        control_card.setFrameShape(QFrame.StyledPanel)
        controls_layout = QGridLayout(control_card)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(6)

        self.host_edit = QLineEdit(self.config.host)
        self.host_edit.setPlaceholderText("Device IP, e.g. 192.168.4.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.port)

        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["wifi_shield", "tcp", "udp"])
        self.protocol_combo.setCurrentText(self.config.protocol)
        self.protocol_combo.currentTextChanged.connect(self._on_protocol_changed)

        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["16"])
        self.channels_combo.setCurrentText("16")
        self.channels_combo.setEnabled(False)
        self.channels_combo.setToolTip("Fixed to 16 channels for this window.")

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1, 5000)
        self.sample_rate_spin.setValue(self.config.sample_rate)

        self.endian_combo = QComboBox()
        self.endian_combo.addItems(["big"])
        self.endian_combo.setCurrentText("big")
        self.endian_combo.setEnabled(False)
        self.endian_combo.setToolTip("Fixed to big-endian (MSB first).")

        self.input_channels_combo = QComboBox()
        self.input_channels_combo.addItems(["16"])
        self.input_channels_combo.setCurrentText("16")
        self.input_channels_combo.setEnabled(False)
        self.input_channels_combo.setToolTip(
            "Fixed to 16-channel pair decode (sample-identical row as 1-8/9-16 is accepted)."
        )

        self.display_seconds_spin = QSpinBox()
        self.display_seconds_spin.setRange(2, 20)
        self.display_seconds_spin.setValue(int(self.config.display_seconds))

        self.subject_edit = QLineEdit(self.config.subject)
        self.save_dir_edit = QLineEdit(self.config.save_dir)

        form1 = QFormLayout()
        form1.addRow("Host", self.host_edit)
        form1.addRow("Port", self.port_spin)
        form1.addRow("Protocol", self.protocol_combo)
        form1.addRow("Channels", self.channels_combo)

        form2 = QFormLayout()
        form2.addRow("Sample Rate", self.sample_rate_spin)
        form2.addRow("Endian", self.endian_combo)
        form2.addRow("Decode Input", self.input_channels_combo)
        form2.addRow("Display Seconds", self.display_seconds_spin)
        form2.addRow("Subject", self.subject_edit)

        form3 = QFormLayout()
        form3.addRow("Save Dir", self.save_dir_edit)

        controls_layout.addLayout(form1, 0, 0)
        controls_layout.addLayout(form2, 0, 1)
        controls_layout.addLayout(form3, 0, 2)

        button_col = QVBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        button_col.addWidget(self.start_btn)
        button_col.addWidget(self.stop_btn)
        button_col.addStretch(1)
        controls_layout.addLayout(button_col, 0, 3)

        status_row = QHBoxLayout()
        self.connection_label = QLabel("Connection: disconnected")
        self.rate_label = QLabel("Rate: 0.0 Hz")
        self.drop_label = QLabel("Packet loss: 0")
        self.ch_label = QLabel("Channels: 16 (fixed)")
        self.save_label = QLabel("Saved: 0")
        self.path_label = QLabel("CSV: -")
        self.error_label = QLabel("Last error: -")
        status_row.addWidget(self.connection_label)
        status_row.addWidget(self.rate_label)
        status_row.addWidget(self.drop_label)
        status_row.addWidget(self.ch_label)
        status_row.addWidget(self.save_label)
        status_row.addWidget(self.path_label, 1)
        status_row.addWidget(self.error_label, 2)

        layout.addWidget(control_card)
        layout.addLayout(status_row)

        pg.setConfigOptions(antialias=False)
        self.plot_widget = pg.PlotWidget(background="#ffffff")
        layout.addWidget(self.plot_widget, 1)

        self.setStyleSheet(
            """
            QMainWindow { background-color: #f6f8fb; }
            QFrame { background: #ffffff; border: 1px solid #d7e1ea; border-radius: 8px; }
            QLabel { color: #1f3347; }
            QLineEdit, QSpinBox, QComboBox {
                background: #ffffff;
                border: 1px solid #c6d8e8;
                border-radius: 6px;
                padding: 4px;
            }
            QPushButton {
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:enabled { background: #1f6fb2; color: white; }
            QPushButton:disabled { background: #d0d8df; color: #6e7e8c; }
            """
        )
        self._on_protocol_changed(self.protocol_combo.currentText())

    def _on_protocol_changed(self, protocol: str) -> None:
        if protocol == "wifi_shield":
            if self.host_edit.text().strip() in {"", "127.0.0.1"}:
                self.host_edit.setText("192.168.4.1")
            self.host_edit.setPlaceholderText("Shield IP, e.g. 192.168.4.1")
            self.port_spin.setToolTip(
                "Local TCP listen port used by shield /tcp push (example: 9000)"
            )
        elif protocol == "udp":
            self.host_edit.setPlaceholderText("UDP bind host, e.g. 0.0.0.0")
            self.port_spin.setToolTip("UDP listen port")
        else:
            self.host_edit.setPlaceholderText("TCP device host")
            self.port_spin.setToolTip("TCP device port")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in [
            self.host_edit,
            self.port_spin,
            self.protocol_combo,
            self.sample_rate_spin,
            self.endian_combo,
            self.input_channels_combo,
            self.display_seconds_spin,
            self.subject_edit,
            self.save_dir_edit,
        ]:
            w.setEnabled(enabled)
        self.channels_combo.setEnabled(False)

    def _on_start_clicked(self) -> None:
        try:
            self.controller.update_config(
                host=self.host_edit.text().strip(),
                port=int(self.port_spin.value()),
                protocol=self.protocol_combo.currentText(),
                channels=self.DISPLAY_CHANNELS,
                sample_rate=int(self.sample_rate_spin.value()),
                endian="big",
                force_input_channels=16,
                display_seconds=float(self.display_seconds_spin.value()),
                save_dir=self.save_dir_edit.text().strip(),
                subject=self.subject_edit.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return

        self._reset_plot_layout()
        ok, msg = self.controller.start()
        if not ok:
            QMessageBox.warning(self, "Cannot start", msg)
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_controls_enabled(False)

    def _on_stop_clicked(self) -> None:
        self.controller.stop(wait=False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_controls_enabled(True)

    def _reset_plot_layout(self) -> None:
        self._buffer = np.full(
            (self.DISPLAY_CHANNELS, self.controller.config.buffer_samples),
            np.nan,
            dtype=np.float32,
        )
        self._write_index = 0
        self._filled = 0
        fs = float(max(1, self.controller.config.sample_rate))
        self._x_axis = (
            np.arange(self.controller.config.buffer_samples, dtype=np.float32) / fs
        )
        self._curves.clear()

        plot = self.plot_widget.getPlotItem()
        plot.clear()
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", "Channels / scaled uV")
        plot.setLabel("bottom", "Time in Window", "s")
        plot.setMouseEnabled(x=True, y=False)
        plot.setMenuEnabled(False)

        y_ticks = [(float(self._channel_offsets[i]), f"CH{i + 1}") for i in range(self.DISPLAY_CHANNELS)]
        plot.getAxis("left").setTicks([y_ticks])
        plot.getAxis("left").setStyle(autoExpandTextSpace=False)

        y_min = -self.TRACE_SPACING_UV
        y_max = float(self._channel_offsets[0] + self.TRACE_SPACING_UV)
        plot.setXRange(0.0, float(self.controller.config.display_seconds), padding=0.0)
        plot.setYRange(y_min, y_max, padding=0.02)
        self._plot_item = plot

        for i in range(self.DISPLAY_CHANNELS):
            curve = plot.plot(pen=pg.mkPen(self.TRACE_COLORS[i % len(self.TRACE_COLORS)], width=1.1))
            curve.setClipToView(True)
            # Keep true polyline shape for EEG traces (avoid peak-style bar rendering).
            curve.setDownsampling(auto=False)
            self._curves.append(curve)

        self.ch_label.setText("Channels: 16 (fixed)")

    def _on_timer(self) -> None:
        changed = False
        drained = 0
        while drained < self.controller.config.max_plot_drain_per_tick:
            try:
                sample = self.controller.plot_queue.get_nowait()
            except queue.Empty:
                break
            self._append_sample(sample.uV)
            self._rate_times.append(time.time())
            changed = True
            drained += 1

        self._trim_rate_window()
        if changed:
            self._update_plots()
        self._update_status_labels()

        if not self.controller.is_running():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._set_controls_enabled(True)

    def _append_sample(self, uV: list[float]) -> None:
        sample_vec = np.full((self.DISPLAY_CHANNELS,), np.nan, dtype=np.float32)
        if uV:
            src = np.asarray(uV, dtype=np.float32)
            n = min(src.size, self.DISPLAY_CHANNELS)
            sample_vec[:n] = src[:n]

        self._buffer[:, self._write_index] = sample_vec
        self._write_index = (self._write_index + 1) % self._buffer.shape[1]
        self._filled = min(self._filled + 1, self._buffer.shape[1])

    def _ordered_buffer(self) -> np.ndarray:
        if self._filled == 0:
            return self._buffer[:, :0]
        if self._filled < self._buffer.shape[1]:
            return self._buffer[:, : self._filled]
        idx = self._write_index
        return np.concatenate((self._buffer[:, idx:], self._buffer[:, :idx]), axis=1)

    def _update_plots(self) -> None:
        if self._filled == 0:
            return
        ordered = self._ordered_buffer()
        n = ordered.shape[1]
        if n == 0:
            return

        window_s = float(self.controller.config.display_seconds)
        fs = float(max(1, self.controller.config.sample_rate))
        if n <= 1:
            x = np.asarray([window_s], dtype=np.float32)
        else:
            duration = (n - 1) / fs
            shift = max(0.0, window_s - duration)
            x = shift + (np.arange(n, dtype=np.float32) / fs)

        if self._plot_item is not None:
            self._plot_item.setXRange(
                0.0, window_s, padding=0.0
            )

        for i, curve in enumerate(self._curves):
            y = ordered[i].copy()

            finite = np.isfinite(y)
            if np.any(finite):
                yf = y[finite]
                center = float(np.median(yf))
                dev = np.abs(yf - center)
                amp95 = float(np.percentile(dev, 95))
                if amp95 < 1e-9:
                    gain = 1.0
                else:
                    gain = self.DISPLAY_TARGET_UV / amp95
                    gain = float(
                        np.clip(gain, self.DISPLAY_MIN_GAIN, self.DISPLAY_MAX_GAIN)
                    )
                yf = (yf - center) * gain
                yf = np.clip(yf, -self.DISPLAY_CLIP_UV, self.DISPLAY_CLIP_UV)
                y[finite] = yf
            else:
                y.fill(0.0)
            curve.setData(x, y + self._channel_offsets[i], connect="finite")

    def _trim_rate_window(self) -> None:
        now = time.time()
        while self._rate_times and now - self._rate_times[0] > 2.0:
            self._rate_times.popleft()

    def _update_status_labels(self) -> None:
        snap = self.controller.stats.snapshot()
        connected = snap["connected"]
        self.connection_label.setText(
            f"Connection: {'connected' if connected else 'disconnected'}"
        )
        self.connection_label.setStyleSheet(
            "color: #1a7f37;" if connected else "color: #b42318;"
        )
        window_seconds = 2.0
        est_rate = len(self._rate_times) / window_seconds
        self.rate_label.setText(f"Rate: {est_rate:.1f} Hz")
        self.drop_label.setText(f"Packet loss: {snap['packet_drop_count']}")
        input_channels = 16
        input_mode = "forced"
        self.ch_label.setText(
            f"Channels: input {input_channels} ({input_mode}) / display {self.DISPLAY_CHANNELS}"
        )
        self.save_label.setText(f"Saved: {snap['saved_samples']}")
        csv_path = snap["output_csv"] or "-"
        self.path_label.setText(f"CSV: {csv_path}")
        last_error = snap["last_error"] or "-"
        if len(last_error) > 70:
            last_error = last_error[:67] + "..."
        self.error_label.setText(f"Last error: {last_error}")

    def closeEvent(self, event) -> None:
        self.controller.stop(wait=True)
        super().closeEvent(event)
